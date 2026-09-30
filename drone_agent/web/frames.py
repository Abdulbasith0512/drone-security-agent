"""Video frame thumbnails for the dashboard, from explicitly registered videos only.

A stored observation's `frame_id` is the frame index in its source video (see video/ingest.py), so
the picture behind an observation is that exact frame. Nothing is stored: frames are decoded on
demand with the existing `VideoSource` / `read_frame_at`, re-encoded as JPEG and kept in a small
LRU.
"""

from collections import OrderedDict
from pathlib import Path

import cv2

from drone_agent.video import VideoError, VideoSource, read_frame_at

MIN_WIDTH, MAX_WIDTH, DEFAULT_WIDTH = 64, 960, 480
_CACHE_ITEMS = 128


class FrameNotFound(LookupError):
    """The requested frame is not in the registered video."""


class FrameProvider:
    def __init__(self, video: Path) -> None:
        self.source = VideoSource(video)  # validates the file once (and hashes it)
        self.meta = self.source.metadata
        self._cache: OrderedDict[tuple[int, int], bytes] = OrderedDict()

    @property
    def frame_count(self) -> int:
        return self.meta.frame_count

    def jpeg(self, frame_id: int, width: int = DEFAULT_WIDTH) -> bytes:
        if not 0 <= frame_id < self.meta.frame_count:
            raise FrameNotFound(f"frame {frame_id} is not in this video")
        width = max(MIN_WIDTH, min(int(width), MAX_WIDTH))
        key = (frame_id, width)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        try:
            frame = read_frame_at(self.source, frame_id / self.meta.fps)
        except VideoError as exc:
            raise FrameNotFound(str(exc)) from exc
        image = frame.image
        if image is None:
            raise FrameNotFound(f"frame {frame_id} could not be decoded")
        if image.shape[1] > width:
            height = max(1, round(image.shape[0] * width / image.shape[1]))
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 82])
        frame.release()
        if not ok:
            raise FrameNotFound(f"frame {frame_id} could not be encoded")
        data = encoded.tobytes()
        self._cache[key] = data
        while len(self._cache) > _CACHE_ITEMS:
            self._cache.popitem(last=False)
        return data
