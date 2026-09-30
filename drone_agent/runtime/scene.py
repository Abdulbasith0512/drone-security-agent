"""A cheap scene-change signal for SCRIPTED frames (which have no pixels).

Real video is compared pixel by pixel (video/gate.py). A scripted text frame has no image, so the
sampler is given a stand-in made of the two things the script supplies:

* the objects: a multiset of (type, activity, attributes), so repeated identical objects keep their
  count and "one person standing" differs from "two people running";
* the description text, compared after trimming and lower-casing.

It scores 0 when both are identical and 100 when no object is shared. A changed description with
identical objects scores `DESCRIPTION_ONLY_SCORE`: wording alone is weaker evidence than a changed
object, but it is still new supplied evidence, so the old observation is not carried over it.

It is the same kind of signal (was there a change?) but it is derived from the script, not from
pixels, so every trace and report labels it `scripted_scene_signature`. It says nothing about
real-world accuracy.
"""

from collections import Counter
from typing import NamedTuple

from drone_agent.scenario import ScenarioFrame

SIGNAL_NAME = "scripted_scene_signature"
# Score for "same objects, different wording". Above the sampler's default threshold (6).
DESCRIPTION_ONLY_SCORE = 50.0


class Signature(NamedTuple):
    # (class, activity, sorted attributes) -> how many such objects the frame shows
    objects: Counter[tuple[str, str, tuple[tuple[str, str], ...]]]
    description: str


def signature(frame: ScenarioFrame) -> Signature:
    return Signature(
        Counter(
            (o.object_type.value, o.activity or "", tuple(sorted(o.attributes.items())))
            for o in frame.objects
        ),
        " ".join(frame.description.lower().split()),
    )


def scene_score(reference: Signature | None, current: Signature) -> float:
    """0 = identical, 100 = nothing in common (multiset Jaccard x 100). No reference = 100."""
    if reference is None:
        return 100.0
    union = sum((reference.objects | current.objects).values())
    if union:
        shared = sum((reference.objects & current.objects).values())
        objects_score = round(100.0 * (1 - shared / union), 2)
    else:
        objects_score = 0.0
    if objects_score == 0.0 and reference.description != current.description:
        return DESCRIPTION_ONLY_SCORE
    return objects_score
