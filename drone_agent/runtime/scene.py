"""A cheap scene-change signal for SCRIPTED frames (which have no pixels).

Real video is compared pixel by pixel (video/gate.py). A scripted text frame has no image, so the
sampler is given a stand-in: the distance between the sets of (type, attributes) objects the frame
describes. It scores 0 when the scene is identical and 100 when nothing is shared. It is the same
kind of signal (was there a change?) but it is derived from the script, not from pixels, so every
trace and report labels it `scripted_scene_signature`. It says nothing about real-world accuracy.
"""

from drone_agent.scenario import ScenarioFrame

Signature = frozenset[tuple[str, tuple[tuple[str, str], ...]]]

SIGNAL_NAME = "scripted_scene_signature"


def signature(frame: ScenarioFrame) -> Signature:
    """The frame's objects as a set of (class, sorted attributes); activity is not part of it."""
    return frozenset(
        (o.object_type.value, tuple(sorted(o.attributes.items()))) for o in frame.objects
    )


def scene_score(reference: Signature | None, current: Signature) -> float:
    """0 = identical, 100 = nothing in common (Jaccard distance x 100). No reference = 100."""
    if reference is None:
        return 100.0
    union = reference | current
    if not union:
        return 0.0
    return round(100.0 * (1 - len(reference & current) / len(union)), 2)
