"""The scale axis: what metric reference, if any, is visible in frame.

This axis is deliberately **not produced** for this corpus, and the module
exists to say so precisely.  Ruling R3 surveyed a stratified 52/379 sample
spanning all 18 task-by-view cells and resolved exact dimensional identity in
**0 of 52**: the task apparatus is widely visible, every fallback
(anthropometrics, furniture, rig baseline, calibration target, audio
time-of-flight, skeletal priors) is absent rather than imprecise, and the best
conditional route floors at 17.7% before lens distortion.

The negative is *sampled*, so no asset may publish a measured ``none``.  A
sampled negative published as a measurement is certainty the survey never
bought.  With no manifest entry the axis stays absent, every asset keeps its
``scale_unmeasured`` flag, and the distinction between "surveyed and found
nothing" and "never surveyed" survives into the published artifact.

The alphabets below are therefore the contract an exhaustive survey would fill,
pinned here so a future producer cannot invent its own vocabulary.
"""

from __future__ import annotations

AXIS = "scale"

# The sentinel is a class, not an absence: a row saying "this frame was searched
# and holds no reference" is a measurement, and an empty cell is not.
NO_REFERENCE = "none"

# Object classes the survey looked for.  The task apparatus leads because it is
# what the corpus actually shows; the rest are the fallback routes R3 priced.
SCALE_CLASSES: frozenset[str] = frozenset(
    {
        NO_REFERENCE,
        "closure",
        "coin",
        "vessel",
        "key",
        "nut",
        "peg",
        "anthropometric",
        "furniture",
        "calibration_target",
    }
)

# Confidence is a token, never a number.  A decimal would invite arithmetic on
# what is an identification verdict: knowing an object is a coin fixes no
# dimension, because coin diameters differ by denomination and by country.
CLASS_ONLY = "class_only"
VARIANT_VERIFIED = "variant_verified"
DIMENSION_VERIFIED = "dimension_verified"

SCALE_CONFIDENCES: frozenset[str] = frozenset(
    {NO_REFERENCE, CLASS_ONLY, VARIANT_VERIFIED, DIMENSION_VERIFIED}
)
