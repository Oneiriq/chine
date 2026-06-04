//! Single source of truth for the multi-channel constraint-timeline layouts.
//!
//! A transform- or path-constraint mix timeline stores its channels in a fixed
//! order. Both loaders must agree on that order: the JSON loader matches keys by
//! name, the binary loader reads floats positionally, and [`super::timeline`]
//! reads them back in the same order when applying. Defining each layout once,
//! here, keeps the two loaders from drifting apart: the JSON loader takes the
//! names and setup defaults, while the binary loader derives its channel count
//! from the same slice's length.
//!
//! Each entry is `(json_key, setup_default)`: the key the JSON loader looks for,
//! and the value a channel takes when its key is absent. The binary format
//! always stores every channel explicitly, so it uses only the order and count.

/// Transform-constraint mix channels (Spine `TransformConstraintTimeline`).
pub(crate) const TRANSFORM_MIX: &[(&str, f32)] = &[
    ("mixRotate", 1.0),
    ("mixX", 1.0),
    ("mixY", 1.0),
    ("mixScaleX", 1.0),
    ("mixScaleY", 1.0),
    ("mixShearY", 1.0),
];

/// Path-constraint mix channels (`PathConstraintMixTimeline`).
pub(crate) const PATH_MIX: &[(&str, f32)] = &[("mixRotate", 1.0), ("mixX", 1.0), ("mixY", 1.0)];

/// Path-constraint position channel (`PathConstraintPositionTimeline`).
pub(crate) const PATH_POSITION: &[(&str, f32)] = &[("position", 0.0)];

/// Path-constraint spacing channel (`PathConstraintSpacingTimeline`).
pub(crate) const PATH_SPACING: &[(&str, f32)] = &[("spacing", 0.0)];
