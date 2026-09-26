//! Single source of truth for the multi-channel constraint-timeline layouts.
//!
//! A transform- or path-constraint mix timeline stores its channels in a fixed
//! order. Both loaders must agree on that order: the JSON loader matches keys by
//! name, the binary loader reads floats positionally, and [`super::timeline`]
//! reads them back in the same order when applying. Defining each layout once,
//! here, keeps the two loaders from drifting apart: the JSON loader takes the
//! names and fallbacks, while the binary loader derives its channel count from
//! the same slice's length.
//!
//! Each entry is `(json_key, fallback)`: the key the JSON loader looks for, and
//! the value a channel takes when its key is absent. The binary format always
//! stores every channel explicitly, so it uses only the order and count.

/// The value a JSON timeline channel takes when a keyframe omits its key.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Fallback {
    /// A fixed value.
    Value(f32),
    /// The same keyframe's value for an earlier channel (by index).
    Channel(usize),
}

/// Transform-constraint mix channels (Spine `TransformConstraintTimeline`).
/// An omitted `mixY` takes the key's `mixX`.
pub(crate) const TRANSFORM_MIX: &[(&str, Fallback)] = &[
    ("mixRotate", Fallback::Value(1.0)),
    ("mixX", Fallback::Value(1.0)),
    ("mixY", Fallback::Channel(1)),
    ("mixScaleX", Fallback::Value(1.0)),
    ("mixScaleY", Fallback::Value(1.0)),
    ("mixShearY", Fallback::Value(1.0)),
];

/// Path-constraint mix channels (`PathConstraintMixTimeline`). An omitted
/// `mixY` takes the key's `mixX`.
pub(crate) const PATH_MIX: &[(&str, Fallback)] = &[
    ("mixRotate", Fallback::Value(1.0)),
    ("mixX", Fallback::Value(1.0)),
    ("mixY", Fallback::Channel(1)),
];

/// Path-constraint position channel (`PathConstraintPositionTimeline`).
pub(crate) const PATH_POSITION: &[(&str, Fallback)] = &[("value", Fallback::Value(0.0))];

/// Path-constraint spacing channel (`PathConstraintSpacingTimeline`).
pub(crate) const PATH_SPACING: &[(&str, Fallback)] = &[("value", Fallback::Value(0.0))];
