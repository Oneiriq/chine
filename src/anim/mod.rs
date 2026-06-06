//! Animation: keyframed timelines, curves, and playback.
//!
//! An [`Animation`] is a set of timelines that pose a [`Skeleton`] over time.
//! Each timeline interpolates one bone property with stepped / linear / Bezier
//! curves (see the `curve` module); [`AnimationState`] plays an animation on a
//! track.
//! Interpolation and blending are transcribed from Spine 4.3 for fidelity.
//!
//! Covered: every Spine 4.3 timeline (bone rotate / translate / scale / shear
//! and single-axis variants; IK / transform / path / physics / slider mixes;
//! slot color / alpha / two-color / attachment / draw-order; deform; events;
//! sequences) plus multi-track playback with crossfade mixing and a queue.

mod channels;
mod curve;
mod state;
mod timeline;

pub(crate) use channels::{PATH_MIX, PATH_POSITION, PATH_SPACING, TRANSFORM_MIX};
pub use state::{AnimationState, TrackEntry};
pub(crate) use timeline::{
    compute_draw_order, AttachmentTimeline, BoneAxis, BoneTimeline, ConstraintTimeline,
    DeformTimeline, DrawOrderTimeline, EventTimeline, PhysicsProperty, PhysicsResetTimeline,
    SequenceTimeline, Timeline, GLOBAL_PHYSICS,
};

use crate::skel::Skeleton;

/// Which value a timeline blends from (Spine 4.3 `MixFrom`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MixFrom {
    /// Blend from the bone's setup value (replaces the current pose).
    Setup,
    /// Blend from the current value, treating the first application as setup.
    First,
    /// Blend from the current value.
    Current,
}

/// A named set of timelines that pose a skeleton over a fixed duration.
#[derive(Debug, Clone)]
pub struct Animation {
    name: String,
    duration: f32,
    timelines: Vec<Timeline>,
}

impl Animation {
    /// Build an animation from its timelines. `duration` is the last keyframe
    /// time, in seconds.
    pub(crate) fn new(name: impl Into<String>, duration: f32, timelines: Vec<Timeline>) -> Self {
        Self {
            name: name.into(),
            duration,
            timelines,
        }
    }

    /// The animation's name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The animation's duration in seconds (its last keyframe time).
    #[must_use]
    pub fn duration(&self) -> f32 {
        self.duration
    }

    /// Apply every timeline to `skeleton` over the window `(last_time, time]`
    /// (seconds), mixing with weight `alpha` from `from`. `add` selects additive
    /// blending. `last_time` is used only by the physics reset timeline.
    pub fn apply(
        &self,
        skeleton: &mut Skeleton,
        last_time: f32,
        time: f32,
        alpha: f32,
        from: MixFrom,
        add: bool,
    ) {
        for timeline in &self.timelines {
            timeline.apply(skeleton, last_time, time, alpha, from, add, false);
        }
    }
}
