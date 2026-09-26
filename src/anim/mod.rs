//! Animation: keyframed timelines, curves, and playback.
//!
//! An [`Animation`] is a set of timelines that pose a [`Skeleton`] over time.
//! Most timelines interpolate one property with stepped / linear / Bezier
//! curves (see the `curve` module). [`AnimationState`] plays an animation on a
//! track.
//! Interpolation and blending re-implement Spine 4.3's behavior.
//!
//! Covered: every Spine 4.3 timeline plus multi-track playback with crossfade
//! mixing and a queue. The timelines are bone rotate / translate / scale / shear
//! and single-axis variants, bone inherit, IK / transform / path / physics /
//! slider mixes, slot color / alpha / two-color / attachment / draw-order,
//! draw order folders, deform, events, and sequences.

mod channels;
mod curve;
#[cfg(test)]
mod robustness;
mod state;
mod timeline;

#[cfg(feature = "json")]
pub(crate) use channels::Fallback;
pub(crate) use channels::{PATH_MIX, PATH_POSITION, PATH_SPACING, TRANSFORM_MIX};
pub use state::{AnimationState, TrackEntry};
pub(crate) use timeline::{
    compute_draw_order, sort_draw_order_moves, AttachmentTarget, AttachmentTimeline, BoneAxis,
    BoneTimeline, ConstraintTimeline, DeformTimeline, DrawOrderFolderTimeline, DrawOrderTimeline,
    EventTimeline, InheritTimeline, PhysicsProperty, PhysicsResetTimeline, SequenceTimeline,
    Timeline, GLOBAL_PHYSICS,
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

    /// The bones this animation's timelines change, once each.
    pub(crate) fn bones(&self) -> Vec<usize> {
        let mut bones: Vec<usize> = self.timelines.iter().filter_map(Timeline::bone).collect();
        bones.sort_unstable();
        bones.dedup();
        bones
    }

    /// Apply every timeline to `skeleton` over the window `(last_time, time]`
    /// (seconds), mixing with weight `alpha` from `from`. `add` selects additive
    /// blending. `last_time` is used only by the physics reset and event
    /// timelines. Timeline indices out of range for `skeleton` are skipped.
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
