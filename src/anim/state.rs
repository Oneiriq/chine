//! Single-track animation playback.
//!
//! [`AnimationState`] advances one [`TrackEntry`] and applies it to a
//! [`Skeleton`]. Multi-track mixing and the animation queue arrive later; this
//! is the minimal player that drives one animation.

use std::sync::Arc;

use super::{Animation, MixFrom};
use crate::skel::Skeleton;

/// One playing animation on a track.
pub struct TrackEntry {
    animation: Arc<Animation>,
    /// Seconds elapsed on this track (before any loop wrap).
    pub track_time: f32,
    /// Playback speed multiplier applied in [`AnimationState::update`].
    pub time_scale: f32,
    /// Whether playback wraps at the animation's duration.
    pub looping: bool,
    /// Mix weight in `[0, 1]`.
    pub alpha: f32,
}

impl TrackEntry {
    fn new(animation: Arc<Animation>, looping: bool) -> Self {
        Self {
            animation,
            track_time: 0.0,
            time_scale: 1.0,
            looping,
            alpha: 1.0,
        }
    }

    /// The animation playing on this track.
    #[must_use]
    pub fn animation(&self) -> &Animation {
        &self.animation
    }

    /// The playback time fed to the animation (wrapped when looping).
    #[must_use]
    pub fn current_time(&self) -> f32 {
        let duration = self.animation.duration();
        if self.looping && duration > 0.0 {
            self.track_time % duration
        } else {
            self.track_time
        }
    }
}

/// A minimal single-track animation player.
#[derive(Default)]
pub struct AnimationState {
    track: Option<TrackEntry>,
}

impl AnimationState {
    /// An empty player.
    #[must_use]
    pub fn new() -> Self {
        Self { track: None }
    }

    /// Play `animation`, replacing any current track. Returns the new entry so
    /// the caller can tune `time_scale`, `alpha`, etc.
    pub fn set_animation(&mut self, animation: Arc<Animation>, looping: bool) -> &mut TrackEntry {
        self.track = Some(TrackEntry::new(animation, looping));
        self.track.as_mut().expect("just set")
    }

    /// Remove the current track.
    pub fn clear(&mut self) {
        self.track = None;
    }

    /// The current track, if any.
    #[must_use]
    pub fn track(&self) -> Option<&TrackEntry> {
        self.track.as_ref()
    }

    /// Advance playback by `delta` seconds.
    pub fn update(&mut self, delta: f32) {
        if let Some(t) = &mut self.track {
            t.track_time += delta * t.time_scale;
        }
    }

    /// Pose `skeleton` from the current track. Bones keyed by the animation are
    /// set from their setup pose plus the keyed value; bones the animation does
    /// not touch keep their current pose (reset to setup first for a clean
    /// result).
    pub fn apply(&self, skeleton: &mut Skeleton) {
        if let Some(t) = &self.track {
            t.animation
                .apply(skeleton, t.current_time(), t.alpha, MixFrom::Setup, false);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::anim::timeline::{BoneTimeline, Timeline};
    use crate::data::{BoneData, SkeletonData};

    fn spin() -> (Skeleton, Arc<Animation>) {
        let data = SkeletonData {
            bones: vec![BoneData {
                index: 0,
                name: "bone".into(),
                ..Default::default()
            }],
            ..Default::default()
        };
        let mut t = BoneTimeline::one_value(0, 2, 0);
        t.set_frame1(0, 0.0, 0.0);
        t.set_frame1(1, 1.0, 90.0);
        let anim = Animation::new("spin", 1.0, vec![Timeline::Rotate(t)]);
        (Skeleton::new(Arc::new(data)), Arc::new(anim))
    }

    #[test]
    fn plays_and_loops() {
        let (mut sk, anim) = spin();
        let mut state = AnimationState::new();
        state.set_animation(anim, true);

        state.update(0.5);
        state.apply(&mut sk);
        assert!((sk.bone(0).unwrap().rotation - 45.0).abs() < 1e-4);

        // track_time -> 1.5, wraps to 0.5 of a 1.0s loop -> same pose.
        state.update(1.0);
        state.apply(&mut sk);
        assert!((sk.bone(0).unwrap().rotation - 45.0).abs() < 1e-4);
    }

    #[test]
    fn time_scale_speeds_playback() {
        let (_sk, anim) = spin();
        let mut state = AnimationState::new();
        state.set_animation(anim, false).time_scale = 2.0;
        state.update(0.5);
        // 0.5s at 2x -> track_time 1.0.
        assert!((state.track().unwrap().current_time() - 1.0).abs() < 1e-4);
    }
}
