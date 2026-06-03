//! Multi-track animation playback with crossfade mixing.
//!
//! [`AnimationState`] holds a stack of tracks; each [`TrackEntry`] plays one
//! animation, and higher tracks layer over lower ones. Setting a new animation
//! on a track while [`AnimationState::default_mix`] is non-zero crossfades from
//! the previous animation over that duration.

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
    // The previous applied time, for keyframe-crossing timelines (physics reset).
    last_time: f32,
    // Crossfade: how long to mix in over `mixing_from`, and how far along.
    mix_duration: f32,
    mix_time: f32,
    // The animation being mixed out of (the previous entry on this track).
    mixing_from: Option<Box<TrackEntry>>,
    // The queued animation, and the track time at which it takes over.
    next: Option<Box<TrackEntry>>,
    delay: f32,
}

impl TrackEntry {
    fn new(animation: Arc<Animation>, looping: bool) -> Self {
        Self {
            animation,
            track_time: 0.0,
            time_scale: 1.0,
            looping,
            alpha: 1.0,
            last_time: -1.0,
            mix_duration: 0.0,
            mix_time: 0.0,
            mixing_from: None,
            next: None,
            delay: 0.0,
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

    /// Advance this entry, and the entry it is mixing from, by `delta` seconds,
    /// clearing the crossfade once it completes.
    fn advance(&mut self, delta: f32) {
        self.track_time += delta * self.time_scale;
        if let Some(from) = self.mixing_from.as_mut() {
            from.advance(delta);
            self.mix_time += delta;
            if self.mix_duration <= 0.0 || self.mix_time >= self.mix_duration {
                self.mixing_from = None;
            }
        }
    }

    /// The crossfade proportion in `[0, 1]` (`1` once mixing completes).
    fn mix(&self) -> f32 {
        if self.mix_duration > 0.0 {
            (self.mix_time / self.mix_duration).clamp(0.0, 1.0)
        } else {
            1.0
        }
    }

    /// Apply this entry to `skeleton`. `base` selects setup-pose blending (the
    /// lowest track) versus layering over the current pose.
    fn apply(&mut self, skeleton: &mut Skeleton, base: bool) {
        let base_blend = if base {
            MixFrom::Setup
        } else {
            MixFrom::Current
        };
        if let Some(from) = self.mixing_from.as_mut() {
            // Establish the outgoing pose, then blend the incoming one over it.
            let from_time = from.current_time();
            from.animation.apply(
                skeleton,
                from.last_time,
                from_time,
                from.alpha,
                base_blend,
                false,
            );
            from.last_time = from_time;
            let time = self.current_time();
            self.animation.apply(
                skeleton,
                self.last_time,
                time,
                self.alpha * self.mix(),
                MixFrom::Current,
                false,
            );
            self.last_time = time;
        } else {
            let time = self.current_time();
            self.animation.apply(
                skeleton,
                self.last_time,
                time,
                self.alpha,
                base_blend,
                false,
            );
            self.last_time = time;
        }
    }
}

/// A multi-track animation player with crossfade mixing.
#[derive(Default)]
pub struct AnimationState {
    tracks: Vec<Option<TrackEntry>>,
    /// Crossfade duration, in seconds, applied when an animation replaces
    /// another on the same track (`0` switches instantly).
    pub default_mix: f32,
}

impl AnimationState {
    /// An empty player.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Play `animation` on track 0, replacing any current animation there
    /// (crossfading over [`Self::default_mix`] when non-zero). Returns the new
    /// entry so the caller can tune `time_scale`, `alpha`, etc.
    pub fn set_animation(&mut self, animation: Arc<Animation>, looping: bool) -> &mut TrackEntry {
        self.set_animation_on(0, animation, looping)
    }

    /// Play `animation` on `track`, layering over lower tracks and crossfading
    /// from the track's current animation over [`Self::default_mix`].
    pub fn set_animation_on(
        &mut self,
        track: usize,
        animation: Arc<Animation>,
        looping: bool,
    ) -> &mut TrackEntry {
        while self.tracks.len() <= track {
            self.tracks.push(None);
        }
        let mut entry = TrackEntry::new(animation, looping);
        if self.default_mix > 0.0 {
            if let Some(mut current) = self.tracks[track].take() {
                current.mixing_from = None; // collapse any in-progress crossfade
                current.next = None; // interrupt the queue
                entry.mix_duration = self.default_mix;
                entry.mixing_from = Some(Box::new(current));
            }
        }
        self.tracks[track] = Some(entry);
        self.tracks[track].as_mut().expect("just set")
    }

    /// Queue `animation` on track 0 after the current animation and any already
    /// queued; it crossfades in over [`Self::default_mix`] as the previous one
    /// ends.
    pub fn add_animation(&mut self, animation: Arc<Animation>, looping: bool) {
        self.add_animation_on(0, animation, looping);
    }

    /// Queue `animation` on `track` after the current animation and any already
    /// queued. If the track is empty it plays immediately.
    pub fn add_animation_on(&mut self, track: usize, animation: Arc<Animation>, looping: bool) {
        let default_mix = self.default_mix;
        while self.tracks.len() <= track {
            self.tracks.push(None);
        }
        let entry = TrackEntry::new(animation, looping);
        if let Some(current) = self.tracks[track].as_mut() {
            let mut node = current;
            while node.next.is_some() {
                node = node.next.as_deref_mut().expect("checked some");
            }
            node.delay = (node.animation.duration() - default_mix).max(0.0);
            node.next = Some(Box::new(entry));
        } else {
            self.tracks[track] = Some(entry);
        }
    }

    /// Remove every track.
    pub fn clear(&mut self) {
        self.tracks.clear();
    }

    /// Remove the animation on `track`.
    pub fn clear_track(&mut self, track: usize) {
        if let Some(slot) = self.tracks.get_mut(track) {
            *slot = None;
        }
    }

    /// The entry on `track`, if any.
    #[must_use]
    pub fn track(&self, track: usize) -> Option<&TrackEntry> {
        self.tracks.get(track).and_then(Option::as_ref)
    }

    /// Advance every track by `delta` seconds.
    pub fn update(&mut self, delta: f32) {
        let default_mix = self.default_mix;
        for slot in &mut self.tracks {
            if let Some(entry) = slot.as_mut() {
                entry.advance(delta);
            }
            promote(slot, default_mix);
        }
    }

    /// Pose `skeleton` from every track in order: track 0 from the setup pose,
    /// higher tracks layered on top. Reset bones to setup first for a clean
    /// result.
    pub fn apply(&mut self, skeleton: &mut Skeleton) {
        skeleton.clear_events();
        for (i, slot) in self.tracks.iter_mut().enumerate() {
            if let Some(entry) = slot {
                entry.apply(skeleton, i == 0);
            }
        }
    }
}

/// Promote a track's queued entry to current once the current entry reaches its
/// delay, crossfading the previous current out over `default_mix`.
fn promote(slot: &mut Option<TrackEntry>, default_mix: f32) {
    let ready = slot
        .as_ref()
        .is_some_and(|c| c.next.is_some() && c.track_time >= c.delay);
    if !ready {
        return;
    }
    let mut current = slot.take().expect("checked some");
    let mut next = current.next.take().expect("checked some");
    current.mixing_from = None;
    if default_mix > 0.0 {
        next.mix_duration = default_mix;
        next.mix_time = 0.0;
        next.mixing_from = Some(Box::new(current));
    }
    *slot = Some(*next);
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

    /// A one-bone skeleton plus an animation that holds the bone at `rotation`.
    fn hold(rotation: f32) -> Arc<Animation> {
        let mut t = BoneTimeline::one_value(0, 1, 0);
        t.set_frame1(0, 0.0, rotation);
        Arc::new(Animation::new("hold", 0.0, vec![Timeline::Rotate(t)]))
    }

    fn one_bone() -> Skeleton {
        let data = SkeletonData {
            bones: vec![BoneData {
                index: 0,
                name: "bone".into(),
                ..Default::default()
            }],
            ..Default::default()
        };
        Skeleton::new(Arc::new(data))
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
        assert!((state.track(0).unwrap().current_time() - 1.0).abs() < 1e-4);
    }

    #[test]
    fn crossfade_blends_between_animations() {
        let mut sk = one_bone();
        let mut state = AnimationState::new();
        state.default_mix = 1.0; // a one-second crossfade

        // Start holding rotation 0.
        state.set_animation(hold(0.0), true);
        sk.set_bones_to_setup_pose();
        state.apply(&mut sk);
        assert!((sk.bone(0).unwrap().rotation - 0.0).abs() < 1e-4);

        // Switch to rotation 90 with a crossfade; halfway is ~45.
        state.set_animation(hold(90.0), true);
        state.update(0.5);
        sk.set_bones_to_setup_pose();
        state.apply(&mut sk);
        let mid = sk.bone(0).unwrap().rotation;
        assert!(
            (mid - 45.0).abs() < 2.0,
            "expected mid-crossfade ~45, got {mid}"
        );

        // Past the mix duration the crossfade completes at 90.
        state.update(0.6);
        sk.set_bones_to_setup_pose();
        state.apply(&mut sk);
        let done = sk.bone(0).unwrap().rotation;
        assert!(
            (done - 90.0).abs() < 1e-3,
            "expected 90 after mix, got {done}"
        );
    }

    #[test]
    fn higher_tracks_layer_over_lower() {
        let mut sk = one_bone();
        let mut state = AnimationState::new();
        // Track 0 holds 100; track 1 layers 20 at half weight over it.
        state.set_animation_on(0, hold(100.0), true);
        state.set_animation_on(1, hold(20.0), true).alpha = 0.5;
        sk.set_bones_to_setup_pose();
        state.apply(&mut sk);
        // base 100, then blended halfway toward 20 -> 60.
        let r = sk.bone(0).unwrap().rotation;
        assert!((r - 60.0).abs() < 1e-3, "expected layered 60, got {r}");
    }

    #[test]
    fn queued_animation_transitions_after_the_first() {
        let mut sk = one_bone();
        let mut state = AnimationState::new();
        state.default_mix = 0.2;
        // A holds rotation 0 for one second (duration 1).
        let mut a = BoneTimeline::one_value(0, 2, 0);
        a.set_frame1(0, 0.0, 0.0);
        a.set_frame1(1, 1.0, 0.0);
        state.set_animation(
            Arc::new(Animation::new("a", 1.0, vec![Timeline::Rotate(a)])),
            false,
        );
        // Queue B (holds rotation 90) to follow A.
        state.add_animation(hold(90.0), false);

        // Before A's delay (1.0 - 0.2 = 0.8), still on A.
        state.update(0.5);
        sk.set_bones_to_setup_pose();
        state.apply(&mut sk);
        assert!(
            (sk.bone(0).unwrap().rotation - 0.0).abs() < 1e-3,
            "should still be A"
        );

        // Cross the delay (promotes B), then finish the crossfade.
        state.update(0.4);
        state.update(0.3);
        sk.set_bones_to_setup_pose();
        state.apply(&mut sk);
        let r = sk.bone(0).unwrap().rotation;
        assert!((r - 90.0).abs() < 1e-3, "expected B (90), got {r}");
    }
}
