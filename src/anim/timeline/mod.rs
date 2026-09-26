//! Animation timelines: bone, constraint, slot, draw order, event, deform, and
//! sequence.
//!
//! Curve timelines key their values over time via a [`Curve`]. The apply logic
//! re-implements Spine 4.3's: rotate / translate / shear values are **added**
//! to the setup pose, scale values **multiply** it (with sign-adjusted mixing),
//! and constraint and slot values replace it. The `apply` module holds the
//! bone, constraint, physics reset, and event timelines, and the `slot` module
//! holds the slot timelines.

use std::sync::Arc;

use super::curve::{absolute_value_with, Curve};
use super::MixFrom;
use crate::attach::{Attachment, AttachmentKey, MeshAttachment};
use crate::constraint::physics::{PhysicsConstraint, PhysicsConstraintData};
use crate::data::{Color, Inherit};
use crate::event::Event;
use crate::skel::Skeleton;

/// A bone index plus its keyframe curve. The [`Timeline`] variant selects how
/// the curve value combines with the bone's setup pose.
#[derive(Debug, Clone)]
pub(crate) struct BoneTimeline {
    bone: usize,
    curve: Curve,
}

impl BoneTimeline {
    /// A one-value timeline (rotate) for `bone`.
    pub(crate) fn one_value(bone: usize, frame_count: usize, bezier_count: usize) -> Self {
        Self {
            bone,
            curve: Curve::new(frame_count, bezier_count, 2),
        }
    }

    /// A two-value timeline (translate / scale) for `bone`.
    pub(crate) fn two_value(bone: usize, frame_count: usize, bezier_count: usize) -> Self {
        Self {
            bone,
            curve: Curve::new(frame_count, bezier_count, 3),
        }
    }

    /// Set a one-value frame.
    pub(crate) fn set_frame1(&mut self, frame: usize, time: f32, value: f32) {
        self.curve.set_frame1(frame, time, value);
    }

    /// Set a two-value frame.
    pub(crate) fn set_frame2(&mut self, frame: usize, time: f32, value1: f32, value2: f32) {
        self.curve.set_frame2(frame, time, value1, value2);
    }

    /// Mark `frame` as stepped.
    pub(crate) fn set_stepped(&mut self, frame: usize) {
        self.curve.set_stepped(frame);
    }

    /// Store a Bezier segment table for `frame`'s value ordinal `value`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn set_bezier(
        &mut self,
        bezier: usize,
        frame: usize,
        value: usize,
        time1: f32,
        value1: f32,
        cx1: f32,
        cy1: f32,
        cx2: f32,
        cy2: f32,
        time2: f32,
        value2: f32,
    ) {
        self.curve.set_bezier(
            bezier, frame, value, time1, value1, cx1, cy1, cx2, cy2, time2, value2,
        );
    }
}

/// A keyframed set of constraint-mix channels (IK / transform / path). The
/// [`Timeline`] variant selects which constraint and channels it drives.
#[derive(Debug, Clone)]
pub(crate) struct ConstraintTimeline {
    constraint: usize,
    curve: Curve,
}

impl ConstraintTimeline {
    /// A constraint timeline for `constraint` with `entries` floats per frame
    /// (1 time + N channels).
    pub(crate) fn new(
        constraint: usize,
        frame_count: usize,
        bezier_count: usize,
        entries: usize,
    ) -> Self {
        Self {
            constraint,
            curve: Curve::new(frame_count, bezier_count, entries),
        }
    }

    /// Set a frame: `time` followed by one value per channel.
    pub(crate) fn set_frame(&mut self, frame: usize, time: f32, values: &[f32]) {
        self.curve.set_frame_n(frame, time, values);
    }

    /// Mark `frame` as stepped.
    pub(crate) fn set_stepped(&mut self, frame: usize) {
        self.curve.set_stepped(frame);
    }

    /// Store a Bezier segment table for `frame`'s channel `value`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn set_bezier(
        &mut self,
        bezier: usize,
        frame: usize,
        value: usize,
        time1: f32,
        value1: f32,
        cx1: f32,
        cy1: f32,
        cx2: f32,
        cy2: f32,
        time2: f32,
        value2: f32,
    ) {
        self.curve.set_bezier(
            bezier, frame, value, time1, value1, cx1, cy1, cx2, cy2, time2, value2,
        );
    }
}

/// A physics constraint reset timeline: keyframe times that, when crossed, reset
/// the target constraint's simulation state (or every physics constraint when
/// the target is [`GLOBAL_PHYSICS`]).
#[derive(Debug, Clone)]
pub(crate) struct PhysicsResetTimeline {
    constraint: usize,
    times: Vec<f32>,
}

impl PhysicsResetTimeline {
    /// A reset timeline for `constraint` firing at each time in `times`.
    pub(crate) fn new(constraint: usize, times: Vec<f32>) -> Self {
        Self { constraint, times }
    }
}

/// A bone inherit timeline: stepped `(time, mode)` pairs setting how a bone
/// inherits its parent's transform.
#[derive(Debug, Clone)]
pub(crate) struct InheritTimeline {
    bone: usize,
    times: Vec<f32>,
    modes: Vec<Inherit>,
}

impl InheritTimeline {
    /// An inherit timeline for `bone`, one mode per keyframe time.
    pub(crate) fn new(bone: usize, times: Vec<f32>, modes: Vec<Inherit>) -> Self {
        Self { bone, times, modes }
    }
}

/// A slot attachment timeline: stepped `(time, name)` pairs selecting which
/// attachment a slot shows (`None` hides it).
#[derive(Debug, Clone)]
pub(crate) struct AttachmentTimeline {
    slot: usize,
    times: Vec<f32>,
    names: Vec<Option<String>>,
}

impl AttachmentTimeline {
    /// An attachment timeline for `slot`.
    pub(crate) fn new(slot: usize, times: Vec<f32>, names: Vec<Option<String>>) -> Self {
        Self { slot, times, names }
    }
}

/// A draw-order timeline: stepped per-keyframe slot orderings (each a full
/// permutation of slot indices, back to front).
#[derive(Debug, Clone)]
pub(crate) struct DrawOrderTimeline {
    times: Vec<f32>,
    orders: Vec<Vec<usize>>,
}

impl DrawOrderTimeline {
    /// A draw-order timeline from per-keyframe orderings.
    pub(crate) fn new(times: Vec<f32>, orders: Vec<Vec<usize>>) -> Self {
        Self { times, orders }
    }
}

/// Compute a draw order from slot offsets, mirroring Spine: each listed slot
/// moves by its offset, and the rest keep their relative order. Shared by the JSON
/// and binary loaders.
///
/// Spine never exports a slot index past `slot_count`, a slot listed twice, or
/// a move outside the draw order. Such entries are skipped here. A position no
/// slot fills keeps `usize::MAX`, which the renderer skips.
pub(crate) fn compute_draw_order(slot_count: usize, offsets: &mut [(usize, i32)]) -> Vec<usize> {
    offsets.sort_by_key(|(slot, _)| *slot);
    let mut draw_order = vec![usize::MAX; slot_count];
    let mut unchanged = Vec::with_capacity(slot_count.saturating_sub(offsets.len()));
    let mut original_index = 0;
    for &(slot_index, offset) in offsets.iter() {
        if slot_index >= slot_count || slot_index < original_index {
            continue;
        }
        unchanged.extend(original_index..slot_index);
        let pos = i64::try_from(slot_index)
            .ok()
            .and_then(|slot| slot.checked_add(i64::from(offset)))
            .and_then(|pos| usize::try_from(pos).ok());
        if let Some(entry) = pos.and_then(|pos| draw_order.get_mut(pos)) {
            *entry = slot_index;
        }
        original_index = slot_index + 1;
    }
    unchanged.extend(original_index..slot_count);
    for entry in draw_order.iter_mut().rev() {
        if *entry == usize::MAX {
            if let Some(slot) = unchanged.pop() {
                *entry = slot;
            }
        }
    }
    draw_order
}

/// An event timeline: keyframe times and the [`Event`] fired at each (with
/// keyframe-overridden values resolved at load time).
#[derive(Debug, Clone)]
pub(crate) struct EventTimeline {
    times: Vec<f32>,
    events: Vec<Event>,
}

impl EventTimeline {
    /// An event timeline from keyframe times and their resolved events.
    pub(crate) fn new(times: Vec<f32>, events: Vec<Event>) -> Self {
        Self { times, events }
    }
}

/// The attachment a deform or sequence timeline drives, and the slots it
/// reaches.
///
/// The timeline applies to its own slot and to each timeline slot, wherever
/// the slot currently shows an attachment that takes its timelines from
/// `key`: the attachment at `key` itself, or a linked mesh inheriting its
/// timelines.
#[derive(Debug, Clone)]
pub(crate) struct AttachmentTarget {
    key: AttachmentKey,
    timeline_slots: Arc<[usize]>,
}

impl AttachmentTarget {
    /// A target for the attachment at `key`, also reaching `timeline_slots`.
    pub(crate) fn new(key: AttachmentKey, timeline_slots: Arc<[usize]>) -> Self {
        Self {
            key,
            timeline_slots,
        }
    }

    /// The slots the timeline applies to: its own, then its timeline slots.
    fn slots(&self) -> impl Iterator<Item = usize> + '_ {
        std::iter::once(self.key.slot).chain(self.timeline_slots.iter().copied())
    }
}

/// A sequence (flipbook) timeline: per keyframe a time, a packed mode-and-index,
/// and a delay. The shown region index advances from the active keyframe by the
/// elapsed time over the delay, wrapped per the sequence mode over the regions
/// of the sequence the slot shows.
#[derive(Debug, Clone)]
pub(crate) struct SequenceTimeline {
    target: AttachmentTarget,
    times: Vec<f32>,
    mode_and_index: Vec<u32>,
    delays: Vec<f32>,
}

impl SequenceTimeline {
    /// A sequence timeline for the attachment `target` names.
    pub(crate) fn new(
        target: AttachmentTarget,
        times: Vec<f32>,
        mode_and_index: Vec<u32>,
        delays: Vec<f32>,
    ) -> Self {
        Self {
            target,
            times,
            mode_and_index,
            delays,
        }
    }
}

/// A mesh deform timeline: per-keyframe vertex offsets. For an unweighted mesh
/// they add to the setup vertices. For a weighted mesh the setup is zero and the
/// offsets add per-influence in `compute_vertices`. Only applies while a slot
/// shows the attachment `target` names.
#[derive(Debug, Clone)]
pub(crate) struct DeformTimeline {
    target: AttachmentTarget,
    setup: Vec<f32>,
    times: Vec<f32>,
    frames: Vec<Vec<f32>>,
    curve: Curve,
}

impl DeformTimeline {
    /// A deform timeline for the attachment `target` names, with setup
    /// vertices, the per-keyframe offset frames, and room for `bezier_count`
    /// Bezier segments.
    pub(crate) fn new(
        target: AttachmentTarget,
        setup: Vec<f32>,
        times: Vec<f32>,
        frames: Vec<Vec<f32>>,
        bezier_count: usize,
    ) -> Self {
        let mut curve = Curve::new(times.len(), bezier_count, 2);
        for (i, &t) in times.iter().enumerate() {
            curve.set_frame1(i, t, 0.0);
        }
        Self {
            target,
            setup,
            times,
            frames,
            curve,
        }
    }

    /// Mark `frame` as stepped (the deform snaps to the earlier keyframe).
    pub(crate) fn set_stepped(&mut self, frame: usize) {
        self.curve.set_stepped(frame);
    }

    /// Store a Bezier segment driving the interpolation percent at `frame`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn set_bezier(
        &mut self,
        bezier: usize,
        frame: usize,
        value: usize,
        time1: f32,
        value1: f32,
        cx1: f32,
        cy1: f32,
        cx2: f32,
        cy2: f32,
        time2: f32,
        value2: f32,
    ) {
        self.curve.set_bezier(
            bezier, frame, value, time1, value1, cx1, cy1, cx2, cy2, time2, value2,
        );
    }

    /// The curve-aware interpolation percent at `time`.
    fn percent(&self, time: f32) -> f32 {
        self.curve.percent(time)
    }
}

/// Sentinel constraint index marking a physics timeline as global: it drives
/// every physics constraint whose matching global flag is set.
pub(crate) const GLOBAL_PHYSICS: usize = usize::MAX;

/// Which tunable a physics constraint timeline drives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PhysicsProperty {
    /// Inertia.
    Inertia,
    /// Spring strength.
    Strength,
    /// Damping.
    Damping,
    /// Mass (animated as a mass value, stored inverted).
    Mass,
    /// Wind (blended additively).
    Wind,
    /// Gravity (blended additively).
    Gravity,
    /// Mix.
    Mix,
}

/// A single bone axis driven by a one-value timeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BoneAxis {
    /// Local x translation.
    TranslateX,
    /// Local y translation.
    TranslateY,
    /// Local x scale.
    ScaleX,
    /// Local y scale.
    ScaleY,
    /// Local x shear.
    ShearX,
    /// Local y shear.
    ShearY,
}

/// A keyframed animation channel: one bone property, constraint mix, or slot
/// property, or the draw order, events, or a mesh deform.
#[derive(Debug, Clone)]
pub(crate) enum Timeline {
    /// Local rotation (degrees).
    Rotate(BoneTimeline),
    /// Local translation (x, y).
    Translate(BoneTimeline),
    /// Local scale (x, y).
    Scale(BoneTimeline),
    /// Local shear (x, y).
    Shear(BoneTimeline),
    /// A single bone axis (translateX/Y, scaleX/Y, shearX/Y).
    BoneAxis(BoneTimeline, BoneAxis),
    /// How a bone inherits its parent's transform (stepped).
    Inherit(InheritTimeline),
    /// IK constraint mix / softness / bend / compress / stretch.
    Ik(ConstraintTimeline),
    /// Transform constraint mixes (rotate / x / y / scaleX / scaleY / shearY).
    TransformMix(ConstraintTimeline),
    /// Path constraint position.
    PathPosition(ConstraintTimeline),
    /// Path constraint spacing.
    PathSpacing(ConstraintTimeline),
    /// Path constraint mixes (rotate / x / y).
    PathMix(ConstraintTimeline),
    /// Physics constraint tunable (the selected [`PhysicsProperty`]).
    Physics(ConstraintTimeline, PhysicsProperty),
    /// Physics constraint reset, fired on keyframe crossings.
    PhysicsReset(PhysicsResetTimeline),
    /// Slider scrub time (drives a bone-less slider's animation playhead).
    SliderTime(ConstraintTimeline),
    /// Slider mix.
    SliderMix(ConstraintTimeline),
    /// Slot tint color. The index is the slot, and the flag is whether alpha is keyed
    /// (RGBA vs RGB).
    SlotColor(ConstraintTimeline, bool),
    /// Slot tint alpha only (the slot color's alpha channel).
    SlotAlpha(ConstraintTimeline),
    /// Slot two-color (light + dark). The flag is whether the light has alpha
    /// (RGBA2 vs RGB2).
    SlotTwoColor(ConstraintTimeline, bool),
    /// Slot attachment swap (stepped).
    Attachment(AttachmentTimeline),
    /// Slot draw order (stepped permutations).
    DrawOrder(DrawOrderTimeline),
    /// Animation events fired on keyframe crossings.
    Event(EventTimeline),
    /// Mesh deform (per-vertex offsets).
    Deform(DeformTimeline),
    /// Slot sequence (flipbook) frame index.
    Sequence(SequenceTimeline),
}

impl Timeline {
    /// Apply this timeline to `skeleton` over the window `(last_time, time]`.
    /// `from`, `add`, and `out` follow Spine's mix semantics. `out` only affects
    /// scale, IK, and inherit. `last_time` is used only by the physics reset and event
    /// timelines.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn apply(
        &self,
        skeleton: &mut Skeleton,
        last_time: f32,
        time: f32,
        alpha: f32,
        from: MixFrom,
        add: bool,
        out: bool,
    ) {
        match self {
            Timeline::Rotate(t) => apply_rotate(t, skeleton, time, alpha, from, add),
            Timeline::Translate(t) => apply_translate(t, skeleton, time, alpha, from, add),
            Timeline::Scale(t) => apply_scale(t, skeleton, time, alpha, from, add, out),
            Timeline::Shear(t) => apply_shear(t, skeleton, time, alpha, from, add),
            Timeline::BoneAxis(t, axis) => {
                apply_bone_axis(t, *axis, skeleton, time, alpha, from, add, out);
            }
            Timeline::Inherit(t) => apply_inherit(t, skeleton, time, from, out),
            Timeline::Ik(t) => apply_ik(t, skeleton, time, alpha, from, out),
            Timeline::TransformMix(t) => apply_transform_mix(t, skeleton, time, alpha, from, add),
            Timeline::PathPosition(t) => apply_path_position(t, skeleton, time, alpha, from, add),
            Timeline::PathSpacing(t) => apply_path_spacing(t, skeleton, time, alpha, from),
            Timeline::PathMix(t) => apply_path_mix(t, skeleton, time, alpha, from, add),
            Timeline::Physics(t, property) => {
                apply_physics(t, *property, skeleton, time, alpha, from, add);
            }
            Timeline::PhysicsReset(t) => apply_physics_reset(t, skeleton, last_time, time),
            Timeline::SliderTime(t) => apply_slider_time(t, skeleton, time, alpha, from, add),
            Timeline::SliderMix(t) => apply_slider_mix(t, skeleton, time, alpha, from, add),
            Timeline::SlotColor(t, has_alpha) => {
                apply_slot_color(t, *has_alpha, skeleton, time, alpha, from, add);
            }
            Timeline::SlotAlpha(t) => apply_slot_alpha(t, skeleton, time, alpha, from, add),
            Timeline::SlotTwoColor(t, light_alpha) => {
                apply_slot_two_color(t, *light_alpha, skeleton, time, alpha, from, add);
            }
            Timeline::Attachment(t) => apply_attachment(t, skeleton, time, from),
            Timeline::DrawOrder(t) => apply_draw_order(t, skeleton, time),
            Timeline::Event(t) => apply_event(t, skeleton, last_time, time),
            Timeline::Deform(t) => apply_deform(t, skeleton, time, alpha, from),
            Timeline::Sequence(t) => apply_sequence(t, skeleton, time),
        }
    }
}

mod apply;
mod slot;
use apply::*;
use slot::*;

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::data::{BoneData, Inherit, SkeletonData};

    fn skeleton(setup_rotation: f32) -> Skeleton {
        let data = SkeletonData {
            bones: vec![BoneData {
                index: 0,
                name: "bone".into(),
                rotation: setup_rotation,
                ..Default::default()
            }],
            ..Default::default()
        };
        Skeleton::new(Arc::new(data))
    }

    fn rotation(sk: &Skeleton) -> f32 {
        sk.bone(0).unwrap().rotation
    }

    #[test]
    fn rotate_adds_curve_value_to_setup() {
        let mut sk = skeleton(0.0);
        let mut t = BoneTimeline::one_value(0, 2, 0);
        t.set_frame1(0, 0.0, 0.0);
        t.set_frame1(1, 1.0, 90.0);
        Timeline::Rotate(t).apply(&mut sk, -1.0, 0.5, 1.0, MixFrom::Setup, false, false);
        assert!((rotation(&sk) - 45.0).abs() < 1e-4);
    }

    #[test]
    fn rotate_respects_setup_offset_and_alpha() {
        // Setup rotation 30. Animation keys +60 at full. Mix at half weight.
        let mut sk = skeleton(30.0);
        let mut t = BoneTimeline::one_value(0, 2, 0);
        t.set_frame1(0, 0.0, 0.0);
        t.set_frame1(1, 1.0, 60.0);
        Timeline::Rotate(t).apply(&mut sk, -1.0, 1.0, 0.5, MixFrom::Setup, false, false);
        // setup + value*alpha = 30 + 60*0.5 = 60.
        assert!((rotation(&sk) - 60.0).abs() < 1e-4);
    }

    #[test]
    fn before_first_frame_setup_resets_to_setup() {
        let mut sk = skeleton(30.0);
        sk.bone_mut(0).unwrap().rotation = 99.0;
        let mut t = BoneTimeline::one_value(0, 2, 0);
        t.set_frame1(0, 1.0, 0.0);
        t.set_frame1(1, 2.0, 90.0);
        Timeline::Rotate(t).apply(&mut sk, -1.0, 0.0, 1.0, MixFrom::Setup, false, false);
        assert!((rotation(&sk) - 30.0).abs() < 1e-4);
    }

    #[test]
    fn translate_adds_offsets_to_setup_position() {
        let mut sk = skeleton(0.0);
        let mut t = BoneTimeline::two_value(0, 2, 0);
        t.set_frame2(0, 0.0, 0.0, 0.0);
        t.set_frame2(1, 1.0, 10.0, 20.0);
        Timeline::Translate(t).apply(&mut sk, -1.0, 0.5, 1.0, MixFrom::Setup, false, false);
        let b = sk.bone(0).unwrap();
        assert!((b.x - 5.0).abs() < 1e-4 && (b.y - 10.0).abs() < 1e-4);
    }

    #[test]
    fn scale_multiplies_setup_scale() {
        let mut sk = skeleton(0.0); // setup scale defaults to (1, 1)
        let mut t = BoneTimeline::two_value(0, 2, 0);
        t.set_frame2(0, 0.0, 1.0, 1.0);
        t.set_frame2(1, 1.0, 2.0, 2.0);
        Timeline::Scale(t).apply(&mut sk, -1.0, 0.5, 1.0, MixFrom::Setup, false, false);
        let b = sk.bone(0).unwrap();
        assert!((b.scale_x - 1.5).abs() < 1e-4 && (b.scale_y - 1.5).abs() < 1e-4);
    }

    /// A root rotated 90 degrees and a child that inherits normally.
    fn rotated_parent() -> Skeleton {
        let data = SkeletonData {
            bones: vec![
                BoneData {
                    index: 0,
                    name: "root".into(),
                    rotation: 90.0,
                    ..Default::default()
                },
                BoneData {
                    index: 1,
                    name: "child".into(),
                    parent: Some(0),
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        Skeleton::new(Arc::new(data))
    }

    #[test]
    fn inherit_timeline_steps_the_bone_inherit_mode() {
        let mut sk = rotated_parent();
        let t = Timeline::Inherit(InheritTimeline::new(
            1,
            vec![0.5, 1.0],
            vec![Inherit::OnlyTranslation, Inherit::NoScale],
        ));
        let inherit = |sk: &Skeleton| sk.bone(1).unwrap().inherit();

        // Keys hold until the next one.
        t.apply(&mut sk, -1.0, 0.75, 1.0, MixFrom::Setup, false, false);
        assert_eq!(inherit(&sk), Inherit::OnlyTranslation);
        // The child no longer takes the parent's rotation.
        sk.update_world_transform();
        assert!((sk.bone(1).unwrap().a() - 1.0).abs() < 1e-4);
        t.apply(&mut sk, -1.0, 2.0, 1.0, MixFrom::Setup, false, false);
        assert_eq!(inherit(&sk), Inherit::NoScale);

        // Before the first key, blending from the current pose keeps the mode,
        // and blending from the setup pose restores it.
        t.apply(&mut sk, -1.0, 0.0, 1.0, MixFrom::Current, false, false);
        assert_eq!(inherit(&sk), Inherit::NoScale);
        t.apply(&mut sk, -1.0, 0.0, 1.0, MixFrom::Setup, false, false);
        assert_eq!(inherit(&sk), Inherit::Normal);

        // Resetting the bones restores the setup mode too.
        t.apply(&mut sk, -1.0, 0.75, 1.0, MixFrom::Setup, false, false);
        sk.set_bones_to_setup_pose();
        assert_eq!(inherit(&sk), Inherit::Normal);
    }
}
