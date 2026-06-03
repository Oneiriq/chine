//! Bone-property animation timelines (rotate / translate / scale).
//!
//! Each timeline keys one bone over time via a [`Curve`]. The apply logic is
//! transcribed from Spine 4.3: rotate/translate values are **added** to the
//! setup pose, scale values **multiply** it (with sign-adjusted mixing).

use super::curve::{absolute_value_with, Curve};
use super::MixFrom;
use crate::constraint::physics::{PhysicsConstraint, PhysicsConstraintData};
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

/// A mesh deform timeline (unweighted): per-keyframe local-vertex offsets added
/// to the attachment's setup vertices. Only applies while the slot shows the
/// matching attachment.
#[derive(Debug, Clone)]
pub(crate) struct DeformTimeline {
    slot: usize,
    attachment: String,
    setup: Vec<f32>,
    times: Vec<f32>,
    frames: Vec<Vec<f32>>,
}

impl DeformTimeline {
    /// A deform timeline for `slot`/`attachment` with setup vertices and the
    /// per-keyframe offset frames.
    pub(crate) fn new(
        slot: usize,
        attachment: String,
        setup: Vec<f32>,
        times: Vec<f32>,
        frames: Vec<Vec<f32>>,
    ) -> Self {
        Self {
            slot,
            attachment,
            setup,
            times,
            frames,
        }
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

/// A keyframed animation channel for one bone property or constraint mix.
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
    /// Slot tint color; the index is the slot, the flag whether alpha is keyed
    /// (RGBA vs RGB).
    SlotColor(ConstraintTimeline, bool),
    /// Slot attachment swap (stepped).
    Attachment(AttachmentTimeline),
    /// Slot draw order (stepped permutations).
    DrawOrder(DrawOrderTimeline),
    /// Animation events fired on keyframe crossings.
    Event(EventTimeline),
    /// Mesh deform (unweighted local-vertex offsets).
    Deform(DeformTimeline),
}

impl Timeline {
    /// Apply this timeline to `skeleton` over the window `(last_time, time]`.
    /// `from`/`add`/`out` follow Spine's mix semantics; `out` only affects
    /// scale; `last_time` is used only by the physics reset timeline.
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
            Timeline::Ik(t) => apply_ik(t, skeleton, time, alpha, from, out),
            Timeline::TransformMix(t) => apply_transform_mix(t, skeleton, time, alpha, from, add),
            Timeline::PathPosition(t) => apply_path_position(t, skeleton, time, alpha, from, add),
            Timeline::PathSpacing(t) => apply_path_spacing(t, skeleton, time, alpha, from),
            Timeline::PathMix(t) => apply_path_mix(t, skeleton, time, alpha, from, add),
            Timeline::Physics(t, property) => {
                apply_physics(t, *property, skeleton, time, alpha, from, add);
            }
            Timeline::PhysicsReset(t) => apply_physics_reset(t, skeleton, last_time, time),
            Timeline::SlotColor(t, has_alpha) => {
                apply_slot_color(t, *has_alpha, skeleton, time, alpha, from, add);
            }
            Timeline::Attachment(t) => apply_attachment(t, skeleton, time, from),
            Timeline::DrawOrder(t) => apply_draw_order(t, skeleton, time),
            Timeline::Event(t) => apply_event(t, skeleton, last_time, time),
            Timeline::Deform(t) => apply_deform(t, skeleton, time, alpha, from),
        }
    }
}

fn apply_rotate(
    t: &BoneTimeline,
    skel: &mut Skeleton,
    time: f32,
    alpha: f32,
    from: MixFrom,
    add: bool,
) {
    if let Some((bone, setup)) = skel.bone_and_setup(t.bone) {
        bone.rotation =
            t.curve
                .relative_value(time, alpha, from, add, bone.rotation, setup.rotation);
    }
}

fn apply_translate(
    t: &BoneTimeline,
    skel: &mut Skeleton,
    time: f32,
    alpha: f32,
    from: MixFrom,
    add: bool,
) {
    let Some((bone, setup)) = skel.bone_and_setup(t.bone) else {
        return;
    };
    if time < t.curve.first_time() {
        match from {
            MixFrom::Setup => {
                bone.x = setup.position.x;
                bone.y = setup.position.y;
            }
            MixFrom::First => {
                bone.x += (setup.position.x - bone.x) * alpha;
                bone.y += (setup.position.y - bone.y) * alpha;
            }
            MixFrom::Current => {}
        }
        return;
    }
    let x = t.curve.value(time, 1);
    let y = t.curve.value(time, 2);
    if matches!(from, MixFrom::Setup) {
        bone.x = setup.position.x + x * alpha;
        bone.y = setup.position.y + y * alpha;
    } else if add {
        bone.x += x * alpha;
        bone.y += y * alpha;
    } else {
        bone.x += (setup.position.x + x - bone.x) * alpha;
        bone.y += (setup.position.y + y - bone.y) * alpha;
    }
}

#[allow(clippy::similar_names)]
fn apply_scale(
    t: &BoneTimeline,
    skel: &mut Skeleton,
    time: f32,
    alpha: f32,
    from: MixFrom,
    add: bool,
    out: bool,
) {
    let Some((bone, setup)) = skel.bone_and_setup(t.bone) else {
        return;
    };
    if time < t.curve.first_time() {
        match from {
            MixFrom::Setup => {
                bone.scale_x = setup.scale.x;
                bone.scale_y = setup.scale.y;
            }
            MixFrom::First => {
                bone.scale_x += (setup.scale.x - bone.scale_x) * alpha;
                bone.scale_y += (setup.scale.y - bone.scale_y) * alpha;
            }
            MixFrom::Current => {}
        }
        return;
    }
    let x = t.curve.value(time, 1) * setup.scale.x;
    let y = t.curve.value(time, 2) * setup.scale.y;
    if alpha == 1.0 && !add {
        bone.scale_x = x;
        bone.scale_y = y;
        return;
    }
    let (mut bx, mut by) = if matches!(from, MixFrom::Setup) {
        (setup.scale.x, setup.scale.y)
    } else {
        (bone.scale_x, bone.scale_y)
    };
    if add {
        bone.scale_x = bx + (x - setup.scale.x) * alpha;
        bone.scale_y = by + (y - setup.scale.y) * alpha;
    } else if out {
        bone.scale_x = bx + (x.abs() * signum(bx) - bx) * alpha;
        bone.scale_y = by + (y.abs() * signum(by) - by) * alpha;
    } else {
        bx = bx.abs() * signum(x);
        by = by.abs() * signum(y);
        bone.scale_x = bx + (x - bx) * alpha;
        bone.scale_y = by + (y - by) * alpha;
    }
}

/// Sign of `x`, matching Java's `Math.signum` (zero stays zero, unlike
/// `f32::signum`, which returns `±1` for `±0`).
fn signum(x: f32) -> f32 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

fn apply_shear(
    t: &BoneTimeline,
    skel: &mut Skeleton,
    time: f32,
    alpha: f32,
    from: MixFrom,
    add: bool,
) {
    let Some((bone, setup)) = skel.bone_and_setup(t.bone) else {
        return;
    };
    if time < t.curve.first_time() {
        match from {
            MixFrom::Setup => {
                bone.shear_x = setup.shear.x;
                bone.shear_y = setup.shear.y;
            }
            MixFrom::First => {
                bone.shear_x += (setup.shear.x - bone.shear_x) * alpha;
                bone.shear_y += (setup.shear.y - bone.shear_y) * alpha;
            }
            MixFrom::Current => {}
        }
        return;
    }
    let x = t.curve.value(time, 1);
    let y = t.curve.value(time, 2);
    if matches!(from, MixFrom::Setup) {
        bone.shear_x = setup.shear.x + x * alpha;
        bone.shear_y = setup.shear.y + y * alpha;
    } else if add {
        bone.shear_x += x * alpha;
        bone.shear_y += y * alpha;
    } else {
        bone.shear_x += (setup.shear.x + x - bone.shear_x) * alpha;
        bone.shear_y += (setup.shear.y + y - bone.shear_y) * alpha;
    }
}

/// Apply a single-axis bone timeline. Translation and shear axes add to the
/// setup; scale axes multiply it with sign-aware mixing.
#[allow(clippy::too_many_arguments)]
fn apply_bone_axis(
    t: &BoneTimeline,
    axis: BoneAxis,
    skel: &mut Skeleton,
    time: f32,
    alpha: f32,
    from: MixFrom,
    add: bool,
    out: bool,
) {
    let Some((bone, setup)) = skel.bone_and_setup(t.bone) else {
        return;
    };
    match axis {
        BoneAxis::TranslateX => {
            bone.x = t
                .curve
                .relative_value(time, alpha, from, add, bone.x, setup.position.x);
        }
        BoneAxis::TranslateY => {
            bone.y = t
                .curve
                .relative_value(time, alpha, from, add, bone.y, setup.position.y);
        }
        BoneAxis::ShearX => {
            bone.shear_x =
                t.curve
                    .relative_value(time, alpha, from, add, bone.shear_x, setup.shear.x);
        }
        BoneAxis::ShearY => {
            bone.shear_y =
                t.curve
                    .relative_value(time, alpha, from, add, bone.shear_y, setup.shear.y);
        }
        BoneAxis::ScaleX => {
            if time < t.curve.first_time() {
                apply_scale_setup(&mut bone.scale_x, setup.scale.x, alpha, from);
            } else {
                bone.scale_x = scale_channel(
                    t.curve.value(time, 1),
                    setup.scale.x,
                    bone.scale_x,
                    alpha,
                    from,
                    add,
                    out,
                );
            }
        }
        BoneAxis::ScaleY => {
            if time < t.curve.first_time() {
                apply_scale_setup(&mut bone.scale_y, setup.scale.y, alpha, from);
            } else {
                bone.scale_y = scale_channel(
                    t.curve.value(time, 1),
                    setup.scale.y,
                    bone.scale_y,
                    alpha,
                    from,
                    add,
                    out,
                );
            }
        }
    }
}

/// Reset one scale channel toward its setup value (the before-first-frame case).
fn apply_scale_setup(scale: &mut f32, setup: f32, alpha: f32, from: MixFrom) {
    match from {
        MixFrom::Setup => *scale = setup,
        MixFrom::First => *scale += (setup - *scale) * alpha,
        MixFrom::Current => {}
    }
}

/// Blend one scale channel: the keyed value scales the setup, with Spine's
/// sign-aware mixing. `out` is the mix-out direction.
fn scale_channel(
    value: f32,
    setup: f32,
    current: f32,
    alpha: f32,
    from: MixFrom,
    add: bool,
    out: bool,
) -> f32 {
    let target = value * setup;
    if alpha == 1.0 && !add {
        return target;
    }
    let base = if matches!(from, MixFrom::Setup) {
        setup
    } else {
        current
    };
    if add {
        base + (target - setup) * alpha
    } else if out {
        base + (target.abs() * signum(base) - base) * alpha
    } else {
        let signed = base.abs() * signum(target);
        signed + (target - signed) * alpha
    }
}

/// IK constraint timeline: mix and softness are interpolated; bend direction,
/// compress, and stretch are stepped (read from the frame).
fn apply_ik(
    t: &ConstraintTimeline,
    skel: &mut Skeleton,
    time: f32,
    alpha: f32,
    from: MixFrom,
    out: bool,
) {
    let Some((pose, setup)) = skel.ik_pose_and_setup(t.constraint) else {
        return;
    };
    if time < t.curve.first_time() {
        match from {
            MixFrom::Setup => {
                pose.mix = setup.mix;
                pose.softness = setup.softness;
                pose.bend_direction = setup.bend_direction;
                pose.compress = setup.compress;
                pose.stretch = setup.stretch;
            }
            MixFrom::First => {
                pose.mix += (setup.mix - pose.mix) * alpha;
                pose.softness += (setup.softness - pose.softness) * alpha;
                pose.bend_direction = setup.bend_direction;
                pose.compress = setup.compress;
                pose.stretch = setup.stretch;
            }
            MixFrom::Current => {}
        }
        return;
    }
    let mix = t.curve.value(time, 1);
    let softness = t.curve.value(time, 2);
    let (base_mix, base_soft) = if matches!(from, MixFrom::Setup) {
        (setup.mix, setup.softness)
    } else {
        (pose.mix, pose.softness)
    };
    pose.mix = base_mix + (mix - base_mix) * alpha;
    pose.softness = base_soft + (softness - base_soft) * alpha;
    if out {
        if matches!(from, MixFrom::Setup) {
            pose.bend_direction = setup.bend_direction;
            pose.compress = setup.compress;
            pose.stretch = setup.stretch;
        }
    } else {
        pose.bend_direction = t.curve.frame_value(time, 3) as i32;
        pose.compress = t.curve.frame_value(time, 4) != 0.0;
        pose.stretch = t.curve.frame_value(time, 5) != 0.0;
    }
}

/// Transform constraint timeline: six interpolated mixes.
fn apply_transform_mix(
    t: &ConstraintTimeline,
    skel: &mut Skeleton,
    time: f32,
    alpha: f32,
    from: MixFrom,
    add: bool,
) {
    let Some((pose, setup)) = skel.transform_pose_and_setup(t.constraint) else {
        return;
    };
    if time < t.curve.first_time() {
        match from {
            MixFrom::Setup => {
                pose.mix_rotate = setup.mix_rotate;
                pose.mix_x = setup.mix_x;
                pose.mix_y = setup.mix_y;
                pose.mix_scale_x = setup.mix_scale_x;
                pose.mix_scale_y = setup.mix_scale_y;
                pose.mix_shear_y = setup.mix_shear_y;
            }
            MixFrom::First => {
                pose.mix_rotate += (setup.mix_rotate - pose.mix_rotate) * alpha;
                pose.mix_x += (setup.mix_x - pose.mix_x) * alpha;
                pose.mix_y += (setup.mix_y - pose.mix_y) * alpha;
                pose.mix_scale_x += (setup.mix_scale_x - pose.mix_scale_x) * alpha;
                pose.mix_scale_y += (setup.mix_scale_y - pose.mix_scale_y) * alpha;
                pose.mix_shear_y += (setup.mix_shear_y - pose.mix_shear_y) * alpha;
            }
            MixFrom::Current => {}
        }
        return;
    }
    pose.mix_rotate = absolute_value_with(
        t.curve.value(time, 1),
        alpha,
        from,
        add,
        pose.mix_rotate,
        setup.mix_rotate,
    );
    pose.mix_x = absolute_value_with(
        t.curve.value(time, 2),
        alpha,
        from,
        add,
        pose.mix_x,
        setup.mix_x,
    );
    pose.mix_y = absolute_value_with(
        t.curve.value(time, 3),
        alpha,
        from,
        add,
        pose.mix_y,
        setup.mix_y,
    );
    pose.mix_scale_x = absolute_value_with(
        t.curve.value(time, 4),
        alpha,
        from,
        add,
        pose.mix_scale_x,
        setup.mix_scale_x,
    );
    pose.mix_scale_y = absolute_value_with(
        t.curve.value(time, 5),
        alpha,
        from,
        add,
        pose.mix_scale_y,
        setup.mix_scale_y,
    );
    pose.mix_shear_y = absolute_value_with(
        t.curve.value(time, 6),
        alpha,
        from,
        add,
        pose.mix_shear_y,
        setup.mix_shear_y,
    );
}

/// Path constraint position timeline (single absolute value).
fn apply_path_position(
    t: &ConstraintTimeline,
    skel: &mut Skeleton,
    time: f32,
    alpha: f32,
    from: MixFrom,
    add: bool,
) {
    if let Some((pose, setup)) = skel.path_pose_and_setup(t.constraint) {
        pose.position =
            t.curve
                .absolute_value(time, alpha, from, add, pose.position, setup.position);
    }
}

/// Path constraint spacing timeline (single absolute value, never additive).
fn apply_path_spacing(
    t: &ConstraintTimeline,
    skel: &mut Skeleton,
    time: f32,
    alpha: f32,
    from: MixFrom,
) {
    if let Some((pose, setup)) = skel.path_pose_and_setup(t.constraint) {
        pose.spacing =
            t.curve
                .absolute_value(time, alpha, from, false, pose.spacing, setup.spacing);
    }
}

/// Path constraint mix timeline (rotate / x / y).
fn apply_path_mix(
    t: &ConstraintTimeline,
    skel: &mut Skeleton,
    time: f32,
    alpha: f32,
    from: MixFrom,
    add: bool,
) {
    let Some((pose, setup)) = skel.path_pose_and_setup(t.constraint) else {
        return;
    };
    if time < t.curve.first_time() {
        match from {
            MixFrom::Setup => {
                pose.mix_rotate = setup.mix_rotate;
                pose.mix_x = setup.mix_x;
                pose.mix_y = setup.mix_y;
            }
            MixFrom::First => {
                pose.mix_rotate += (setup.mix_rotate - pose.mix_rotate) * alpha;
                pose.mix_x += (setup.mix_x - pose.mix_x) * alpha;
                pose.mix_y += (setup.mix_y - pose.mix_y) * alpha;
            }
            MixFrom::Current => {}
        }
        return;
    }
    pose.mix_rotate = absolute_value_with(
        t.curve.value(time, 1),
        alpha,
        from,
        add,
        pose.mix_rotate,
        setup.mix_rotate,
    );
    pose.mix_x = absolute_value_with(
        t.curve.value(time, 2),
        alpha,
        from,
        add,
        pose.mix_x,
        setup.mix_x,
    );
    pose.mix_y = absolute_value_with(
        t.curve.value(time, 3),
        alpha,
        from,
        add,
        pose.mix_y,
        setup.mix_y,
    );
}

/// Physics constraint timeline: drives one tunable on one constraint, or on
/// every constraint whose matching global flag is set when the target is
/// [`GLOBAL_PHYSICS`].
fn apply_physics(
    t: &ConstraintTimeline,
    property: PhysicsProperty,
    skel: &mut Skeleton,
    time: f32,
    alpha: f32,
    from: MixFrom,
    add: bool,
) {
    if t.constraint == GLOBAL_PHYSICS {
        let data = skel.data_arc();
        for (pose, setup) in skel
            .physics_constraints_mut()
            .iter_mut()
            .zip(&data.physics_constraints)
        {
            if property_global(setup, property) {
                apply_physics_one(pose, setup, property, &t.curve, time, alpha, from, add);
            }
        }
    } else if let Some((pose, setup)) = skel.physics_pose_and_setup(t.constraint) {
        apply_physics_one(pose, setup, property, &t.curve, time, alpha, from, add);
    }
}

/// Whether `property` is flagged global on `data` (so a global timeline drives
/// it).
fn property_global(data: &PhysicsConstraintData, property: PhysicsProperty) -> bool {
    match property {
        PhysicsProperty::Inertia => data.inertia_global,
        PhysicsProperty::Strength => data.strength_global,
        PhysicsProperty::Damping => data.damping_global,
        PhysicsProperty::Mass => data.mass_global,
        PhysicsProperty::Wind => data.wind_global,
        PhysicsProperty::Gravity => data.gravity_global,
        PhysicsProperty::Mix => data.mix_global,
    }
}

/// Apply one physics tunable to a single constraint pose. Mass is animated as a
/// mass value but stored inverted; wind and gravity blend additively.
#[allow(clippy::too_many_arguments)]
fn apply_physics_one(
    pose: &mut PhysicsConstraint,
    setup: &PhysicsConstraintData,
    property: PhysicsProperty,
    curve: &Curve,
    time: f32,
    alpha: f32,
    from: MixFrom,
    add: bool,
) {
    match property {
        PhysicsProperty::Inertia => {
            pose.inertia =
                curve.absolute_value(time, alpha, from, add, pose.inertia, setup.inertia);
        }
        PhysicsProperty::Strength => {
            pose.strength =
                curve.absolute_value(time, alpha, from, add, pose.strength, setup.strength);
        }
        PhysicsProperty::Damping => {
            pose.damping =
                curve.absolute_value(time, alpha, from, add, pose.damping, setup.damping);
        }
        PhysicsProperty::Mass => {
            let cur = 1.0 / pose.mass_inverse;
            let base = 1.0 / setup.mass_inverse;
            pose.mass_inverse = 1.0 / curve.absolute_value(time, alpha, from, add, cur, base);
        }
        PhysicsProperty::Wind => {
            pose.wind = curve.absolute_value(time, alpha, from, true, pose.wind, setup.wind);
        }
        PhysicsProperty::Gravity => {
            pose.gravity =
                curve.absolute_value(time, alpha, from, true, pose.gravity, setup.gravity);
        }
        PhysicsProperty::Mix => {
            pose.mix = curve.absolute_value(time, alpha, from, add, pose.mix, setup.mix);
        }
    }
}

/// Physics reset timeline: if a keyframe time falls in the window
/// `(last_time, time]` (handling a loop wrap where `time < last_time`), reset
/// the target constraint, or every physics constraint when global.
fn apply_physics_reset(t: &PhysicsResetTimeline, skel: &mut Skeleton, last_time: f32, time: f32) {
    let fired = if time >= last_time {
        t.times.iter().any(|&kt| kt > last_time && kt <= time)
    } else {
        t.times.iter().any(|&kt| kt > last_time || kt <= time)
    };
    if !fired {
        return;
    }
    if t.constraint == GLOBAL_PHYSICS {
        skel.request_all_physics_reset();
    } else {
        skel.request_physics_reset(t.constraint);
    }
}

/// Slot color timeline: blends the slot's RGB (and alpha when `has_alpha`) tint
/// from its setup color toward the keyed colors. The index is the slot.
fn apply_slot_color(
    t: &ConstraintTimeline,
    has_alpha: bool,
    skel: &mut Skeleton,
    time: f32,
    alpha: f32,
    from: MixFrom,
    add: bool,
) {
    let Some((slot, setup)) = skel.slot_pose_and_setup(t.constraint) else {
        return;
    };
    if time < t.curve.first_time() {
        match from {
            MixFrom::Setup => {
                slot.color.r = setup.color.r;
                slot.color.g = setup.color.g;
                slot.color.b = setup.color.b;
                if has_alpha {
                    slot.color.a = setup.color.a;
                }
            }
            MixFrom::First => {
                slot.color.r += (setup.color.r - slot.color.r) * alpha;
                slot.color.g += (setup.color.g - slot.color.g) * alpha;
                slot.color.b += (setup.color.b - slot.color.b) * alpha;
                if has_alpha {
                    slot.color.a += (setup.color.a - slot.color.a) * alpha;
                }
            }
            MixFrom::Current => {}
        }
        return;
    }
    slot.color.r = absolute_value_with(
        t.curve.value(time, 1),
        alpha,
        from,
        add,
        slot.color.r,
        setup.color.r,
    );
    slot.color.g = absolute_value_with(
        t.curve.value(time, 2),
        alpha,
        from,
        add,
        slot.color.g,
        setup.color.g,
    );
    slot.color.b = absolute_value_with(
        t.curve.value(time, 3),
        alpha,
        from,
        add,
        slot.color.b,
        setup.color.b,
    );
    if has_alpha {
        slot.color.a = absolute_value_with(
            t.curve.value(time, 4),
            alpha,
            from,
            add,
            slot.color.a,
            setup.color.a,
        );
    }
}

/// Slot attachment timeline: a stepped switch to the keyed attachment name.
fn apply_attachment(t: &AttachmentTimeline, skel: &mut Skeleton, time: f32, from: MixFrom) {
    let Some((slot, setup)) = skel.slot_pose_and_setup(t.slot) else {
        return;
    };
    if t.times.is_empty() || time < t.times[0] {
        if matches!(from, MixFrom::Setup | MixFrom::First) {
            slot.attachment = setup.attachment.clone();
        }
        return;
    }
    let idx = search_step(&t.times, time);
    slot.attachment = t.names[idx].clone();
}

/// Draw-order timeline: a stepped switch to the keyed slot ordering.
fn apply_draw_order(t: &DrawOrderTimeline, skel: &mut Skeleton, time: f32) {
    if t.times.is_empty() || time < t.times[0] {
        return; // before the first key: keep the setup order
    }
    let idx = search_step(&t.times, time);
    skel.set_draw_order(&t.orders[idx]);
}

/// Event timeline: fire each event whose keyframe time falls in the window
/// `(last_time, time]` (handling a loop wrap), collecting it on the skeleton.
fn apply_event(t: &EventTimeline, skel: &mut Skeleton, last_time: f32, time: f32) {
    let wrapped = time < last_time;
    for (i, &kt) in t.times.iter().enumerate() {
        let fired = if wrapped {
            kt > last_time || kt <= time
        } else {
            kt > last_time && kt <= time
        };
        if fired {
            skel.push_event(t.events[i].clone());
        }
    }
}

/// Mesh deform timeline: set the slot's deform buffer to the setup vertices plus
/// the interpolated keyframe offsets (scaled by `alpha`). Only applies while the
/// slot shows the timeline's attachment.
fn apply_deform(t: &DeformTimeline, skel: &mut Skeleton, time: f32, alpha: f32, from: MixFrom) {
    let Some((slot, _)) = skel.slot_pose_and_setup(t.slot) else {
        return;
    };
    if slot.attachment.as_deref() != Some(t.attachment.as_str()) {
        return;
    }
    let n = t.setup.len();
    if t.times.is_empty() || time < t.times[0] {
        if matches!(from, MixFrom::Setup) {
            slot.deform.clear();
        }
        return;
    }
    let offset = interp_deform(&t.times, &t.frames, time);
    slot.deform.resize(n, 0.0);
    for (i, d) in slot.deform.iter_mut().enumerate() {
        *d = t.setup[i] + offset.get(i).copied().unwrap_or(0.0) * alpha;
    }
}

/// Linearly interpolate the deform offset frames at `time`.
fn interp_deform(times: &[f32], frames: &[Vec<f32>], time: f32) -> Vec<f32> {
    let last = times.len() - 1;
    if time >= times[last] {
        return frames[last].clone();
    }
    let i = search_step(times, time);
    let (t0, t1) = (times[i], times[i + 1]);
    let alpha = if t1 > t0 {
        (time - t0) / (t1 - t0)
    } else {
        0.0
    };
    let a = &frames[i];
    let b = &frames[i + 1];
    let n = a.len().max(b.len());
    let mut out = vec![0.0; n];
    for (j, v) in out.iter_mut().enumerate() {
        let av = a.get(j).copied().unwrap_or(0.0);
        let bv = b.get(j).copied().unwrap_or(0.0);
        *v = av + (bv - av) * alpha;
    }
    out
}

/// Index of the last keyframe at or before `time` (assumes `time >= times[0]`).
fn search_step(times: &[f32], time: f32) -> usize {
    times.iter().rposition(|&t| t <= time).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::data::{BoneData, SkeletonData};

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
        // Setup rotation 30; animation keys +60 at full; mix at half weight.
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
}
