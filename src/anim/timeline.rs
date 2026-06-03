//! Bone-property animation timelines (rotate / translate / scale).
//!
//! Each timeline keys one bone over time via a [`Curve`]. The apply logic is
//! transcribed from Spine 4.3: rotate/translate values are **added** to the
//! setup pose, scale values **multiply** it (with sign-adjusted mixing).

use super::curve::{absolute_value_with, Curve};
use super::MixFrom;
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

/// A keyframed animation channel for one bone property or constraint mix.
#[derive(Debug, Clone)]
pub(crate) enum Timeline {
    /// Local rotation (degrees).
    Rotate(BoneTimeline),
    /// Local translation (x, y).
    Translate(BoneTimeline),
    /// Local scale (x, y).
    Scale(BoneTimeline),
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
}

impl Timeline {
    /// Apply this timeline to `skeleton` at `time`. `from`/`add`/`out` follow
    /// Spine's mix semantics; `out` only affects scale.
    pub(crate) fn apply(
        &self,
        skeleton: &mut Skeleton,
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
            Timeline::Ik(t) => apply_ik(t, skeleton, time, alpha, from, out),
            Timeline::TransformMix(t) => apply_transform_mix(t, skeleton, time, alpha, from, add),
            Timeline::PathPosition(t) => apply_path_position(t, skeleton, time, alpha, from, add),
            Timeline::PathSpacing(t) => apply_path_spacing(t, skeleton, time, alpha, from),
            Timeline::PathMix(t) => apply_path_mix(t, skeleton, time, alpha, from, add),
            Timeline::Physics(t, property) => {
                apply_physics(t, *property, skeleton, time, alpha, from, add);
            }
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

/// Physics constraint timeline: drives one tunable. Mass is animated as a mass
/// value but stored inverted; wind and gravity blend additively.
fn apply_physics(
    t: &ConstraintTimeline,
    property: PhysicsProperty,
    skel: &mut Skeleton,
    time: f32,
    alpha: f32,
    from: MixFrom,
    add: bool,
) {
    let Some((pose, setup)) = skel.physics_pose_and_setup(t.constraint) else {
        return;
    };
    match property {
        PhysicsProperty::Inertia => {
            pose.inertia =
                t.curve
                    .absolute_value(time, alpha, from, add, pose.inertia, setup.inertia);
        }
        PhysicsProperty::Strength => {
            pose.strength =
                t.curve
                    .absolute_value(time, alpha, from, add, pose.strength, setup.strength);
        }
        PhysicsProperty::Damping => {
            pose.damping =
                t.curve
                    .absolute_value(time, alpha, from, add, pose.damping, setup.damping);
        }
        PhysicsProperty::Mass => {
            let cur = 1.0 / pose.mass_inverse;
            let base = 1.0 / setup.mass_inverse;
            let m = t.curve.absolute_value(time, alpha, from, add, cur, base);
            pose.mass_inverse = 1.0 / m;
        }
        PhysicsProperty::Wind => {
            pose.wind = t
                .curve
                .absolute_value(time, alpha, from, true, pose.wind, setup.wind);
        }
        PhysicsProperty::Gravity => {
            pose.gravity =
                t.curve
                    .absolute_value(time, alpha, from, true, pose.gravity, setup.gravity);
        }
        PhysicsProperty::Mix => {
            pose.mix = t
                .curve
                .absolute_value(time, alpha, from, add, pose.mix, setup.mix);
        }
    }
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
        Timeline::Rotate(t).apply(&mut sk, 0.5, 1.0, MixFrom::Setup, false, false);
        assert!((rotation(&sk) - 45.0).abs() < 1e-4);
    }

    #[test]
    fn rotate_respects_setup_offset_and_alpha() {
        // Setup rotation 30; animation keys +60 at full; mix at half weight.
        let mut sk = skeleton(30.0);
        let mut t = BoneTimeline::one_value(0, 2, 0);
        t.set_frame1(0, 0.0, 0.0);
        t.set_frame1(1, 1.0, 60.0);
        Timeline::Rotate(t).apply(&mut sk, 1.0, 0.5, MixFrom::Setup, false, false);
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
        Timeline::Rotate(t).apply(&mut sk, 0.0, 1.0, MixFrom::Setup, false, false);
        assert!((rotation(&sk) - 30.0).abs() < 1e-4);
    }

    #[test]
    fn translate_adds_offsets_to_setup_position() {
        let mut sk = skeleton(0.0);
        let mut t = BoneTimeline::two_value(0, 2, 0);
        t.set_frame2(0, 0.0, 0.0, 0.0);
        t.set_frame2(1, 1.0, 10.0, 20.0);
        Timeline::Translate(t).apply(&mut sk, 0.5, 1.0, MixFrom::Setup, false, false);
        let b = sk.bone(0).unwrap();
        assert!((b.x - 5.0).abs() < 1e-4 && (b.y - 10.0).abs() < 1e-4);
    }

    #[test]
    fn scale_multiplies_setup_scale() {
        let mut sk = skeleton(0.0); // setup scale defaults to (1, 1)
        let mut t = BoneTimeline::two_value(0, 2, 0);
        t.set_frame2(0, 0.0, 1.0, 1.0);
        t.set_frame2(1, 1.0, 2.0, 2.0);
        Timeline::Scale(t).apply(&mut sk, 0.5, 1.0, MixFrom::Setup, false, false);
        let b = sk.bone(0).unwrap();
        assert!((b.scale_x - 1.5).abs() < 1e-4 && (b.scale_y - 1.5).abs() < 1e-4);
    }
}
