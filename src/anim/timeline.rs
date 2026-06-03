//! Bone-property animation timelines (rotate / translate / scale).
//!
//! Each timeline keys one bone over time via a [`Curve`]. The apply logic is
//! transcribed from Spine 4.3: rotate/translate values are **added** to the
//! setup pose, scale values **multiply** it (with sign-adjusted mixing).

use super::curve::Curve;
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

/// A keyframed animation channel for one bone property.
#[derive(Debug, Clone)]
pub(crate) enum Timeline {
    /// Local rotation (degrees).
    Rotate(BoneTimeline),
    /// Local translation (x, y).
    Translate(BoneTimeline),
    /// Local scale (x, y).
    Scale(BoneTimeline),
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
