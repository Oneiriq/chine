//! Inverse-kinematics constraints.
//!
//! An [`IkConstraint`] rotates 1 or 2 bones so the tip of the last bone reaches
//! toward a target bone. The solver math is transcribed from Spine 4.3
//! `IkConstraint` for fidelity; it reads each bone's world transform and writes
//! the constrained bones' local rotation (and scale, when stretching).

use core::f32::consts::PI;

use super::ScaleYMode;
use crate::data::{Inherit, SkeletonData};
use crate::skel::Bone;

/// Determinant / length epsilon (libgdx `MathUtils` float rounding error).
const EPSILON: f32 = 1e-6;

/// Setup data for an IK constraint.
#[derive(Debug, Clone)]
pub struct IkConstraintData {
    /// Constraint name, unique within the skeleton.
    pub name: String,
    /// Global constraint order (lower applies first).
    pub order: usize,
    /// The 1 or 2 constrained bone indices (parent first).
    pub bones: Vec<usize>,
    /// Target bone index.
    pub target: usize,
    /// How Y scale reacts to stretch / compress.
    pub scale_y_mode: ScaleYMode,
    /// Setup mix (`0` disables, `1` full).
    pub mix: f32,
    /// Setup softness (length over which 2-bone IK eases in).
    pub softness: f32,
    /// Setup bend direction (`+1` or `-1`).
    pub bend_direction: i32,
    /// Whether the bone may shorten to reach a closer target.
    pub compress: bool,
    /// Whether the bone may lengthen to reach a farther target.
    pub stretch: bool,
}

/// A runtime IK constraint pose, mixable by animation.
#[derive(Debug, Clone)]
pub(crate) struct IkConstraint {
    /// Mix weight in `[0, 1]`.
    pub mix: f32,
    /// Softness length.
    pub softness: f32,
    /// Bend direction (`+1` or `-1`).
    pub bend_direction: i32,
    /// Allow compression.
    pub compress: bool,
    /// Allow stretching.
    pub stretch: bool,
}

impl IkConstraint {
    /// Build a runtime pose from setup data.
    #[must_use]
    pub(crate) fn from_data(data: &IkConstraintData) -> Self {
        Self {
            mix: data.mix,
            softness: data.softness,
            bend_direction: data.bend_direction,
            compress: data.compress,
            stretch: data.stretch,
        }
    }
}

/// Solve IK constraint `c` against the current world transforms, writing the
/// constrained bones' local pose. The update cache recomputes their world
/// transforms afterward.
pub(crate) fn solve(
    bones: &mut [Bone],
    data: &SkeletonData,
    c: usize,
    pose: &IkConstraint,
    skel_sx: f32,
    skel_sy: f32,
) {
    let ik = &data.ik_constraints[c];
    if pose.mix == 0.0 || ik.bones.is_empty() {
        return;
    }
    let target_x = bones[ik.target].world_x();
    let target_y = bones[ik.target].world_y();
    match ik.bones.len() {
        1 => apply1(
            bones,
            data,
            ik.bones[0],
            target_x,
            target_y,
            pose.compress,
            pose.stretch,
            ik.scale_y_mode,
            pose.mix,
            skel_sx,
            skel_sy,
        ),
        2 => apply2(bones, data, ik, target_x, target_y, pose, skel_sx, skel_sy),
        _ => {}
    }
}

/// Sign of `x` matching Java `Math.signum` (zero stays zero).
fn signum(x: f32) -> f32 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

/// 1-bone IK: rotate `bone_idx` so its tip points at the world target.
#[allow(clippy::too_many_arguments)]
fn apply1(
    bones: &mut [Bone],
    data: &SkeletonData,
    bone_idx: usize,
    target_x: f32,
    target_y: f32,
    compress: bool,
    stretch: bool,
    scale_y_mode: ScaleYMode,
    mix: f32,
    skel_sx: f32,
    skel_sy: f32,
) {
    let Some(parent_idx) = bones[bone_idx].parent() else {
        return;
    };
    let inherit = data.bones[bone_idx].inherit;
    let length = data.bones[bone_idx].length;
    let (bx, by, brot, bshear_x, bscale_x, bscale_y, bwx, bwy) = {
        let b = &bones[bone_idx];
        (
            b.x,
            b.y,
            b.rotation,
            b.shear_x,
            b.scale_x,
            b.scale_y,
            b.world_x(),
            b.world_y(),
        )
    };
    let (pa, mut pb, pc, mut pd, pwx, pwy) = {
        let p = &bones[parent_idx];
        (p.a(), p.b(), p.c(), p.d(), p.world_x(), p.world_y())
    };

    let mut rotation_ik = -bshear_x - brot;
    let mut tx;
    let mut ty;
    if matches!(inherit, Inherit::OnlyTranslation) {
        tx = (target_x - bwx) * signum(skel_sx);
        ty = (target_y - bwy) * signum(skel_sy);
    } else {
        if matches!(inherit, Inherit::NoRotationOrReflection) {
            let s = (pa * pd - pb * pc).abs() / (pa * pa + pc * pc).max(EPSILON);
            let sa = pa / skel_sx;
            let sc = pc / skel_sy;
            pb = -sc * s * skel_sx;
            pd = sa * s * skel_sy;
            rotation_ik += sc.atan2(sa).to_degrees();
        }
        let x = target_x - pwx;
        let y = target_y - pwy;
        let det = pa * pd - pb * pc;
        if det.abs() <= EPSILON {
            tx = 0.0;
            ty = 0.0;
        } else {
            tx = (x * pd - y * pb) / det - bx;
            ty = (y * pa - x * pc) / det - by;
        }
    }
    rotation_ik += ty.atan2(tx).to_degrees();
    if bscale_x < 0.0 {
        rotation_ik += 180.0;
    }
    if rotation_ik > 180.0 {
        rotation_ik -= 360.0;
    } else if rotation_ik <= -180.0 {
        rotation_ik += 360.0;
    }
    bones[bone_idx].rotation = brot + rotation_ik * mix;

    if compress || stretch {
        if matches!(inherit, Inherit::NoScale | Inherit::NoScaleOrReflection) {
            tx = target_x - bwx;
            ty = target_y - bwy;
        }
        let b_len = length * bscale_x;
        if b_len > EPSILON {
            let dd = tx * tx + ty * ty;
            if (compress && dd < b_len * b_len) || (stretch && dd > b_len * b_len) {
                let s = (dd.sqrt() / b_len - 1.0) * mix + 1.0;
                bones[bone_idx].scale_x = bscale_x * s;
                match scale_y_mode {
                    ScaleYMode::Uniform => bones[bone_idx].scale_y = bscale_y * s,
                    ScaleYMode::Volume => {
                        let div = if s < 0.7 { 0.25 + 0.642_857 * s } else { s };
                        bones[bone_idx].scale_y = bscale_y / div;
                    }
                    ScaleYMode::None => {}
                }
            }
        }
    }
}

/// 2-bone IK: bend `parent`/`child` so the child's tip reaches the world target.
#[allow(
    clippy::too_many_arguments,
    clippy::similar_names,
    clippy::many_single_char_names
)]
fn apply2(
    bones: &mut [Bone],
    data: &SkeletonData,
    ik: &IkConstraintData,
    target_x: f32,
    target_y: f32,
    pose: &IkConstraint,
    skel_sx: f32,
    skel_sy: f32,
) {
    let parent_idx = ik.bones[0];
    let child_idx = ik.bones[1];
    if !matches!(data.bones[parent_idx].inherit, Inherit::Normal)
        || !matches!(data.bones[child_idx].inherit, Inherit::Normal)
    {
        return;
    }
    let bend_dir = pose.bend_direction as f32;
    let mix = pose.mix;
    let stretch = pose.stretch;
    let mut softness = pose.softness;

    let (px, py) = (bones[parent_idx].x, bones[parent_idx].y);
    let (mut psx, mut psy) = (bones[parent_idx].scale_x, bones[parent_idx].scale_y);
    let cx = bones[child_idx].x;
    let cy = bones[child_idx].y;
    let mut csx = bones[child_idx].scale_x;
    let parent_rot = bones[parent_idx].rotation;
    let child_rot = bones[child_idx].rotation;
    let child_shear_x = bones[child_idx].shear_x;

    let os1;
    let os2;
    let mut s2;
    if psx < 0.0 {
        psx = -psx;
        os1 = 180.0_f32;
        s2 = -1.0_f32;
    } else {
        os1 = 0.0;
        s2 = 1.0;
    }
    if psy < 0.0 {
        psy = -psy;
        s2 = -s2;
    }
    if csx < 0.0 {
        csx = -csx;
        os2 = 180.0_f32;
    } else {
        os2 = 0.0;
    }

    let (pa, pb, pc, pd, pwx, pwy) = {
        let p = &bones[parent_idx];
        (p.a(), p.b(), p.c(), p.d(), p.world_x(), p.world_y())
    };
    let u = (psx - psy).abs() <= EPSILON;
    let mut child_y = cy;
    let (cwx, cwy);
    if !u || stretch {
        child_y = 0.0;
        bones[child_idx].y = 0.0;
        cwx = pa * cx + pwx;
        cwy = pc * cx + pwy;
    } else {
        cwx = pa * cx + pb * cy + pwx;
        cwy = pc * cx + pd * cy + pwy;
    }

    let Some(gp_idx) = bones[parent_idx].parent() else {
        return;
    };
    let (ga, gb, gc, gd, gwx, gwy) = {
        let g = &bones[gp_idx];
        (g.a(), g.b(), g.c(), g.d(), g.world_x(), g.world_y())
    };
    let id0 = ga * gd - gb * gc;
    let id = if id0.abs() <= EPSILON { 0.0 } else { 1.0 / id0 };
    let mut x = cwx - gwx;
    let mut y = cwy - gwy;
    let dx = (x * gd - y * gb) * id - px;
    let dy = (y * ga - x * gc) * id - py;
    let l1 = (dx * dx + dy * dy).sqrt();
    let mut l2 = data.bones[child_idx].length * csx;

    if l1 < EPSILON {
        apply1(
            bones,
            data,
            parent_idx,
            target_x,
            target_y,
            false,
            stretch,
            ScaleYMode::None,
            mix,
            skel_sx,
            skel_sy,
        );
        bones[child_idx].rotation = 0.0;
        return;
    }

    x = target_x - gwx;
    y = target_y - gwy;
    let mut tx = (x * gd - y * gb) * id - px;
    let mut ty = (y * ga - x * gc) * id - py;
    let mut dd = tx * tx + ty * ty;
    if softness != 0.0 {
        softness *= psx * (csx + 1.0) * 0.5;
        let td = dd.sqrt();
        let sd = td - l1 - l2 * psx + softness;
        if sd > 0.0 {
            let mut p = 1.0_f32.min(sd / (softness * 2.0)) - 1.0;
            p = (sd - softness * (1.0 - p * p)) / td;
            tx -= p * tx;
            ty -= p * ty;
            dd = tx * tx + ty * ty;
        }
    }

    let a1;
    let a2;
    if u {
        l2 *= psx;
        let mut cos = (dd - l1 * l1 - l2 * l2) / (2.0 * l1 * l2);
        if cos < -1.0 {
            cos = -1.0;
            a2 = PI * bend_dir;
        } else if cos > 1.0 {
            cos = 1.0;
            a2 = 0.0;
            if stretch {
                let s = (dd.sqrt() / (l1 + l2) - 1.0) * mix + 1.0;
                bones[parent_idx].scale_x *= s;
                match ik.scale_y_mode {
                    ScaleYMode::Uniform => bones[parent_idx].scale_y *= s,
                    ScaleYMode::Volume => {
                        let div = if s < 0.7 { 0.25 + 0.642_857 * s } else { s };
                        bones[parent_idx].scale_y /= div;
                    }
                    ScaleYMode::None => {}
                }
            }
        } else {
            a2 = cos.acos() * bend_dir;
        }
        let aa = l1 + l2 * cos;
        let bb = l2 * a2.sin();
        a1 = (ty * aa - tx * bb).atan2(tx * aa + ty * bb);
    } else {
        let av = psx * l2;
        let bv = psy * l2;
        let aa = av * av;
        let bb = bv * bv;
        let ta = ty.atan2(tx);
        let cc = bb * l1 * l1 + aa * dd - aa * bb;
        let c1 = -2.0 * bb * l1;
        let c2 = bb - aa;
        let dv = c1 * c1 - 4.0 * c2 * cc;
        let mut solved = None;
        if dv >= 0.0 {
            let mut q = dv.sqrt();
            if c1 < 0.0 {
                q = -q;
            }
            q = -(c1 + q) * 0.5;
            let r0 = q / c2;
            let r1 = cc / q;
            let r = if r0.abs() < r1.abs() { r0 } else { r1 };
            let r0b = dd - r * r;
            if r0b >= 0.0 {
                let yy = r0b.sqrt() * bend_dir;
                solved = Some((ta - yy.atan2(r), (yy / psy).atan2((r - l1) / psx)));
            }
        }
        if let Some((sa1, sa2)) = solved {
            a1 = sa1;
            a2 = sa2;
        } else {
            let mut min_angle = PI;
            let mut min_x = l1 - av;
            let mut min_dist = min_x * min_x;
            let mut min_y = 0.0_f32;
            let mut max_angle = 0.0_f32;
            let mut max_x = l1 + av;
            let mut max_dist = max_x * max_x;
            let mut max_y = 0.0_f32;
            let cv = -av * l1 / (aa - bb);
            if (-1.0..=1.0).contains(&cv) {
                let ca = cv.acos();
                let xx = av * ca.cos() + l1;
                let yy = bv * ca.sin();
                let dval = xx * xx + yy * yy;
                if dval < min_dist {
                    min_angle = ca;
                    min_dist = dval;
                    min_x = xx;
                    min_y = yy;
                }
                if dval > max_dist {
                    max_angle = ca;
                    max_dist = dval;
                    max_x = xx;
                    max_y = yy;
                }
            }
            if dd <= (min_dist + max_dist) * 0.5 {
                a1 = ta - (min_y * bend_dir).atan2(min_x);
                a2 = min_angle * bend_dir;
            } else {
                a1 = ta - (max_y * bend_dir).atan2(max_x);
                a2 = max_angle * bend_dir;
            }
        }
    }

    let os = child_y.atan2(cx) * s2;
    let mut a1d = (a1 - os).to_degrees() + os1 - parent_rot;
    if a1d > 180.0 {
        a1d -= 360.0;
    } else if a1d <= -180.0 {
        a1d += 360.0;
    }
    bones[parent_idx].rotation = parent_rot + a1d * mix;
    let mut a2d = ((a2 + os).to_degrees() - child_shear_x) * s2 + os2 - child_rot;
    if a2d > 180.0 {
        a2d -= 360.0;
    } else if a2d <= -180.0 {
        a2d += 360.0;
    }
    bones[child_idx].rotation = child_rot + a2d * mix;
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use glam::Vec2;

    use super::*;
    use crate::data::{BoneData, SkeletonData};
    use crate::skel::Skeleton;

    fn bone(
        index: usize,
        name: &str,
        parent: Option<usize>,
        x: f32,
        y: f32,
        length: f32,
    ) -> BoneData {
        BoneData {
            index,
            name: name.into(),
            parent,
            length,
            position: Vec2::new(x, y),
            ..Default::default()
        }
    }

    fn ik(name: &str, bones: Vec<usize>, target: usize) -> IkConstraintData {
        IkConstraintData {
            name: name.into(),
            order: 0,
            bones,
            target,
            scale_y_mode: ScaleYMode::None,
            mix: 1.0,
            softness: 0.0,
            bend_direction: 1,
            compress: false,
            stretch: false,
        }
    }

    #[test]
    fn one_bone_ik_points_at_target() {
        let data = SkeletonData {
            bones: vec![
                bone(0, "root", None, 0.0, 0.0, 0.0),
                bone(1, "aim", Some(0), 0.0, 0.0, 10.0),
                bone(2, "target", Some(0), 0.0, 10.0, 0.0),
            ],
            ik_constraints: vec![ik("aim-ik", vec![1], 2)],
            ..Default::default()
        };
        let mut sk = Skeleton::new(Arc::new(data));
        sk.update_world_transform();
        // Aim should point straight up (+y): world x-axis = (a, c) ~= (0, 1).
        let aim = sk.bone(1).unwrap();
        assert!(aim.a().abs() < 1e-3, "a={}", aim.a());
        assert!((aim.c() - 1.0).abs() < 1e-3, "c={}", aim.c());
    }

    #[test]
    fn two_bone_ik_reaches_target() {
        let data = SkeletonData {
            bones: vec![
                bone(0, "root", None, 0.0, 0.0, 0.0),
                bone(1, "thigh", Some(0), 0.0, 0.0, 10.0),
                bone(2, "shin", Some(1), 10.0, 0.0, 10.0),
                bone(3, "target", Some(0), 10.0, 10.0, 0.0),
            ],
            ik_constraints: vec![ik("leg-ik", vec![1, 2], 3)],
            ..Default::default()
        };
        let mut sk = Skeleton::new(Arc::new(data));
        sk.update_world_transform();
        // Tip of the shin = its world origin + its world x-axis * length.
        let shin = sk.bone(2).unwrap();
        let tip_x = shin.world_x() + shin.a() * 10.0;
        let tip_y = shin.world_y() + shin.c() * 10.0;
        assert!((tip_x - 10.0).abs() < 0.05, "tip_x={tip_x}");
        assert!((tip_y - 10.0).abs() < 0.05, "tip_y={tip_y}");
    }

    #[test]
    fn child_of_ik_chain_follows() {
        // foot is a child of the IK-controlled shin (not itself constrained);
        // the update cache must recompute it after the IK runs.
        let data = SkeletonData {
            bones: vec![
                bone(0, "root", None, 0.0, 0.0, 0.0),
                bone(1, "thigh", Some(0), 0.0, 0.0, 10.0),
                bone(2, "shin", Some(1), 10.0, 0.0, 10.0),
                bone(3, "foot", Some(2), 10.0, 0.0, 0.0),
                bone(4, "target", Some(0), 10.0, 10.0, 0.0),
            ],
            ik_constraints: vec![ik("leg-ik", vec![1, 2], 4)],
            ..Default::default()
        };
        let mut sk = Skeleton::new(Arc::new(data));
        sk.update_world_transform();
        // foot sits at the shin's tip, which IK drove toward ~(10, 10).
        let foot = sk.bone(3).unwrap();
        assert!(
            (foot.world_x() - 10.0).abs() < 0.05,
            "fx={}",
            foot.world_x()
        );
        assert!(
            (foot.world_y() - 10.0).abs() < 0.05,
            "fy={}",
            foot.world_y()
        );
    }

    #[test]
    fn zero_mix_leaves_pose_unchanged() {
        let mut cdata = ik("aim-ik", vec![1], 2);
        cdata.mix = 0.0;
        let data = SkeletonData {
            bones: vec![
                bone(0, "root", None, 0.0, 0.0, 0.0),
                bone(1, "aim", Some(0), 0.0, 0.0, 10.0),
                bone(2, "target", Some(0), 0.0, 10.0, 0.0),
            ],
            ik_constraints: vec![cdata],
            ..Default::default()
        };
        let mut sk = Skeleton::new(Arc::new(data));
        sk.update_world_transform();
        // mix 0 -> aim keeps its setup orientation (+x): (a, c) ~= (1, 0).
        let aim = sk.bone(1).unwrap();
        assert!((aim.a() - 1.0).abs() < 1e-3 && aim.c().abs() < 1e-3);
    }
}
