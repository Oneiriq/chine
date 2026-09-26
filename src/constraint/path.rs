//! Path constraints.
//!
//! A [`PathConstraint`] positions and rotates a chain of bones along a
//! [`crate::attach::PathAttachment`] on a target slot. The Bezier sampling
//! (`compute_world_positions`) and the bone apply re-implement Spine 4.3's
//! `PathConstraint`. chine samples every path with Spine's constant-speed
//! arc-length parameterization and ignores the attachment's `constant_speed`
//! flag and precomputed `lengths`. Spine samples a path without
//! `constant_speed` by those lengths instead, so bones on such a path can sit
//! at slightly different points than in Spine.
//!
//! Path constraints modify the constrained bones' **world** transforms, like
//! world-mode transform constraints. The update cache recomputes descendants.

use core::f32::consts::{PI, TAU};

use glam::Vec2;

use crate::attach::Attachment;
use crate::data::SkeletonData;
use crate::skel::Skeleton;

/// Degrees-to-radians factor.
const DEG_RAD: f32 = PI / 180.0;
/// Degenerate-length / spacing epsilon.
const EPSILON: f32 = 1e-5;

/// How the first bone is positioned along the path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PositionMode {
    /// Position is a world length along the path.
    Fixed,
    /// Position is a fraction of the path length.
    #[default]
    Percent,
}

/// How bones after the first are spaced along the path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SpacingMode {
    /// Space by each bone's setup length plus spacing.
    Length,
    /// A fixed world spacing between bones.
    Fixed,
    /// Spacing as a fraction of the path length.
    #[default]
    Percent,
    /// Spacing proportional to each bone's length.
    Proportional,
}

/// How bones are rotated along the path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RotateMode {
    /// Rotate to the path tangent.
    Tangent,
    /// Rotate as a chain toward the next bone's position.
    #[default]
    Chain,
    /// Like chain, also scaling bones to span gaps.
    ChainScale,
}

/// Setup data for a path constraint.
#[derive(Debug, Clone)]
pub struct PathConstraintData {
    /// Constraint name.
    pub name: String,
    /// Global constraint order (lower applies first).
    pub order: usize,
    /// Constrained bone indices, in chain order.
    pub bones: Vec<usize>,
    /// Slot index whose path attachment is followed.
    pub slot: usize,
    /// First-bone positioning mode.
    pub position_mode: PositionMode,
    /// Bone spacing mode.
    pub spacing_mode: SpacingMode,
    /// Bone rotation mode.
    pub rotate_mode: RotateMode,
    /// Rotation offset (degrees) added to each bone.
    pub offset_rotation: f32,
    /// Setup position along the path.
    pub position: f32,
    /// Setup spacing.
    pub spacing: f32,
    /// Setup rotation mix.
    pub mix_rotate: f32,
    /// Setup x mix.
    pub mix_x: f32,
    /// Setup y mix.
    pub mix_y: f32,
}

/// A runtime path-constraint pose (animatable).
#[derive(Debug, Clone, Copy)]
pub(crate) struct PathConstraint {
    /// Position along the path.
    pub position: f32,
    /// Spacing between bones.
    pub spacing: f32,
    /// Rotation mix in `[0, 1]`.
    pub mix_rotate: f32,
    /// X mix.
    pub mix_x: f32,
    /// Y mix.
    pub mix_y: f32,
}

impl PathConstraint {
    /// Build a runtime pose from setup data.
    #[must_use]
    pub(crate) fn from_data(data: &PathConstraintData) -> Self {
        Self {
            position: data.position,
            spacing: data.spacing,
            mix_rotate: data.mix_rotate,
            mix_x: data.mix_x,
            mix_y: data.mix_y,
        }
    }
}

/// Apply path constraint `c` to its constrained bones.
pub(crate) fn solve(skeleton: &mut Skeleton, c: usize, pose: PathConstraint) {
    if pose.mix_rotate == 0.0 && pose.mix_x == 0.0 && pose.mix_y == 0.0 {
        return;
    }
    let data = skeleton.data_arc();
    let Some(pc) = data.path_constraints.get(c) else {
        return;
    };
    apply(skeleton, &data, pc, pose);
}

/// Position and rotate the bones of `pc` along the path attachment shown on its
/// slot. `data` supplies the slots, the default skin, and the bones' setup
/// lengths. A missing slot, slot bone, or path attachment, or a path too short
/// to hold one curve, leaves every bone unchanged. A constrained bone index
/// that is out of range is skipped.
fn apply(
    skeleton: &mut Skeleton,
    data: &SkeletonData,
    pc: &PathConstraintData,
    pose: PathConstraint,
) {
    let slot = pc.slot;
    let Some(slot_data) = data.slots.get(slot) else {
        return;
    };
    let slot_bone = slot_data.bone;
    let Some(att_name) = slot_data.attachment.as_deref() else {
        return;
    };
    let Some(Attachment::Path(path)) = data.attachment(slot, att_name, None) else {
        return;
    };

    let bone_count = pc.bones.len();
    if bone_count == 0 {
        return;
    }
    let tangents = matches!(pc.rotate_mode, RotateMode::Tangent);
    let scale = matches!(pc.rotate_mode, RotateMode::ChainScale);
    let spaces_count = if tangents { bone_count } else { bone_count + 1 };

    // Spacing between bones. `spaces[0]` stays zero. Each loop below fills
    // `spaces[i + 1]` from constrained bone `i`.
    let mut spaces = vec![0.0_f32; spaces_count];
    let mut lengths = if scale {
        vec![0.0_f32; bone_count]
    } else {
        Vec::new()
    };
    let spacing = pose.spacing;
    // A constrained bone that is out of range reads as zero length.
    let bone_length = |i: usize| {
        pc.bones
            .get(i)
            .and_then(|&b| data.bones.get(b))
            .map_or(0.0, |b| b.length)
    };
    let world_len = |skel: &Skeleton, i: usize, setup: f32| {
        let Some(b) = pc.bones.get(i).and_then(|&b| skel.bone(b)) else {
            return 0.0;
        };
        let x = setup * b.a();
        let y = setup * b.c();
        (x * x + y * y).sqrt()
    };
    match pc.spacing_mode {
        SpacingMode::Percent => {
            if scale {
                for (i, l) in lengths.iter_mut().enumerate() {
                    *l = world_len(skeleton, i, bone_length(i));
                }
            }
            for s in spaces.iter_mut().skip(1) {
                *s = spacing;
            }
        }
        SpacingMode::Proportional => {
            let mut sum = 0.0;
            for (i, space) in spaces.iter_mut().skip(1).enumerate() {
                let setup = bone_length(i);
                if setup < EPSILON {
                    *space = spacing;
                } else {
                    let length = world_len(skeleton, i, setup);
                    if let Some(l) = lengths.get_mut(i) {
                        *l = length;
                    }
                    *space = length;
                    sum += length;
                }
            }
            if sum > 0.0 {
                sum = spaces_count as f32 / sum * spacing;
                for s in spaces.iter_mut().skip(1) {
                    *s *= sum;
                }
            }
        }
        SpacingMode::Length | SpacingMode::Fixed => {
            let length_spacing = matches!(pc.spacing_mode, SpacingMode::Length);
            for (i, space) in spaces.iter_mut().skip(1).enumerate() {
                let setup = bone_length(i);
                if setup < EPSILON {
                    *space = spacing;
                } else {
                    let length = world_len(skeleton, i, setup);
                    if let Some(l) = lengths.get_mut(i) {
                        *l = length;
                    }
                    let base = if length_spacing {
                        (setup + spacing).max(0.0)
                    } else {
                        spacing
                    };
                    *space = base * length / setup;
                }
            }
        }
    }

    let world_pts = path.compute_world_vertices(skeleton, slot_bone);
    let Some(positions) = compute_world_positions(
        &world_pts,
        path.closed,
        &spaces,
        tangents,
        pose.position,
        pc.position_mode,
        pc.spacing_mode,
    ) else {
        return;
    };

    // Apply positions and rotations to the bones (world transforms).
    let (mix_rotate, mix_x, mix_y) = (pose.mix_rotate, pose.mix_x, pose.mix_y);
    let [mut bone_x, mut bone_y, ..] = *positions.as_slice() else {
        return;
    };
    let mut offset_rotation = pc.offset_rotation;
    let tip;
    if offset_rotation == 0.0 {
        tip = matches!(pc.rotate_mode, RotateMode::Chain);
    } else {
        tip = false;
        let Some(sb) = skeleton.bone(slot_bone) else {
            return;
        };
        let det = sb.a() * sb.d() - sb.b() * sb.c();
        offset_rotation *= if det > 0.0 { DEG_RAD } else { -DEG_RAD };
    }
    // `positions` holds `3 * spaces_count + 2` values, so every index below is
    // in range: bone `i` reads at most `ip + 2 = 3 * i + 5`.
    for (i, &bone_idx) in pc.bones.iter().enumerate() {
        let ip = 3 * i + 3;
        let x = positions[ip];
        let y = positions[ip + 1];
        let dx = x - bone_x;
        let dy = y - bone_y;
        let (prev_x, prev_y) = (bone_x, bone_y);
        bone_x = x;
        bone_y = y;
        let Some(bn) = skeleton.bone(bone_idx) else {
            continue;
        };
        let (mut a, mut b, mut c, mut d, mut wx, mut wy) =
            (bn.a(), bn.b(), bn.c(), bn.d(), bn.world_x(), bn.world_y());
        wx += (prev_x - wx) * mix_x;
        wy += (prev_y - wy) * mix_y;
        if scale {
            let length = lengths[i];
            if length >= EPSILON {
                let s = ((dx * dx + dy * dy).sqrt() / length - 1.0) * mix_rotate + 1.0;
                a *= s;
                c *= s;
            }
        }
        if mix_rotate > 0.0 {
            let mut r = if tangents {
                positions[ip - 1]
            } else if spaces[i + 1] < EPSILON {
                positions[ip + 2]
            } else {
                dy.atan2(dx)
            };
            r -= c.atan2(a);
            if tip {
                let (sin, cos) = r.sin_cos();
                let length = data.bones.get(bone_idx).map_or(0.0, |bd| bd.length);
                wx += (length * (cos * a - sin * c) - dx) * mix_rotate;
                wy += (length * (sin * a + cos * c) - dy) * mix_rotate;
            } else {
                r += offset_rotation;
            }
            if r > PI {
                r -= TAU;
            } else if r < -PI {
                r += TAU;
            }
            r *= mix_rotate;
            let (sin, cos) = r.sin_cos();
            let (na, nb, nc, nd) = (
                cos * a - sin * c,
                cos * b - sin * d,
                sin * a + cos * c,
                sin * b + cos * d,
            );
            a = na;
            b = nb;
            c = nc;
            d = nd;
        }
        if let Some(bn) = skeleton.bone_mut(bone_idx) {
            bn.set_world(a, b, c, d, wx, wy);
        }
    }
}

/// Sample one position (and tangent angle) per entry of `spaces` along the
/// path's composite cubic Bezier, using constant-speed arc-length
/// parameterization. Returns `[x, y, angle]` per sample plus two trailing
/// zeros, the size Spine uses. The tangent mode has one sample per bone and
/// reads the zeros as the unused next position of the last bone. Returns
/// `None` when `world_pts` is too short to hold one curve.
fn compute_world_positions(
    world_pts: &[Vec2],
    closed: bool,
    spaces: &[f32],
    tangents: bool,
    position: f32,
    position_mode: PositionMode,
    spacing_mode: SpacingMode,
) -> Option<Vec<f32>> {
    let spaces_count = spaces.len();
    let vertices_length = world_pts.len() * 2;

    // Build the flat working vertex array (skips the leading control point,
    // and closed paths wrap the start back onto the end).
    let mut world: Vec<f32> = Vec::new();
    let curve_count = if closed {
        let [first, second, ..] = world_pts else {
            return None;
        };
        for p in world_pts.iter().skip(1) {
            world.push(p.x);
            world.push(p.y);
        }
        world.push(first.x);
        world.push(first.y);
        world.push(second.x);
        world.push(second.y);
        vertices_length / 6
    } else {
        let [_, inner @ .., _] = world_pts else {
            return None;
        };
        for p in inner {
            world.push(p.x);
            world.push(p.y);
        }
        (vertices_length / 6).saturating_sub(1)
    };
    // Curve `k` reads `world[6 * k..6 * k + 8]`. The vertex counts above always
    // back every curve, and this check keeps each read below in range.
    if curve_count == 0 || world.len() < curve_count * 6 + 2 {
        return None;
    }

    // Cumulative curve lengths (coarse 4-sample estimate per curve).
    let mut curves = vec![0.0_f32; curve_count];
    let mut path_length = 0.0_f32;
    let mut x1 = world[0];
    let mut y1 = world[1];
    let mut w = 2;
    for curve in curves.iter_mut() {
        let cx1 = world[w];
        let cy1 = world[w + 1];
        let cx2 = world[w + 2];
        let cy2 = world[w + 3];
        let x2 = world[w + 4];
        let y2 = world[w + 5];
        let tmpx = (x1 - cx1 * 2.0 + cx2) * 0.1875;
        let tmpy = (y1 - cy1 * 2.0 + cy2) * 0.1875;
        let dddfx = ((cx1 - cx2) * 3.0 - x1 + x2) * 0.093_75;
        let dddfy = ((cy1 - cy2) * 3.0 - y1 + y2) * 0.093_75;
        let mut ddfx = tmpx * 2.0 + dddfx;
        let mut ddfy = tmpy * 2.0 + dddfy;
        let mut dfx = (cx1 - x1) * 0.75 + tmpx + dddfx * 0.166_666_67;
        let mut dfy = (cy1 - y1) * 0.75 + tmpy + dddfy * 0.166_666_67;
        path_length += (dfx * dfx + dfy * dfy).sqrt();
        dfx += ddfx;
        dfy += ddfy;
        ddfx += dddfx;
        ddfy += dddfy;
        path_length += (dfx * dfx + dfy * dfy).sqrt();
        dfx += ddfx;
        dfy += ddfy;
        path_length += (dfx * dfx + dfy * dfy).sqrt();
        dfx += ddfx + dddfx;
        dfy += ddfy + dddfy;
        path_length += (dfx * dfx + dfy * dfy).sqrt();
        *curve = path_length;
        x1 = x2;
        y1 = y2;
        w += 6;
    }

    let mut position = position;
    if matches!(position_mode, PositionMode::Percent) {
        position *= path_length;
    }
    let multiplier = match spacing_mode {
        SpacingMode::Percent => path_length,
        SpacingMode::Proportional => path_length / spaces_count as f32,
        _ => 1.0,
    };

    let mut out = vec![0.0_f32; spaces_count * 3 + 2];
    let mut segments = [0.0_f32; 10];
    let mut curve_length = 0.0_f32;
    let mut prev_curve = None;
    let (mut cx1, mut cy1, mut cx2, mut cy2, mut x2, mut y2) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    x1 = world[0];
    y1 = world[1];
    let mut curve = 0usize;
    for (i, &space) in spaces.iter().enumerate() {
        let o = i * 3;
        let space = space * multiplier;
        position += space;
        let mut p = position;

        if closed {
            p %= path_length;
            if p < 0.0 {
                p += path_length;
            }
            curve = 0;
        } else if p < 0.0 {
            add_before_position(p, &world, 0, &mut out, o);
            continue;
        } else if p > path_length {
            add_after_position(p - path_length, &world, world.len() - 4, &mut out, o);
            continue;
        }

        // Curve containing p: the first curve from `curve` on whose cumulative
        // length reaches p, or the last curve. The cumulative lengths never
        // decrease before a NaN and stay NaN after one, so a binary search
        // finds the same curve as a linear scan. It keeps a closed path with
        // many bones and many curves from costing their product.
        curve += curves[curve..curve_count - 1].partition_point(|&length| p > length);
        if curve == 0 {
            p /= curves[0];
        } else {
            let prev = curves[curve - 1];
            p = (p - prev) / (curves[curve] - prev);
        }

        if prev_curve != Some(curve) {
            prev_curve = Some(curve);
            let ii = curve * 6;
            x1 = world[ii];
            y1 = world[ii + 1];
            cx1 = world[ii + 2];
            cy1 = world[ii + 3];
            cx2 = world[ii + 4];
            cy2 = world[ii + 5];
            x2 = world[ii + 6];
            y2 = world[ii + 7];
            let tmpx = (x1 - cx1 * 2.0 + cx2) * 0.03;
            let tmpy = (y1 - cy1 * 2.0 + cy2) * 0.03;
            let dddfx = ((cx1 - cx2) * 3.0 - x1 + x2) * 0.006;
            let dddfy = ((cy1 - cy2) * 3.0 - y1 + y2) * 0.006;
            let mut ddfx = tmpx * 2.0 + dddfx;
            let mut ddfy = tmpy * 2.0 + dddfy;
            let mut dfx = (cx1 - x1) * 0.3 + tmpx + dddfx * 0.166_666_67;
            let mut dfy = (cy1 - y1) * 0.3 + tmpy + dddfy * 0.166_666_67;
            curve_length = (dfx * dfx + dfy * dfy).sqrt();
            segments[0] = curve_length;
            for seg in segments.iter_mut().take(8).skip(1) {
                dfx += ddfx;
                dfy += ddfy;
                ddfx += dddfx;
                ddfy += dddfy;
                curve_length += (dfx * dfx + dfy * dfy).sqrt();
                *seg = curve_length;
            }
            dfx += ddfx;
            dfy += ddfy;
            curve_length += (dfx * dfx + dfy * dfy).sqrt();
            segments[8] = curve_length;
            dfx += ddfx + dddfx;
            dfy += ddfy + dddfy;
            curve_length += (dfx * dfx + dfy * dfy).sqrt();
            segments[9] = curve_length;
        }

        // Segment within the curve, weighted by segment length.
        p *= curve_length;
        let mut segment = 0usize;
        while segment + 1 < 10 && p > segments[segment] {
            segment += 1;
        }
        if segment == 0 {
            p /= segments[0];
        } else {
            let prev = segments[segment - 1];
            p = segment as f32 + (p - prev) / (segments[segment] - prev);
        }
        add_curve_position(
            p * 0.1,
            x1,
            y1,
            cx1,
            cy1,
            cx2,
            cy2,
            x2,
            y2,
            &mut out,
            o,
            tangents || (i > 0 && space < EPSILON),
        );
    }
    Some(out)
}

/// Sample a point before the path start, extrapolating along the first tangent.
fn add_before_position(p: f32, temp: &[f32], i: usize, out: &mut [f32], o: usize) {
    let x1 = temp[i];
    let y1 = temp[i + 1];
    let r = (temp[i + 3] - y1).atan2(temp[i + 2] - x1);
    out[o] = x1 + p * r.cos();
    out[o + 1] = y1 + p * r.sin();
    out[o + 2] = r;
}

/// Sample a point past the path end, extrapolating along the last tangent.
fn add_after_position(p: f32, temp: &[f32], i: usize, out: &mut [f32], o: usize) {
    let x1 = temp[i + 2];
    let y1 = temp[i + 3];
    let r = (y1 - temp[i + 1]).atan2(x1 - temp[i]);
    out[o] = x1 + p * r.cos();
    out[o + 1] = y1 + p * r.sin();
    out[o + 2] = r;
}

/// Sample a cubic Bezier at parameter `p`, writing position (and tangent angle
/// if `tangents`) into `out[o..o+3]`.
#[allow(clippy::too_many_arguments)]
fn add_curve_position(
    p: f32,
    x1: f32,
    y1: f32,
    cx1: f32,
    cy1: f32,
    cx2: f32,
    cy2: f32,
    x2: f32,
    y2: f32,
    out: &mut [f32],
    o: usize,
    tangents: bool,
) {
    if p < EPSILON || p.is_nan() {
        out[o] = x1;
        out[o + 1] = y1;
        out[o + 2] = (cy1 - y1).atan2(cx1 - x1);
        return;
    }
    let tt = p * p;
    let ttt = tt * p;
    let u = 1.0 - p;
    let uu = u * u;
    let uuu = uu * u;
    let ut = u * p;
    let ut3 = ut * 3.0;
    let uut3 = u * ut3;
    let utt3 = ut3 * p;
    let x = x1 * uuu + cx1 * uut3 + cx2 * utt3 + x2 * ttt;
    let y = y1 * uuu + cy1 * uut3 + cy2 * utt3 + y2 * ttt;
    out[o] = x;
    out[o + 1] = y;
    if tangents {
        out[o + 2] = if p < 0.001 {
            (cy1 - y1).atan2(cx1 - x1)
        } else {
            (y - (y1 * uu + cy1 * ut * 2.0 + cy2 * tt))
                .atan2(x - (x1 * uu + cx1 * ut * 2.0 + cx2 * tt))
        };
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::attach::{Attachment, MeshVertices, PathAttachment};
    use crate::data::{BlendMode, BoneData, Color, SkeletonData, SlotData};
    use crate::skel::Skeleton;
    use crate::skin::Skin;

    #[test]
    fn bone_positioned_at_path_midpoint() {
        // Straight-line path (0,0)->(30,0): 6 control points
        // [leading handle, p0, c0, c1, p1, trailing handle].
        let mut skin = Skin::new("default");
        skin.set(
            0,
            "path",
            Attachment::Path(PathAttachment::new(
                "path",
                MeshVertices::Unweighted(vec![
                    -10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 20.0, 0.0, 30.0, 0.0, 40.0, 0.0,
                ]),
                6,
                Vec::new(),
                false,
                true,
            )),
        );
        let data = SkeletonData {
            bones: vec![
                BoneData {
                    index: 0,
                    name: "root".into(),
                    ..Default::default()
                },
                BoneData {
                    index: 1,
                    name: "follower".into(),
                    parent: Some(0),
                    ..Default::default()
                },
            ],
            slots: vec![SlotData {
                index: 0,
                name: "path-slot".into(),
                bone: 0,
                color: Color::WHITE,
                dark_color: None,
                attachment: Some("path".into()),
                blend: BlendMode::Normal,
            }],
            default_skin: skin,
            path_constraints: vec![PathConstraintData {
                name: "follow-path".into(),
                order: 0,
                bones: vec![1],
                slot: 0,
                position_mode: PositionMode::Percent,
                spacing_mode: SpacingMode::Percent,
                rotate_mode: RotateMode::Chain,
                offset_rotation: 0.0,
                position: 0.5,
                spacing: 0.0,
                mix_rotate: 0.0,
                mix_x: 1.0,
                mix_y: 1.0,
            }],
            ..Default::default()
        };
        let mut sk = Skeleton::new(Arc::new(data));
        sk.update_world_transform();
        // 50% along a straight (0,0)->(30,0) path is (15, 0).
        let f = sk.bone(1).unwrap();
        assert!((f.world_x() - 15.0).abs() < 1.0, "x={}", f.world_x());
        assert!(f.world_y().abs() < 0.5, "y={}", f.world_y());
    }

    /// A path constraint on `bones` that follows slot 0 at full mix.
    fn follow(bones: Vec<usize>, rotate_mode: RotateMode) -> PathConstraintData {
        PathConstraintData {
            name: "follow-path".into(),
            order: 0,
            bones,
            slot: 0,
            position_mode: PositionMode::Percent,
            spacing_mode: SpacingMode::Length,
            rotate_mode,
            offset_rotation: 0.0,
            position: 0.5,
            spacing: 0.0,
            mix_rotate: 1.0,
            mix_x: 1.0,
            mix_y: 1.0,
        }
    }

    /// A root, a 10-long follower, and slot 0 on `slot_bone` showing a path
    /// with the given control points.
    fn rig(vertices: MeshVertices, count: usize, closed: bool, slot_bone: usize) -> SkeletonData {
        let mut skin = Skin::new("default");
        skin.set(
            0,
            "path",
            Attachment::Path(PathAttachment::new(
                "path",
                vertices,
                count,
                Vec::new(),
                closed,
                true,
            )),
        );
        SkeletonData {
            bones: vec![
                BoneData {
                    index: 0,
                    name: "root".into(),
                    ..Default::default()
                },
                BoneData {
                    index: 1,
                    name: "follower".into(),
                    parent: Some(0),
                    length: 10.0,
                    ..Default::default()
                },
            ],
            slots: vec![SlotData {
                index: 0,
                name: "path-slot".into(),
                bone: slot_bone,
                color: Color::WHITE,
                dark_color: None,
                attachment: Some("path".into()),
                blend: BlendMode::Normal,
            }],
            default_skin: skin,
            ..Default::default()
        }
    }

    /// Straight 45-degree line from (0, 0) to (30, 30), with handles.
    fn diagonal() -> MeshVertices {
        MeshVertices::Unweighted(vec![
            -10.0, -10.0, 0.0, 0.0, 10.0, 10.0, 20.0, 20.0, 30.0, 30.0, 40.0, 40.0,
        ])
    }

    /// Each bone's world transform.
    fn world(sk: &Skeleton) -> Vec<[f32; 6]> {
        sk.bones()
            .iter()
            .map(|b| [b.a(), b.b(), b.c(), b.d(), b.world_x(), b.world_y()])
            .collect()
    }

    #[test]
    fn tangent_mode_rotates_bone_to_the_path_tangent() {
        let data = SkeletonData {
            path_constraints: vec![follow(vec![1], RotateMode::Tangent)],
            ..rig(diagonal(), 6, false, 0)
        };
        let mut sk = Skeleton::new(Arc::new(data));
        sk.update_world_transform();
        // Halfway along the line, with the x-axis along the 45-degree tangent.
        let f = sk.bone(1).unwrap();
        assert!((f.world_x() - 15.0).abs() < 1.0, "x={}", f.world_x());
        assert!((f.world_y() - 15.0).abs() < 1.0, "y={}", f.world_y());
        assert!((f.a() - 0.707).abs() < 1e-2, "a={}", f.a());
        assert!((f.c() - 0.707).abs() < 1e-2, "c={}", f.c());
    }

    #[test]
    fn path_too_short_for_a_curve_is_ignored() {
        let shapes = [
            (false, 0),
            (false, 1),
            (false, 2),
            (false, 3),
            (false, 5),
            (true, 0),
            (true, 1),
            (true, 2),
        ];
        for (closed, count) in shapes {
            for rotate_mode in [RotateMode::Tangent, RotateMode::Chain] {
                let vertices = MeshVertices::Unweighted(vec![5.0; count * 2]);
                let data = SkeletonData {
                    path_constraints: vec![follow(vec![1], rotate_mode)],
                    ..rig(vertices, count, closed, 0)
                };
                let mut sk = Skeleton::new(Arc::new(data));
                sk.update_world_transform();
                // The follower keeps its setup pose at the origin.
                let f = sk.bone(1).unwrap();
                assert_eq!((f.world_x(), f.world_y()), (0.0, 0.0), "{closed} {count}");
            }
        }
    }

    #[test]
    fn constraint_without_bones_is_ignored() {
        let spacing_modes = [
            SpacingMode::Length,
            SpacingMode::Fixed,
            SpacingMode::Percent,
            SpacingMode::Proportional,
        ];
        for spacing_mode in spacing_modes {
            let mut pc = follow(Vec::new(), RotateMode::Tangent);
            pc.spacing_mode = spacing_mode;
            let data = SkeletonData {
                path_constraints: vec![pc],
                ..rig(diagonal(), 6, false, 0)
            };
            let mut sk = Skeleton::new(Arc::new(data));
            sk.update_world_transform();
        }
    }

    #[test]
    fn out_of_range_indices_are_skipped() {
        // A weighted path does not read the slot bone, so a slot bone past the
        // bone list still yields control points.
        let mut weighted = Vec::new();
        let mut xy = Vec::new();
        for i in 0..6 {
            weighted.extend([1, 0]);
            let v = i as f32 * 10.0 - 10.0;
            xy.extend([v, v, 1.0]);
        }
        let weighted = MeshVertices::Weighted {
            bones: weighted,
            vertices: xy,
        };
        let good = rig(diagonal(), 6, false, 0);
        let mut sk = Skeleton::new(Arc::new(good.clone()));
        sk.update_world_transform();
        let before = world(&sk);

        let mut bad_slot = follow(vec![1], RotateMode::Chain);
        bad_slot.slot = 99;
        let mut bad_offset = follow(vec![1], RotateMode::Chain);
        bad_offset.offset_rotation = 30.0;
        let cases = [
            (bad_slot, good.clone()),
            (follow(vec![99], RotateMode::ChainScale), good.clone()),
            (bad_offset, rig(weighted, 6, false, 99)),
        ];
        let pose = PathConstraint::from_data(&follow(vec![1], RotateMode::Chain));
        for (i, (pc, data)) in cases.into_iter().enumerate() {
            apply(&mut sk, &data, &pc, pose);
            assert_eq!(world(&sk), before, "case {i}");
        }

        // A constraint index past the list does nothing.
        solve(&mut sk, 3, pose);
        assert_eq!(world(&sk), before);

        // A bad bone in the chain is skipped and the valid one still moves.
        apply(
            &mut sk,
            &good,
            &follow(vec![99, 1], RotateMode::Chain),
            pose,
        );
        assert_ne!(world(&sk), before);
    }
}
