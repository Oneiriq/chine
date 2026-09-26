//! Transform constraints.
//!
//! A [`TransformConstraint`] copies a source bone's transform onto the
//! constrained bones via a configurable property map: each source property
//! (rotate / x / y / scaleX / scaleY / shearY) drives one or more target
//! properties, scaled and offset. World-target constraints modify the
//! constrained bones' world matrices directly. Local-target ones modify their
//! local pose. It re-implements Spine 4.3's `TransformConstraintData`.

use core::f32::consts::{PI, TAU};

use super::clamp;
use crate::data::SkeletonData;
use crate::skel::Bone;

/// `offsets` indices, matching Spine's `TransformConstraintData` constants.
const ROTATION: usize = 0;
const X: usize = 1;
const Y: usize = 2;
const SCALEX: usize = 3;
const SCALEY: usize = 4;
const SHEARY: usize = 5;

/// A source transform property read from the source bone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FromProp {
    /// World or local rotation.
    Rotate,
    /// World or local x.
    X,
    /// World or local y.
    Y,
    /// World or local x scale.
    ScaleX,
    /// World or local y scale.
    ScaleY,
    /// World or local y shear.
    ShearY,
}

/// A constrained transform property written to a target bone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToProp {
    /// Rotation.
    Rotate,
    /// X.
    X,
    /// Y.
    Y,
    /// X scale.
    ScaleX,
    /// Y scale.
    ScaleY,
    /// Y shear.
    ShearY,
}

/// One target property mapping: `offset + sourceValue * scale`, optionally
/// clamped to `[offset, max]`.
#[derive(Debug, Clone)]
pub struct ToMapping {
    /// The target property.
    pub property: ToProp,
    /// Value of this property corresponding to the source property's offset.
    pub offset: f32,
    /// Clamp ceiling when clamping is enabled.
    pub max: f32,
    /// Scale of the source value in relation to this property.
    pub scale: f32,
}

/// One source property and the target properties it drives.
#[derive(Debug, Clone)]
pub struct FromMapping {
    /// The source property.
    pub property: FromProp,
    /// Source value corresponding to each target's offset.
    pub offset: f32,
    /// Targets this source drives.
    pub to: Vec<ToMapping>,
}

/// Setup data for a transform constraint.
#[derive(Debug, Clone)]
pub struct TransformConstraintData {
    /// Constraint name.
    pub name: String,
    /// Global constraint order (lower applies first).
    pub order: usize,
    /// Constrained bone indices.
    pub bones: Vec<usize>,
    /// Source bone index.
    pub source: usize,
    /// Per-property offsets (`ROTATION`..`SHEARY`).
    pub offsets: [f32; 6],
    /// Read the source's local transform instead of its world transform.
    pub local_source: bool,
    /// Write the constrained bones' local transforms instead of world.
    pub local_target: bool,
    /// Add to the constrained transform instead of setting it.
    pub additive: bool,
    /// Clamp target values to each mapping's `[offset, max]`.
    pub clamp: bool,
    /// Source-to-target property mappings.
    pub properties: Vec<FromMapping>,
    /// Setup mix per property.
    pub mix_rotate: f32,
    /// See [`Self::mix_rotate`].
    pub mix_x: f32,
    /// See [`Self::mix_rotate`].
    pub mix_y: f32,
    /// See [`Self::mix_rotate`].
    pub mix_scale_x: f32,
    /// See [`Self::mix_rotate`].
    pub mix_scale_y: f32,
    /// See [`Self::mix_rotate`].
    pub mix_shear_y: f32,
}

/// A runtime transform-constraint pose (the mixes are animatable).
#[derive(Debug, Clone)]
pub(crate) struct TransformConstraint {
    /// Rotation mix in `[0, 1]`.
    pub mix_rotate: f32,
    /// X mix.
    pub mix_x: f32,
    /// Y mix.
    pub mix_y: f32,
    /// X scale mix.
    pub mix_scale_x: f32,
    /// Y scale mix.
    pub mix_scale_y: f32,
    /// Y shear mix.
    pub mix_shear_y: f32,
}

impl TransformConstraint {
    /// Build a runtime pose from setup data.
    #[must_use]
    pub(crate) fn from_data(data: &TransformConstraintData) -> Self {
        Self {
            mix_rotate: data.mix_rotate,
            mix_x: data.mix_x,
            mix_y: data.mix_y,
            mix_scale_x: data.mix_scale_x,
            mix_scale_y: data.mix_scale_y,
            mix_shear_y: data.mix_shear_y,
        }
    }
}

/// The source bone's transform, snapshotted before any constrained bone is
/// modified (the source is read once).
struct SourcePose {
    a: f32,
    b: f32,
    c: f32,
    d: f32,
    world_x: f32,
    world_y: f32,
    rotation: f32,
    x: f32,
    y: f32,
    scale_x: f32,
    scale_y: f32,
    shear_y: f32,
}

/// Apply transform constraint `c`, copying the source bone's mapped properties
/// onto the constrained bones. World-target writes modify the bones' world
/// matrices. Local-target writes modify their local pose (recomputed by the
/// update cache). An out-of-range constraint or source index skips the
/// constraint, and an out-of-range constrained bone is skipped.
pub(crate) fn solve(
    bones: &mut [Bone],
    data: &SkeletonData,
    c: usize,
    pose: &TransformConstraint,
    sx: f32,
    sy: f32,
) {
    let Some(tc) = data.transform_constraints.get(c) else {
        return;
    };
    if pose.mix_rotate == 0.0
        && pose.mix_x == 0.0
        && pose.mix_y == 0.0
        && pose.mix_scale_x == 0.0
        && pose.mix_scale_y == 0.0
        && pose.mix_shear_y == 0.0
    {
        return;
    }
    let Some(s) = bones.get(tc.source) else {
        return;
    };
    let src = SourcePose {
        a: s.a(),
        b: s.b(),
        c: s.c(),
        d: s.d(),
        world_x: s.world_x(),
        world_y: s.world_y(),
        rotation: s.rotation,
        x: s.x,
        y: s.y,
        scale_x: s.scale_x,
        scale_y: s.scale_y,
        shear_y: s.shear_y,
    };
    for &bone_idx in &tc.bones {
        let Some(bone) = bones.get_mut(bone_idx) else {
            continue;
        };
        for from in &tc.properties {
            let value =
                from_value(from.property, &src, tc.local_source, &tc.offsets, sx, sy) - from.offset;
            for to in &from.to {
                if to_mix(to.property, pose) == 0.0 {
                    continue;
                }
                let mut clamped = to.offset + value * to.scale;
                if tc.clamp {
                    // A NaN offset or max makes `f32::clamp` panic, so use the
                    // total `clamp` helper.
                    clamped = if to.offset < to.max {
                        clamp(clamped, to.offset, to.max)
                    } else {
                        clamp(clamped, to.max, to.offset)
                    };
                }
                to_apply(
                    to.property,
                    bone,
                    pose,
                    clamped,
                    tc.local_target,
                    tc.additive,
                    sx,
                    sy,
                );
            }
        }
    }
}

/// Read a source property (world or local).
fn from_value(
    prop: FromProp,
    s: &SourcePose,
    local: bool,
    offsets: &[f32; 6],
    sx: f32,
    sy: f32,
) -> f32 {
    match prop {
        FromProp::Rotate => {
            if local {
                return s.rotation + offsets[ROTATION];
            }
            let det = s.a * s.d - s.b * s.c;
            let off = if det * sx * sy > 0.0 {
                offsets[ROTATION]
            } else {
                -offsets[ROTATION]
            };
            let mut value = (s.c / sy).atan2(s.a / sx).to_degrees() + off;
            if value < 0.0 {
                value += 360.0;
            }
            value
        }
        FromProp::X => {
            if local {
                s.x + offsets[X]
            } else {
                (offsets[X] * s.a + offsets[Y] * s.b + s.world_x) / sx
            }
        }
        FromProp::Y => {
            if local {
                s.y + offsets[Y]
            } else {
                (offsets[X] * s.c + offsets[Y] * s.d + s.world_y) / sy
            }
        }
        FromProp::ScaleX => {
            if local {
                s.scale_x + offsets[SCALEX]
            } else {
                let a = s.a / sx;
                let c = s.c / sy;
                (a * a + c * c).sqrt() + offsets[SCALEX]
            }
        }
        FromProp::ScaleY => {
            if local {
                s.scale_y + offsets[SCALEY]
            } else {
                let b = s.b / sx;
                let d = s.d / sy;
                (b * b + d * d).sqrt() + offsets[SCALEY]
            }
        }
        FromProp::ShearY => {
            if local {
                s.shear_y + offsets[SHEARY]
            } else {
                let ix = 1.0 / sx;
                let iy = 1.0 / sy;
                (s.d * iy).atan2(s.b * ix).to_degrees()
                    - (s.c * iy).atan2(s.a * ix).to_degrees()
                    - 90.0
                    + offsets[SHEARY]
            }
        }
    }
}

/// The mix weight for a target property.
fn to_mix(prop: ToProp, pose: &TransformConstraint) -> f32 {
    match prop {
        ToProp::Rotate => pose.mix_rotate,
        ToProp::X => pose.mix_x,
        ToProp::Y => pose.mix_y,
        ToProp::ScaleX => pose.mix_scale_x,
        ToProp::ScaleY => pose.mix_scale_y,
        ToProp::ShearY => pose.mix_shear_y,
    }
}

/// Write a target property (world or local) on `bone`.
#[allow(clippy::too_many_arguments)]
fn to_apply(
    prop: ToProp,
    bone: &mut Bone,
    pose: &TransformConstraint,
    value: f32,
    local: bool,
    additive: bool,
    sx: f32,
    sy: f32,
) {
    match prop {
        ToProp::Rotate => {
            let mix = pose.mix_rotate;
            if local {
                bone.rotation += rel(value, bone.rotation, additive) * mix;
            } else {
                let ix = 1.0 / sx;
                let iy = 1.0 / sy;
                let a = bone.a * ix;
                let b = bone.b * ix;
                let c = bone.c * iy;
                let d = bone.d * iy;
                let mut v = value.to_radians();
                if !additive {
                    v -= c.atan2(a);
                }
                v = wrap_pi(v) * mix;
                let (sin, cos) = v.sin_cos();
                bone.a = (cos * a - sin * c) * sx;
                bone.b = (cos * b - sin * d) * sx;
                bone.c = (sin * a + cos * c) * sy;
                bone.d = (sin * b + cos * d) * sy;
            }
        }
        ToProp::X => {
            let mix = pose.mix_x;
            if local {
                bone.x += rel(value, bone.x, additive) * mix;
            } else {
                let mut v = value;
                if !additive {
                    v -= bone.world_x / sx;
                }
                bone.world_x += v * mix * sx;
            }
        }
        ToProp::Y => {
            let mix = pose.mix_y;
            if local {
                bone.y += rel(value, bone.y, additive) * mix;
            } else {
                let mut v = value;
                if !additive {
                    v -= bone.world_y / sy;
                }
                bone.world_y += v * mix * sy;
            }
        }
        ToProp::ScaleX => {
            let mix = pose.mix_scale_x;
            if local {
                if additive {
                    bone.scale_x *= 1.0 + (value - 1.0) * mix;
                } else if bone.scale_x != 0.0 {
                    bone.scale_x += (value - bone.scale_x) * mix;
                }
            } else if additive {
                let s = 1.0 + (value - 1.0) * mix;
                bone.a *= s;
                bone.c *= s;
            } else {
                let a = bone.a / sx;
                let c = bone.c / sy;
                let s = (a * a + c * c).sqrt();
                if s != 0.0 {
                    let s = 1.0 + (value - s) * mix / s;
                    bone.a *= s;
                    bone.c *= s;
                }
            }
        }
        ToProp::ScaleY => {
            let mix = pose.mix_scale_y;
            if local {
                if additive {
                    bone.scale_y *= 1.0 + (value - 1.0) * mix;
                } else if bone.scale_y != 0.0 {
                    bone.scale_y += (value - bone.scale_y) * mix;
                }
            } else if additive {
                let s = 1.0 + (value - 1.0) * mix;
                bone.b *= s;
                bone.d *= s;
            } else {
                let b = bone.b / sx;
                let d = bone.d / sy;
                let s = (b * b + d * d).sqrt();
                if s != 0.0 {
                    let s = 1.0 + (value - s) * mix / s;
                    bone.b *= s;
                    bone.d *= s;
                }
            }
        }
        ToProp::ShearY => {
            let mix = pose.mix_shear_y;
            if local {
                let mut v = value;
                if !additive {
                    v -= bone.shear_y;
                }
                bone.shear_y += v * mix;
            } else {
                let b = bone.b / sx;
                let d = bone.d / sy;
                let by = d.atan2(b);
                let mut v = (value + 90.0).to_radians();
                if additive {
                    v -= PI / 2.0;
                } else {
                    v -= by - (bone.c / sy).atan2(bone.a / sx);
                    v = wrap_pi(v);
                }
                let v = by + v * mix;
                let s = (b * b + d * d).sqrt();
                bone.b = v.cos() * s * sx;
                bone.d = v.sin() * s * sy;
            }
        }
    }
}

/// Relative delta for a local-target write: the value as-is when additive,
/// otherwise relative to the current value.
fn rel(value: f32, current: f32, additive: bool) -> f32 {
    if additive {
        value
    } else {
        value - current
    }
}

/// Wrap a radian angle into `(-PI, PI]`.
fn wrap_pi(v: f32) -> f32 {
    if v > PI {
        v - TAU
    } else if v < -PI {
        v + TAU
    } else {
        v
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use glam::Vec2;

    use super::*;
    use crate::data::{BoneData, SkeletonData};
    use crate::skel::Skeleton;

    fn bone(index: usize, name: &str, parent: Option<usize>, rotation: f32) -> BoneData {
        BoneData {
            index,
            name: name.into(),
            parent,
            rotation,
            position: Vec2::ZERO,
            ..Default::default()
        }
    }

    /// A world-mode transform constraint that maps `source` rotation to the
    /// constrained bones' rotation at the given mix.
    fn follow_rotation(
        bones: Vec<usize>,
        source: usize,
        mix_rotate: f32,
    ) -> TransformConstraintData {
        TransformConstraintData {
            name: "follow".into(),
            order: 0,
            bones,
            source,
            offsets: [0.0; 6],
            local_source: false,
            local_target: false,
            additive: false,
            clamp: false,
            properties: vec![FromMapping {
                property: FromProp::Rotate,
                offset: 0.0,
                to: vec![ToMapping {
                    property: ToProp::Rotate,
                    offset: 0.0,
                    max: 1.0,
                    scale: 1.0,
                }],
            }],
            mix_rotate,
            mix_x: 0.0,
            mix_y: 0.0,
            mix_scale_x: 0.0,
            mix_scale_y: 0.0,
            mix_shear_y: 0.0,
        }
    }

    #[test]
    fn world_rotate_copies_source_rotation() {
        let data = SkeletonData {
            bones: vec![
                bone(0, "root", None, 0.0),
                bone(1, "source", Some(0), 90.0),
                bone(2, "follower", Some(0), 0.0),
            ],
            transform_constraints: vec![follow_rotation(vec![2], 1, 1.0)],
            ..Default::default()
        };
        let mut sk = Skeleton::new(Arc::new(data));
        sk.update_world_transform();
        // follower's world now matches the source's +90deg rotation: a~0, c~1.
        let f = sk.bone(2).unwrap();
        assert!(f.a().abs() < 1e-3, "a={}", f.a());
        assert!((f.c() - 1.0).abs() < 1e-3, "c={}", f.c());
    }

    #[test]
    fn half_mix_blends_toward_source() {
        let data = SkeletonData {
            bones: vec![
                bone(0, "root", None, 0.0),
                bone(1, "source", Some(0), 90.0),
                bone(2, "follower", Some(0), 0.0),
            ],
            transform_constraints: vec![follow_rotation(vec![2], 1, 0.5)],
            ..Default::default()
        };
        let mut sk = Skeleton::new(Arc::new(data));
        sk.update_world_transform();
        // Halfway between 0 and 90 degrees: world x-axis at 45deg, a=c~0.707.
        let f = sk.bone(2).unwrap();
        assert!((f.a() - 0.707).abs() < 1e-2, "a={}", f.a());
        assert!((f.c() - 0.707).abs() < 1e-2, "c={}", f.c());
    }

    #[test]
    fn out_of_range_source_or_bone_is_skipped() {
        let data = SkeletonData {
            bones: vec![
                bone(0, "root", None, 0.0),
                bone(1, "source", Some(0), 90.0),
                bone(2, "follower", Some(0), 0.0),
            ],
            ..Default::default()
        };
        let mut sk = Skeleton::new(Arc::new(data.clone()));
        sk.update_world_transform();
        let before = sk.bones().to_vec();
        let bad = SkeletonData {
            transform_constraints: vec![
                follow_rotation(vec![2], 99, 1.0),
                follow_rotation(vec![99, 2], 1, 1.0),
            ],
            ..data
        };
        let pose = TransformConstraint::from_data(&bad.transform_constraints[0]);
        let world = |bones: &[Bone]| -> Vec<[f32; 4]> {
            bones.iter().map(|b| [b.a(), b.b(), b.c(), b.d()]).collect()
        };

        // A source past the bone list, or a constraint past the constraint
        // list, changes nothing.
        let mut bones = before.clone();
        solve(&mut bones, &bad, 0, &pose, 1.0, 1.0);
        solve(&mut bones, &bad, 2, &pose, 1.0, 1.0);
        assert_eq!(world(&bones), world(&before));

        // A constrained bone past the bone list is skipped. The valid one
        // still follows the source.
        solve(&mut bones, &bad, 1, &pose, 1.0, 1.0);
        assert!(bones[2].a().abs() < 1e-3, "a={}", bones[2].a());
        assert!((bones[2].c() - 1.0).abs() < 1e-3, "c={}", bones[2].c());
    }

    #[test]
    fn clamp_with_nan_bounds_does_not_panic() {
        for (offset, max) in [(f32::NAN, 1.0), (0.0, f32::NAN), (f32::NAN, f32::NAN)] {
            for local_target in [false, true] {
                let mut tc = follow_rotation(vec![2], 1, 1.0);
                tc.clamp = true;
                tc.local_target = local_target;
                tc.properties[0].to[0].offset = offset;
                tc.properties[0].to[0].max = max;
                let data = SkeletonData {
                    bones: vec![
                        bone(0, "root", None, 0.0),
                        bone(1, "source", Some(0), 90.0),
                        bone(2, "follower", Some(0), 0.0),
                    ],
                    transform_constraints: vec![tc],
                    ..Default::default()
                };
                let mut sk = Skeleton::new(Arc::new(data));
                sk.update_world_transform();
            }
        }
    }
}
