//! Posable skeleton instances and forward-kinematics world transforms.
//!
//! A [`Skeleton`] is instantiated from a shared [`SkeletonData`] rig. Each
//! frame the host mutates bone *local* poses (directly or via the animation
//! system) and calls [`Skeleton::update_world_transform`], which computes every
//! bone's *world* transform root-to-children. The world transform is a 2x2
//! matrix `(a, b, c, d)` plus a world position `(world_x, world_y)` — the same
//! representation the official Spine runtimes use. Constraints and physics are
//! layered on in later milestones; M1 is pure forward kinematics.

use std::sync::Arc;

use crate::constraint::ik::{self, IkConstraint};
use crate::data::{BoneData, Inherit, SkeletonData};

/// Degrees-to-radians factor.
const DEG_RAD: f32 = core::f32::consts::PI / 180.0;
/// Below this squared length the `NoRotationOrReflection` branch treats the
/// parent's x-axis as degenerate (mirrors the reference runtime's epsilon).
const EPSILON_SQ: f32 = 0.0001 * 0.0001;

/// A posable bone: a mutable local pose plus its derived world transform.
///
/// The local fields are read and written freely (the animation system drives
/// them). The world transform (`a`/`b`/`c`/`d`, `world_x`/`world_y`) is
/// *computed* by [`Skeleton::update_world_transform`] and must not be set
/// directly.
#[derive(Debug, Clone)]
pub struct Bone {
    parent: Option<usize>,
    inherit: Inherit,
    /// Local x relative to the parent.
    pub x: f32,
    /// Local y relative to the parent.
    pub y: f32,
    /// Local rotation, in degrees.
    pub rotation: f32,
    /// Local x scale (`1.0` is unscaled).
    pub scale_x: f32,
    /// Local y scale.
    pub scale_y: f32,
    /// Local x shear, in degrees.
    pub shear_x: f32,
    /// Local y shear, in degrees.
    pub shear_y: f32,
    a: f32,
    b: f32,
    c: f32,
    d: f32,
    world_x: f32,
    world_y: f32,
}

impl Bone {
    fn from_data(data: &BoneData) -> Self {
        let mut bone = Self {
            parent: data.parent,
            inherit: data.inherit,
            x: 0.0,
            y: 0.0,
            rotation: 0.0,
            scale_x: 1.0,
            scale_y: 1.0,
            shear_x: 0.0,
            shear_y: 0.0,
            a: 0.0,
            b: 0.0,
            c: 0.0,
            d: 0.0,
            world_x: 0.0,
            world_y: 0.0,
        };
        bone.set_to_setup_pose(data);
        bone
    }

    /// Reset the local pose to the setup pose described by `data`.
    pub fn set_to_setup_pose(&mut self, data: &BoneData) {
        self.x = data.position.x;
        self.y = data.position.y;
        self.rotation = data.rotation;
        self.scale_x = data.scale.x;
        self.scale_y = data.scale.y;
        self.shear_x = data.shear.x;
        self.shear_y = data.shear.y;
    }

    /// This bone's parent index, or `None` for the root.
    #[must_use]
    pub fn parent(&self) -> Option<usize> {
        self.parent
    }

    /// World matrix component `a` (world x-axis x).
    #[must_use]
    pub fn a(&self) -> f32 {
        self.a
    }
    /// World matrix component `b` (world y-axis x).
    #[must_use]
    pub fn b(&self) -> f32 {
        self.b
    }
    /// World matrix component `c` (world x-axis y).
    #[must_use]
    pub fn c(&self) -> f32 {
        self.c
    }
    /// World matrix component `d` (world y-axis y).
    #[must_use]
    pub fn d(&self) -> f32 {
        self.d
    }
    /// World x position.
    #[must_use]
    pub fn world_x(&self) -> f32 {
        self.world_x
    }
    /// World y position.
    #[must_use]
    pub fn world_y(&self) -> f32 {
        self.world_y
    }
}

/// A posable instance of a [`SkeletonData`] rig. Many skeletons can share one
/// rig via the [`Arc`].
#[derive(Debug, Clone)]
pub struct Skeleton {
    data: Arc<SkeletonData>,
    bones: Vec<Bone>,
    ik_constraints: Vec<IkConstraint>,
    /// World-space x offset applied to the whole skeleton.
    pub x: f32,
    /// World-space y offset applied to the whole skeleton.
    pub y: f32,
    /// Whole-skeleton x scale (negative flips horizontally).
    pub scale_x: f32,
    /// Whole-skeleton y scale (negative flips vertically).
    pub scale_y: f32,
}

impl Skeleton {
    /// Instantiate a skeleton from shared rig data, posed at the setup pose.
    #[must_use]
    pub fn new(data: Arc<SkeletonData>) -> Self {
        let bones = data.bones.iter().map(Bone::from_data).collect();
        let ik_constraints = data
            .ik_constraints
            .iter()
            .map(IkConstraint::from_data)
            .collect();
        Self {
            data,
            bones,
            ik_constraints,
            x: 0.0,
            y: 0.0,
            scale_x: 1.0,
            scale_y: 1.0,
        }
    }

    /// The shared rig data this skeleton instances.
    #[must_use]
    pub fn data(&self) -> &SkeletonData {
        &self.data
    }

    /// All bones, in hierarchy order.
    #[must_use]
    pub fn bones(&self) -> &[Bone] {
        &self.bones
    }

    /// A bone by index.
    #[must_use]
    pub fn bone(&self, index: usize) -> Option<&Bone> {
        self.bones.get(index)
    }

    /// Mutable access to a bone's local pose by index.
    pub fn bone_mut(&mut self, index: usize) -> Option<&mut Bone> {
        self.bones.get_mut(index)
    }

    /// Split borrow for the animation system: a bone's mutable local pose paired
    /// with its immutable setup data. `None` if `index` is out of range.
    pub(crate) fn bone_and_setup(&mut self, index: usize) -> Option<(&mut Bone, &BoneData)> {
        let setup = self.data.bones.get(index)?;
        let bone = self.bones.get_mut(index)?;
        Some((bone, setup))
    }

    /// Find a bone index by name.
    #[must_use]
    pub fn find_bone(&self, name: &str) -> Option<usize> {
        self.data.find_bone(name)
    }

    /// Reset every bone to its setup pose.
    pub fn set_bones_to_setup_pose(&mut self) {
        for (bone, data) in self.bones.iter_mut().zip(&self.data.bones) {
            bone.set_to_setup_pose(data);
        }
    }

    /// Compute every bone's world transform by forward kinematics, then apply
    /// the skeleton's constraints (IK; transform / path / physics follow) in
    /// order, re-running FK over each constraint's affected bones.
    ///
    /// Constraints modify the local pose, so reset bones to their setup or
    /// animated pose each frame (via [`Self::set_bones_to_setup_pose`] or the
    /// animation system) before calling this.
    pub fn update_world_transform(&mut self) {
        self.fk_from(0);
        for c in 0..self.ik_constraints.len() {
            if let Some(start) = ik::solve(
                &mut self.bones,
                &self.data,
                c,
                &self.ik_constraints[c],
                self.scale_x,
                self.scale_y,
            ) {
                self.fk_from(start);
            }
        }
    }

    /// Recompute world transforms for bones `start..` from their local pose.
    fn fk_from(&mut self, start: usize) {
        let (sx, sy) = (self.scale_x, self.scale_y);
        let (skel_x, skel_y) = (self.x, self.y);
        for i in start..self.bones.len() {
            let world = match self.bones[i].parent {
                None => root_world(&self.bones[i], skel_x, skel_y, sx, sy),
                Some(p) => {
                    let parent = &self.bones[p];
                    let pose = ParentPose {
                        a: parent.a,
                        b: parent.b,
                        c: parent.c,
                        d: parent.d,
                        world_x: parent.world_x,
                        world_y: parent.world_y,
                    };
                    child_world(&self.bones[i], &pose, sx, sy)
                }
            };
            let bone = &mut self.bones[i];
            bone.a = world.0;
            bone.b = world.1;
            bone.c = world.2;
            bone.d = world.3;
            bone.world_x = world.4;
            bone.world_y = world.5;
        }
    }
}

/// A parent bone's already-computed world transform, copied out so the child's
/// computation doesn't borrow the bone array.
struct ParentPose {
    a: f32,
    b: f32,
    c: f32,
    d: f32,
    world_x: f32,
    world_y: f32,
}

type World = (f32, f32, f32, f32, f32, f32);

/// Root bone: local pose composed only with the skeleton's position and scale.
fn root_world(b: &Bone, skel_x: f32, skel_y: f32, sx: f32, sy: f32) -> World {
    let rx = (b.rotation + b.shear_x) * DEG_RAD;
    let ry = (b.rotation + 90.0 + b.shear_y) * DEG_RAD;
    let a = rx.cos() * b.scale_x * sx;
    let bb = ry.cos() * b.scale_y * sx;
    let c = rx.sin() * b.scale_x * sy;
    let d = ry.sin() * b.scale_y * sy;
    (a, bb, c, d, b.x * sx + skel_x, b.y * sy + skel_y)
}

/// Child bone: local pose composed with the parent world transform per the
/// bone's [`Inherit`] mode. Transcribed from Spine 4.3 `BonePose`.
fn child_world(b: &Bone, p: &ParentPose, sx: f32, sy: f32) -> World {
    let world_x = p.a * b.x + p.b * b.y + p.world_x;
    let world_y = p.c * b.x + p.d * b.y + p.world_y;

    match b.inherit {
        Inherit::Normal => {
            let rx = (b.rotation + b.shear_x) * DEG_RAD;
            let ry = (b.rotation + 90.0 + b.shear_y) * DEG_RAD;
            let la = rx.cos() * b.scale_x;
            let lb = ry.cos() * b.scale_y;
            let lc = rx.sin() * b.scale_x;
            let ld = ry.sin() * b.scale_y;
            (
                p.a * la + p.b * lc,
                p.a * lb + p.b * ld,
                p.c * la + p.d * lc,
                p.c * lb + p.d * ld,
                world_x,
                world_y,
            )
        }
        Inherit::OnlyTranslation => {
            let rx = (b.rotation + b.shear_x) * DEG_RAD;
            let ry = (b.rotation + 90.0 + b.shear_y) * DEG_RAD;
            (
                rx.cos() * b.scale_x * sx,
                ry.cos() * b.scale_y * sx,
                rx.sin() * b.scale_x * sy,
                ry.sin() * b.scale_y * sy,
                world_x,
                world_y,
            )
        }
        Inherit::NoRotationOrReflection => {
            let sxi = 1.0 / sx;
            let syi = 1.0 / sy;
            let mut pa = p.a * sxi;
            let mut pc = p.c * syi;
            let mut pb = p.b;
            let mut pd = p.d;
            let s = pa * pa + pc * pc;
            let r;
            if s > EPSILON_SQ {
                let s2 = (pa * pd * syi - pb * sxi * pc).abs() / s;
                pb = pc * s2;
                pd = pa * s2;
                r = b.rotation - pc.atan2(pa).to_degrees();
            } else {
                pa = 0.0;
                pc = 0.0;
                r = b.rotation - 90.0 + pd.atan2(pb).to_degrees();
            }
            let rx = (r + b.shear_x) * DEG_RAD;
            let ry = (r + b.shear_y + 90.0) * DEG_RAD;
            let la = rx.cos() * b.scale_x;
            let lb = ry.cos() * b.scale_y;
            let lc = rx.sin() * b.scale_x;
            let ld = ry.sin() * b.scale_y;
            (
                (pa * la - pb * lc) * sx,
                (pa * lb - pb * ld) * sx,
                (pc * la + pd * lc) * sy,
                (pc * lb + pd * ld) * sy,
                world_x,
                world_y,
            )
        }
        Inherit::NoScale | Inherit::NoScaleOrReflection => {
            let sxi = 1.0 / sx;
            let syi = 1.0 / sy;
            let r = b.rotation * DEG_RAD;
            let (sin_r, cos_r) = r.sin_cos();
            let mut za = (p.a * cos_r + p.b * sin_r) * sxi;
            let mut zc = (p.c * cos_r + p.d * sin_r) * syi;
            let s = 1.0 / (za * za + zc * zc).sqrt();
            za *= s;
            zc *= s;
            let mut zb = -zc;
            let mut zd = za;
            // For NoScale (but not NoScaleOrReflection), flip if the parent's
            // determinant sign disagrees with the skeleton's reflection.
            if matches!(b.inherit, Inherit::NoScale)
                && (p.a * p.d - p.b * p.c < 0.0) != ((sx < 0.0) != (sy < 0.0))
            {
                zb = -zb;
                zd = -zd;
            }
            let rx = b.shear_x * DEG_RAD;
            let ry = (90.0 + b.shear_y) * DEG_RAD;
            let la = rx.cos() * b.scale_x;
            let lb = ry.cos() * b.scale_y;
            let lc = rx.sin() * b.scale_x;
            let ld = ry.sin() * b.scale_y;
            (
                (za * la + zb * lc) * sx,
                (za * lb + zb * ld) * sx,
                (zc * la + zd * lc) * sy,
                (zc * lb + zd * ld) * sy,
                world_x,
                world_y,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::BoneData;
    use glam::Vec2;

    const EPS: f32 = 1e-4;

    fn chain() -> Skeleton {
        // root at (10, 20); child "arm" offset (5, 0) from root.
        let data = SkeletonData {
            bones: vec![
                BoneData {
                    index: 0,
                    name: "root".into(),
                    position: Vec2::new(10.0, 20.0),
                    ..Default::default()
                },
                BoneData {
                    index: 1,
                    name: "arm".into(),
                    parent: Some(0),
                    position: Vec2::new(5.0, 0.0),
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        Skeleton::new(Arc::new(data))
    }

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < EPS
    }

    #[test]
    fn setup_pose_root_is_identity_at_position() {
        let mut sk = chain();
        sk.update_world_transform();
        let root = sk.bone(0).unwrap();
        // rotation 0, scale 1 -> identity matrix.
        assert!(close(root.a(), 1.0) && close(root.b(), 0.0));
        assert!(close(root.c(), 0.0) && close(root.d(), 1.0));
        assert!(close(root.world_x(), 10.0) && close(root.world_y(), 20.0));
    }

    #[test]
    fn child_inherits_parent_world_position() {
        let mut sk = chain();
        sk.update_world_transform();
        let arm = sk.bone(1).unwrap();
        // child offset (5,0) under an unrotated root at (10,20) -> (15,20).
        assert!(close(arm.world_x(), 15.0) && close(arm.world_y(), 20.0));
        assert!(close(arm.a(), 1.0) && close(arm.d(), 1.0));
    }

    #[test]
    fn rotating_root_rotates_and_orbits_child() {
        let mut sk = chain();
        sk.bone_mut(0).unwrap().rotation = 90.0;
        sk.update_world_transform();
        // root world matrix is a +90deg rotation: [0,-1; 1,0].
        let root = sk.bone(0).unwrap();
        assert!(close(root.a(), 0.0) && close(root.b(), -1.0));
        assert!(close(root.c(), 1.0) && close(root.d(), 0.0));
        // the child's (5,0) local offset now points along +y -> (10, 25).
        let arm = sk.bone(1).unwrap();
        assert!(close(arm.world_x(), 10.0) && close(arm.world_y(), 25.0));
    }

    #[test]
    fn skeleton_scale_and_position_offset_the_root() {
        let mut sk = chain();
        sk.x = 100.0;
        sk.scale_x = 2.0;
        sk.update_world_transform();
        let root = sk.bone(0).unwrap();
        assert!(close(root.world_x(), 10.0 * 2.0 + 100.0)); // 120
        assert!(close(root.a(), 2.0)); // local identity * skeleton scale_x
    }

    #[test]
    fn only_translation_child_ignores_parent_rotation() {
        let mut data = (*chain().data()).clone();
        data.bones[1].inherit = Inherit::OnlyTranslation;
        let mut sk = Skeleton::new(Arc::new(data));
        sk.bone_mut(0).unwrap().rotation = 90.0;
        sk.update_world_transform();
        // position still follows the parent (orbits), but orientation stays
        // axis-aligned (identity matrix), not rotated.
        let arm = sk.bone(1).unwrap();
        assert!(close(arm.a(), 1.0) && close(arm.b(), 0.0));
        assert!(close(arm.c(), 0.0) && close(arm.d(), 1.0));
        assert!(close(arm.world_x(), 10.0) && close(arm.world_y(), 25.0));
    }
}
