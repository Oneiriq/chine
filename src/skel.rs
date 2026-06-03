//! Posable skeleton instances and forward-kinematics world transforms.
//!
//! A [`Skeleton`] is instantiated from a shared [`SkeletonData`] rig. Each
//! frame the host mutates bone *local* poses (directly or via the animation
//! system) and calls [`Skeleton::update_world_transform`], which computes every
//! bone's *world* transform root-to-children. The world transform is a 2x2
//! matrix `(a, b, c, d)` plus a world position `(world_x, world_y)`, the same
//! representation the official Spine runtimes use. After the forward-kinematics
//! pass, IK, transform, path, and physics constraints adjust the pose in order.

use std::sync::Arc;

use crate::constraint::ik::{self, IkConstraint, IkConstraintData};
use crate::constraint::path::{self, PathConstraint, PathConstraintData};
use crate::constraint::physics::{self, Physics, PhysicsConstraint, PhysicsConstraintData};
use crate::constraint::transform::{self, TransformConstraint, TransformConstraintData};
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
    pub(crate) a: f32,
    pub(crate) b: f32,
    pub(crate) c: f32,
    pub(crate) d: f32,
    pub(crate) world_x: f32,
    pub(crate) world_y: f32,
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

    /// Set the full world transform (for constraints that write world space).
    pub(crate) fn set_world(&mut self, a: f32, b: f32, c: f32, d: f32, world_x: f32, world_y: f32) {
        self.a = a;
        self.b = b;
        self.c = c;
        self.d = d;
        self.world_x = world_x;
        self.world_y = world_y;
    }

    /// Offset the world position (used by the physics solver).
    pub(crate) fn add_world_pos(&mut self, dx: f32, dy: f32) {
        self.world_x += dx;
        self.world_y += dy;
    }

    /// Rotate the world matrix's `(a, c)` column by `(sin, cos)`.
    pub(crate) fn rotate_a_c(&mut self, sin: f32, cos: f32) {
        let a = self.a;
        self.a = cos * a - sin * self.c;
        self.c = sin * a + cos * self.c;
    }

    /// Rotate the world matrix's `(b, d)` column by `(sin, cos)`.
    pub(crate) fn rotate_b_d(&mut self, sin: f32, cos: f32) {
        let b = self.b;
        self.b = cos * b - sin * self.d;
        self.d = sin * b + cos * self.d;
    }

    /// Scale the world matrix's `(a, c)` column.
    pub(crate) fn scale_a_c(&mut self, s: f32) {
        self.a *= s;
        self.c *= s;
    }

    /// Scale the world matrix's `(b, d)` column.
    pub(crate) fn scale_b_d(&mut self, s: f32) {
        self.b *= s;
        self.d *= s;
    }
}

/// A posable instance of a [`SkeletonData`] rig. Many skeletons can share one
/// rig via the [`Arc`].
#[derive(Debug, Clone)]
pub struct Skeleton {
    data: Arc<SkeletonData>,
    bones: Vec<Bone>,
    ik_constraints: Vec<IkConstraint>,
    transform_constraints: Vec<TransformConstraint>,
    path_constraints: Vec<PathConstraint>,
    physics_constraints: Vec<PhysicsConstraint>,
    update_cache: Vec<Updatable>,
    // Accumulated simulation time, advanced by `update`, read by physics.
    time: f32,
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
        let transform_constraints = data
            .transform_constraints
            .iter()
            .map(TransformConstraint::from_data)
            .collect();
        let path_constraints = data
            .path_constraints
            .iter()
            .map(PathConstraint::from_data)
            .collect();
        let physics_constraints = data
            .physics_constraints
            .iter()
            .map(PhysicsConstraint::from_data)
            .collect();
        let update_cache = build_update_cache(&data);
        Self {
            data,
            bones,
            ik_constraints,
            transform_constraints,
            path_constraints,
            physics_constraints,
            update_cache,
            time: 0.0,
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

    /// A shared handle to the rig data, for constraints that must read the rig
    /// while also mutating bones (decouples from the `&self` borrow).
    pub(crate) fn data_arc(&self) -> Arc<SkeletonData> {
        Arc::clone(&self.data)
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

    /// A physics constraint's runtime state by index (its mixable tunables).
    #[must_use]
    pub fn physics_constraint(&self, index: usize) -> Option<&PhysicsConstraint> {
        self.physics_constraints.get(index)
    }

    /// Every physics constraint's mutable runtime pose, for global physics
    /// timelines. Pair with [`Self::data_arc`] to read the matching setup data.
    pub(crate) fn physics_constraints_mut(&mut self) -> &mut [PhysicsConstraint] {
        &mut self.physics_constraints
    }

    /// Mark physics constraint `i` to reset on the next world-transform update.
    pub(crate) fn request_physics_reset(&mut self, i: usize) {
        if let Some(c) = self.physics_constraints.get_mut(i) {
            c.set_pending_reset();
        }
    }

    /// Mark every physics constraint to reset on the next world-transform update.
    pub(crate) fn request_all_physics_reset(&mut self) {
        for c in &mut self.physics_constraints {
            c.set_pending_reset();
        }
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

    /// An IK constraint's mutable pose paired with its setup data, for the
    /// animation system.
    pub(crate) fn ik_pose_and_setup(
        &mut self,
        i: usize,
    ) -> Option<(&mut IkConstraint, &IkConstraintData)> {
        let setup = self.data.ik_constraints.get(i)?;
        let pose = self.ik_constraints.get_mut(i)?;
        Some((pose, setup))
    }

    /// A transform constraint's mutable pose paired with its setup data.
    pub(crate) fn transform_pose_and_setup(
        &mut self,
        i: usize,
    ) -> Option<(&mut TransformConstraint, &TransformConstraintData)> {
        let setup = self.data.transform_constraints.get(i)?;
        let pose = self.transform_constraints.get_mut(i)?;
        Some((pose, setup))
    }

    /// A path constraint's mutable pose paired with its setup data.
    pub(crate) fn path_pose_and_setup(
        &mut self,
        i: usize,
    ) -> Option<(&mut PathConstraint, &PathConstraintData)> {
        let setup = self.data.path_constraints.get(i)?;
        let pose = self.path_constraints.get_mut(i)?;
        Some((pose, setup))
    }

    /// A physics constraint's mutable pose paired with its setup data.
    pub(crate) fn physics_pose_and_setup(
        &mut self,
        i: usize,
    ) -> Option<(&mut PhysicsConstraint, &PhysicsConstraintData)> {
        let setup = self.data.physics_constraints.get(i)?;
        let pose = self.physics_constraints.get_mut(i)?;
        Some((pose, setup))
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

    /// Advance the physics simulation clock by `dt` seconds.
    ///
    /// Call once per frame before [`Self::update_world_transform`] so physics
    /// constraints integrate over the elapsed time. Skeletons without physics
    /// constraints need not call this.
    pub fn update(&mut self, dt: f32) {
        self.time += dt;
    }

    /// Pose every bone: walk the update cache, computing each bone's world
    /// transform by forward kinematics and applying each constraint (IK,
    /// transform, path, and physics) in dependency order.
    ///
    /// Constraints modify the pose, so reset bones to their setup or animated
    /// pose each frame (via [`Self::set_bones_to_setup_pose`] or the animation
    /// system) before calling this. Call [`Self::update`] first if physics
    /// constraints should advance.
    pub fn update_world_transform(&mut self) {
        let reference_scale = self.data.reference_scale;
        let time = self.time;
        let len = self.update_cache.len();
        for k in 0..len {
            match self.update_cache[k] {
                Updatable::Bone(i) => self.fk_one(i),
                Updatable::Ik(c) => ik::solve(
                    &mut self.bones,
                    &self.data,
                    c,
                    &self.ik_constraints[c],
                    self.scale_x,
                    self.scale_y,
                ),
                Updatable::Transform(c) => transform::solve(
                    &mut self.bones,
                    &self.data,
                    c,
                    &self.transform_constraints[c],
                    self.scale_x,
                    self.scale_y,
                ),
                Updatable::Path(c) => {
                    let pose = self.path_constraints[c];
                    path::solve(self, c, pose);
                }
                Updatable::Physics(c) => {
                    let mode = if self.physics_constraints[c].take_pending_reset() {
                        Physics::Reset
                    } else {
                        Physics::Update
                    };
                    physics::solve(
                        &mut self.bones,
                        &self.data,
                        c,
                        &mut self.physics_constraints[c],
                        time,
                        reference_scale,
                        mode,
                    );
                }
            }
        }
    }

    /// Compute the world transform for bone `i` from its local pose and parent.
    fn fk_one(&mut self, i: usize) {
        let (sx, sy) = (self.scale_x, self.scale_y);
        let (skel_x, skel_y) = (self.x, self.y);
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

/// One entry in a [`Skeleton`]'s update cache: compute a bone's world transform,
/// or apply a constraint. Built by [`build_update_cache`] in dependency order.
#[derive(Debug, Clone, Copy)]
enum Updatable {
    /// Compute the world transform for the bone at this index.
    Bone(usize),
    /// Apply the IK constraint at this index.
    Ik(usize),
    /// Apply the transform constraint at this index.
    Transform(usize),
    /// Apply the path constraint at this index.
    Path(usize),
    /// Apply the physics constraint at this index.
    Physics(usize),
}

/// Build the ordered update cache: a topological interleaving of bone
/// world-transform updates and constraint applications, mirroring Spine's
/// `Skeleton.updateCache`. Bones a constraint reads are computed before it;
/// bones it modifies are recomputed after.
fn build_update_cache(data: &SkeletonData) -> Vec<Updatable> {
    let n = data.bones.len();
    let parents: Vec<Option<usize>> = data.bones.iter().map(|b| b.parent).collect();
    let mut children: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (i, b) in data.bones.iter().enumerate() {
        if let Some(p) = b.parent {
            children[p].push(i);
        }
    }

    // Merge constraints of all kinds and process them in global order.
    let mut ordered: Vec<(usize, Updatable)> = Vec::new();
    for (i, ik) in data.ik_constraints.iter().enumerate() {
        ordered.push((ik.order, Updatable::Ik(i)));
    }
    for (i, tc) in data.transform_constraints.iter().enumerate() {
        ordered.push((tc.order, Updatable::Transform(i)));
    }
    for (i, pc) in data.path_constraints.iter().enumerate() {
        ordered.push((pc.order, Updatable::Path(i)));
    }
    for (i, pc) in data.physics_constraints.iter().enumerate() {
        ordered.push((pc.order, Updatable::Physics(i)));
    }
    ordered.sort_by_key(|(order, _)| *order);

    let mut sorted = vec![false; n];
    let mut cache = Vec::new();
    for (_, kind) in ordered {
        match kind {
            Updatable::Ik(i) => {
                sort_ik(
                    &data.ik_constraints[i],
                    i,
                    &parents,
                    &children,
                    &mut sorted,
                    &mut cache,
                );
            }
            Updatable::Transform(i) => {
                sort_transform(
                    &data.transform_constraints[i],
                    i,
                    &parents,
                    &children,
                    &mut sorted,
                    &mut cache,
                );
            }
            Updatable::Path(i) => {
                let pc = &data.path_constraints[i];
                let slot_bone = data.slots[pc.slot].bone;
                sort_path(
                    pc,
                    i,
                    slot_bone,
                    &parents,
                    &children,
                    &mut sorted,
                    &mut cache,
                );
            }
            Updatable::Physics(i) => {
                sort_physics(
                    &data.physics_constraints[i],
                    i,
                    &parents,
                    &children,
                    &mut sorted,
                    &mut cache,
                );
            }
            Updatable::Bone(_) => {}
        }
    }
    for i in 0..n {
        sort_bone(i, &parents, &mut sorted, &mut cache);
    }
    cache
}

/// Sort an IK constraint into the cache: its target and constrained bones are
/// computed before it, the constrained bones recomputed after.
fn sort_ik(
    ik: &IkConstraintData,
    idx: usize,
    parents: &[Option<usize>],
    children: &[Vec<usize>],
    sorted: &mut [bool],
    cache: &mut Vec<Updatable>,
) {
    let Some(&parent) = ik.bones.first() else {
        return;
    };
    sort_bone(ik.target, parents, sorted, cache);
    sort_bone(parent, parents, sorted, cache);
    cache.push(Updatable::Ik(idx));
    sorted[parent] = false;
    sort_reset(parent, children, sorted);
}

/// Sort a transform constraint into the cache. For world targets the
/// constrained bones are computed before it and kept (their world is the
/// constraint's output); their descendants are recomputed. For local targets
/// the constrained bones themselves are recomputed afterward.
fn sort_transform(
    tc: &TransformConstraintData,
    idx: usize,
    parents: &[Option<usize>],
    children: &[Vec<usize>],
    sorted: &mut [bool],
    cache: &mut Vec<Updatable>,
) {
    if !tc.local_source {
        sort_bone(tc.source, parents, sorted, cache);
    }
    let world_target = !tc.local_target;
    if world_target {
        for &b in &tc.bones {
            sort_bone(b, parents, sorted, cache);
        }
    }
    cache.push(Updatable::Transform(idx));
    for &b in &tc.bones {
        sort_reset(b, children, sorted);
    }
    for &b in &tc.bones {
        sorted[b] = world_target;
    }
}

/// Sort a path constraint into the cache. The slot bone and constrained bones
/// are computed before it; the constrained bones keep their world (the
/// constraint's output) and their descendants are recomputed.
fn sort_path(
    pc: &PathConstraintData,
    idx: usize,
    slot_bone: usize,
    parents: &[Option<usize>],
    children: &[Vec<usize>],
    sorted: &mut [bool],
    cache: &mut Vec<Updatable>,
) {
    sort_bone(slot_bone, parents, sorted, cache);
    for &b in &pc.bones {
        sort_bone(b, parents, sorted, cache);
    }
    cache.push(Updatable::Path(idx));
    for &b in &pc.bones {
        sort_reset(b, children, sorted);
    }
    for &b in &pc.bones {
        sorted[b] = true;
    }
}

/// Sort a physics constraint into the cache. Its bone is computed before it;
/// the constraint then writes that bone's world transform, so the bone keeps
/// its computed slot (the constraint's output) and only its descendants are
/// recomputed afterward.
fn sort_physics(
    pd: &PhysicsConstraintData,
    idx: usize,
    parents: &[Option<usize>],
    children: &[Vec<usize>],
    sorted: &mut [bool],
    cache: &mut Vec<Updatable>,
) {
    sort_bone(pd.bone, parents, sorted, cache);
    cache.push(Updatable::Physics(idx));
    sort_reset(pd.bone, children, sorted);
}

/// Add `bone` (and any unsorted ancestors) to the cache once, parents first.
fn sort_bone(
    bone: usize,
    parents: &[Option<usize>],
    sorted: &mut [bool],
    cache: &mut Vec<Updatable>,
) {
    if sorted[bone] {
        return;
    }
    if let Some(p) = parents[bone] {
        sort_bone(p, parents, sorted, cache);
    }
    sorted[bone] = true;
    cache.push(Updatable::Bone(bone));
}

/// Mark `bone`'s descendants unsorted so they are recomputed after a constraint.
fn sort_reset(bone: usize, children: &[Vec<usize>], sorted: &mut [bool]) {
    for &child in &children[bone] {
        if sorted[child] {
            sort_reset(child, children, sorted);
        }
        sorted[child] = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraint::ScaleYMode;
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

    // A root at the origin with a "tail" child at (10, 0); the physics
    // constraint drives the tail's world y under gravity.
    fn physics_skel(gravity: f32) -> Skeleton {
        let data = SkeletonData {
            reference_scale: 100.0,
            bones: vec![
                BoneData {
                    index: 0,
                    name: "root".into(),
                    ..Default::default()
                },
                BoneData {
                    index: 1,
                    name: "tail".into(),
                    parent: Some(0),
                    position: Vec2::new(10.0, 0.0),
                    length: 10.0,
                    ..Default::default()
                },
            ],
            physics_constraints: vec![PhysicsConstraintData {
                name: "phys".into(),
                order: 0,
                bone: 1,
                x: 0.0,
                y: 1.0,
                rotate: 0.0,
                scale_x: 0.0,
                shear_x: 0.0,
                limit: 5000.0,
                step: 1.0 / 60.0,
                scale_y_mode: ScaleYMode::None,
                inertia: 1.0,
                strength: 50.0,
                damping: 0.9,
                mass_inverse: 1.0,
                wind: 0.0,
                gravity,
                mix: 1.0,
                inertia_global: false,
                strength_global: false,
                damping_global: false,
                mass_global: false,
                wind_global: false,
                gravity_global: false,
                mix_global: false,
            }],
            ..Default::default()
        };
        Skeleton::new(Arc::new(data))
    }

    fn step_frame(sk: &mut Skeleton) {
        sk.update(1.0 / 60.0);
        sk.set_bones_to_setup_pose();
        sk.update_world_transform();
    }

    #[test]
    fn physics_inert_without_forces() {
        // No gravity, no wind, a stationary skeleton: physics adds no offset.
        let mut sk = physics_skel(0.0);
        for _ in 0..120 {
            step_frame(&mut sk);
        }
        let tail = sk.bone(1).unwrap();
        assert!(close(tail.world_x(), 10.0) && close(tail.world_y(), 0.0));
    }

    #[test]
    fn physics_gravity_droops_and_settles() {
        let mut sk = physics_skel(1.0);
        // The tail starts at world y = 0; run long enough for the spring to
        // settle.
        for _ in 0..600 {
            step_frame(&mut sk);
        }
        let settled = sk.bone(1).unwrap().world_y();
        // Gravity pulls the tail down (negative y).
        assert!(settled < -0.5, "expected droop, got {settled}");
        // Steady state balances gravity against the spring:
        // y_offset = -gravity * reference_scale / strength = -100 / 50 = -2.
        assert!(
            (settled + 2.0).abs() < 0.3,
            "expected settle near -2, got {settled}"
        );
        // Confirm it has converged, not still moving.
        let before = sk.bone(1).unwrap().world_y();
        step_frame(&mut sk);
        let after = sk.bone(1).unwrap().world_y();
        assert!(
            (after - before).abs() < 0.001,
            "not settled: {before} -> {after}"
        );
    }
}
