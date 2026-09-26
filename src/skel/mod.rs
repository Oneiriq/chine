//! Posable skeleton instances and forward-kinematics world transforms.
//!
//! A [`Skeleton`] is instantiated from a shared [`SkeletonData`] rig. Each
//! frame the host mutates bone *local* poses (directly or via the animation
//! system) and calls [`Skeleton::update_world_transform`], which computes every
//! bone's *world* transform root-to-children. The world transform is a 2x2
//! matrix `(a, b, c, d)` plus a world position `(world_x, world_y)`, the same
//! representation the official Spine runtimes use. After the forward-kinematics
//! pass, IK, transform, path, physics, and slider constraints adjust the pose
//! in order.

use std::sync::Arc;

use crate::constraint::ik::{self, IkConstraint, IkConstraintData};
use crate::constraint::path::{self, PathConstraint, PathConstraintData};
use crate::constraint::physics::{self, Physics, PhysicsConstraint, PhysicsConstraintData};
use crate::constraint::slider::{self, SliderData, SliderPose};
use crate::constraint::transform::{self, TransformConstraint, TransformConstraintData};
use crate::data::{BoneData, Color, Inherit, SkeletonData, SlotData};
use crate::event::Event;
use crate::skin::Skin;

mod cache;

use cache::{build_update_cache, valid_parent, Updatable};

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
    // How the bone inherits its parent's transform. Starts at the setup
    // mode, and a bone inherit timeline can change it.
    pub(crate) inherit: Inherit,
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
    /// A bone at `data`'s setup pose in a rig of `bone_count` bones. A parent
    /// index outside the rig is dropped, so the bone poses as a root.
    fn from_data(data: &BoneData, bone_count: usize) -> Self {
        let mut bone = Self {
            parent: valid_parent(data.parent, bone_count),
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

    /// Reset the local pose, and the inherit mode, to the setup pose described
    /// by `data`.
    pub fn set_to_setup_pose(&mut self, data: &BoneData) {
        self.inherit = data.inherit;
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

    /// How this bone currently inherits its parent's transform: the setup
    /// mode, unless a bone inherit timeline has changed it.
    #[must_use]
    pub fn inherit(&self) -> Inherit {
        self.inherit
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

/// A posable slot: its current tint, dark tint, and shown attachment. Reset to
/// the [`SlotData`] setup each frame, then driven by slot timelines.
#[derive(Debug, Clone)]
pub struct Slot {
    /// Tint color multiplied into the attachment.
    pub color: Color,
    /// Optional dark color for two-color (tint-black) rendering.
    pub dark_color: Option<Color>,
    /// Name of the attachment currently shown, if any.
    pub attachment: Option<String>,
    /// Deformed local vertex positions for the current mesh attachment, set by a
    /// deform timeline (empty = use the attachment's setup vertices).
    pub deform: Vec<f32>,
    /// Current sequence frame index (`-1` uses the attachment's setup index),
    /// set by a sequence timeline.
    pub sequence_index: i32,
}

impl Slot {
    fn from_data(data: &SlotData) -> Self {
        Self {
            color: data.color,
            dark_color: data.dark_color,
            attachment: data.attachment.clone(),
            deform: Vec::new(),
            sequence_index: -1,
        }
    }

    /// Reset to the setup tint, dark tint, and attachment, and clear the deform.
    fn set_to_setup_pose(&mut self, data: &SlotData) {
        self.color = data.color;
        self.dark_color = data.dark_color;
        self.attachment = data.attachment.clone();
        self.deform.clear();
        self.sequence_index = -1;
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
    sliders: Vec<SliderPose>,
    slots: Vec<Slot>,
    draw_order: Vec<usize>,
    // Active named skin (index into `data.skins`). `None` uses the default skin.
    skin: Option<usize>,
    events: Vec<Event>,
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
        let bone_count = data.bones.len();
        let bones = data
            .bones
            .iter()
            .map(|b| Bone::from_data(b, bone_count))
            .collect();
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
        let sliders = data.sliders.iter().map(SliderPose::from_data).collect();
        let slots = data.slots.iter().map(Slot::from_data).collect();
        let draw_order: Vec<usize> = (0..data.slots.len()).collect();
        let update_cache = build_update_cache(&data);
        Self {
            data,
            bones,
            ik_constraints,
            transform_constraints,
            path_constraints,
            physics_constraints,
            sliders,
            slots,
            draw_order,
            skin: None,
            events: Vec::new(),
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

    /// Set the active skin by name: it overrides the default skin for attachment
    /// lookups (so linked meshes and other variants in that skin are shown). An
    /// unknown name (or `"default"`) clears the active skin.
    pub fn set_skin(&mut self, name: &str) {
        self.skin = self.data.skins.iter().position(|s| s.name == name);
    }

    /// Clear the active skin, using only the default skin.
    pub fn clear_skin(&mut self) {
        self.skin = None;
    }

    /// The active named skin, if one is set.
    #[must_use]
    pub fn active_skin(&self) -> Option<&Skin> {
        self.skin.and_then(|i| self.data.skins.get(i))
    }

    /// The index (into `data().skins`) of the active named skin, if one is set.
    pub(crate) fn active_skin_index(&self) -> Option<usize> {
        self.skin
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

    /// A slot's runtime state by index.
    #[must_use]
    pub fn slot(&self, index: usize) -> Option<&Slot> {
        self.slots.get(index)
    }

    /// The runtime slots, indexed by slot index (not draw order).
    #[must_use]
    pub fn slots(&self) -> &[Slot] {
        &self.slots
    }

    /// The current draw order: slot indices, back to front.
    #[must_use]
    pub fn draw_order(&self) -> &[usize] {
        &self.draw_order
    }

    /// Replace the current draw order (slot indices, back to front), for the
    /// draw-order timeline.
    pub(crate) fn set_draw_order(&mut self, order: &[usize]) {
        self.draw_order.clear();
        self.draw_order.extend_from_slice(order);
    }

    /// The current draw order, to reorder in place (for the draw order folder
    /// timeline).
    pub(crate) fn draw_order_mut(&mut self) -> &mut [usize] {
        &mut self.draw_order
    }

    /// The events fired by the animation since the last apply (footsteps, hit
    /// frames, audio cues).
    #[must_use]
    pub fn events(&self) -> &[Event] {
        &self.events
    }

    /// Record a fired event (used by the event timeline).
    pub(crate) fn push_event(&mut self, event: Event) {
        self.events.push(event);
    }

    /// Clear the fired-event list.
    ///
    /// [`crate::anim::AnimationState::apply`] clears it before each apply. A
    /// host that calls [`crate::anim::Animation::apply`] directly calls this
    /// first, so [`Self::events`] holds only the events that apply fired.
    pub fn clear_events(&mut self) {
        self.events.clear();
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

    /// A slot's mutable runtime pose paired with its setup data, for slot
    /// timelines.
    pub(crate) fn slot_pose_and_setup(&mut self, i: usize) -> Option<(&mut Slot, &SlotData)> {
        let setup = self.data.slots.get(i)?;
        let slot = self.slots.get_mut(i)?;
        Some((slot, setup))
    }

    /// A slider's runtime pose (its current scrub time and mix). An unknown
    /// index reads as a pose with zero mix, which applies nothing.
    pub(crate) fn slider_pose(&self, c: usize) -> SliderPose {
        self.sliders.get(c).copied().unwrap_or(SliderPose {
            time: 0.0,
            mix: 0.0,
        })
    }

    /// A slider's mutable pose paired with its setup data, for the animation
    /// system (the SLIDER_TIME / SLIDER_MIX timelines).
    pub(crate) fn slider_pose_and_setup(
        &mut self,
        i: usize,
    ) -> Option<(&mut SliderPose, &SliderData)> {
        let setup = self.data.sliders.get(i)?;
        let pose = self.sliders.get_mut(i)?;
        Some((pose, setup))
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

    /// Reset every slot's tint, dark tint, and attachment to its setup, and the
    /// draw order to setup order. Call each frame before applying slot timelines.
    pub fn set_slots_to_setup_pose(&mut self) {
        for (slot, data) in self.slots.iter_mut().zip(&self.data.slots) {
            slot.set_to_setup_pose(data);
        }
        self.draw_order.clear();
        self.draw_order.extend(0..self.slots.len());
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
        // The cache only names bones and constraints of this rig (see
        // `build_update_cache`), so the lookups below always succeed. They are
        // checked anyway, and a miss skips the entry.
        let len = self.update_cache.len();
        for k in 0..len {
            let Some(&entry) = self.update_cache.get(k) else {
                break;
            };
            match entry {
                Updatable::Bone(i) => self.fk_one(i),
                Updatable::Ik(c) => {
                    if let Some(pose) = self.ik_constraints.get(c) {
                        ik::solve(
                            &mut self.bones,
                            &self.data,
                            c,
                            pose,
                            self.scale_x,
                            self.scale_y,
                        );
                    }
                }
                Updatable::Transform(c) => {
                    if let Some(pose) = self.transform_constraints.get(c) {
                        transform::solve(
                            &mut self.bones,
                            &self.data,
                            c,
                            pose,
                            self.scale_x,
                            self.scale_y,
                        );
                    }
                }
                Updatable::Path(c) => {
                    if let Some(&pose) = self.path_constraints.get(c) {
                        path::solve(self, c, pose);
                    }
                }
                Updatable::Physics(c) => {
                    if let Some(pose) = self.physics_constraints.get_mut(c) {
                        let mode = if pose.take_pending_reset() {
                            Physics::Reset
                        } else {
                            Physics::Update
                        };
                        physics::solve(
                            &mut self.bones,
                            &self.data,
                            c,
                            pose,
                            time,
                            reference_scale,
                            mode,
                        );
                    }
                }
                Updatable::Slider(c) => slider::solve(self, c),
            }
        }
    }

    /// Compute the world transform for bone `i` from its local pose and parent.
    /// An unknown bone is ignored.
    fn fk_one(&mut self, i: usize) {
        let (sx, sy) = (self.scale_x, self.scale_y);
        let (skel_x, skel_y) = (self.x, self.y);
        let Some(bone) = self.bones.get(i) else {
            return;
        };
        // `Skeleton::new` drops out-of-range parents, so a named parent exists.
        let world = match bone.parent.and_then(|p| self.bones.get(p)) {
            None => root_world(bone, skel_x, skel_y, sx, sy),
            Some(parent) => {
                let pose = ParentPose {
                    a: parent.a,
                    b: parent.b,
                    c: parent.c,
                    d: parent.d,
                    world_x: parent.world_x,
                    world_y: parent.world_y,
                };
                child_world(bone, &pose, sx, sy)
            }
        };
        let Some(bone) = self.bones.get_mut(i) else {
            return;
        };
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
/// bone's [`Inherit`] mode.
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
mod tests;
