//! Physics constraints: a spring-damper simulation that adds secondary motion.
//!
//! A [`PhysicsConstraint`] makes a bone lag, swing, and settle in response to
//! the skeleton's movement (hair, cloth, antennae, and similar jiggle). Unlike
//! the IK, transform, and path constraints it is *stateful*: it carries velocity
//! and offset across frames and integrates over a time delta, and it reads and
//! writes the bone's world transform directly.
//!
//! The integrator re-implements Spine 4.3's `PhysicsConstraint`, using the
//! fixed-timestep formulation with scalar wind and gravity. It omits Spine's
//! lag interpolation between the last two steps, so the pose shows the latest
//! whole step and can move in small jumps when frames are shorter than a step.

use super::{clamp, ScaleYMode};
use crate::data::SkeletonData;
use crate::skel::Bone;

/// `2 * PI`.
const PI2: f32 = core::f32::consts::PI * 2.0;
/// `1 / (2 * PI)`.
const INV_PI2: f32 = 1.0 / PI2;
/// Spine's default `referenceScale` when none is set.
const DEFAULT_REFERENCE_SCALE: f32 = 100.0;
/// Most fixed steps one solve integrates (about 18 minutes at 60 steps per
/// second). A larger backlog, from a huge `dt` passed to `Skeleton::update` or
/// a tiny step, is dropped so each solve does bounded work.
const MAX_STEPS: u32 = 1 << 16;

/// How physics advances on an `update_world_transform` pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum Physics {
    /// Leave physics untouched this pass.
    None,
    /// Reset accumulated physics state, then advance once.
    Reset,
    /// Advance the simulation by the elapsed time (the usual mode).
    #[default]
    Update,
    /// Re-apply the current physics pose without advancing time.
    Pose,
}

/// Setup data for a physics constraint.
#[derive(Debug, Clone)]
pub struct PhysicsConstraintData {
    /// Constraint name, unique within the skeleton.
    pub name: String,
    /// Global constraint order (lower applies first).
    pub order: usize,
    /// Whether the constraint applies only while the active skin lists it.
    pub skin_required: bool,
    /// Constrained bone index.
    pub bone: usize,
    /// Strength of the effect on the bone's world x position (`0` disables).
    pub x: f32,
    /// Strength of the effect on the bone's world y position.
    pub y: f32,
    /// Strength of the effect on the bone's rotation.
    pub rotate: f32,
    /// Strength of the effect on the bone's x scale.
    pub scale_x: f32,
    /// Strength of the effect on the bone's x shear.
    pub shear_x: f32,
    /// Maximum world movement per step before clamping.
    pub limit: f32,
    /// Fixed integration timestep, in seconds. A step that is not positive
    /// never advances the simulation.
    pub step: f32,
    /// How Y scale reacts when X scale changes.
    pub scale_y_mode: ScaleYMode,
    /// Setup inertia (how much the bone follows its parent, `0..1`).
    pub inertia: f32,
    /// Setup spring strength pulling the bone back to rest.
    pub strength: f32,
    /// Setup velocity damping per simulated second (`0..1`).
    pub damping: f32,
    /// Setup inverse mass (higher reacts faster).
    pub mass_inverse: f32,
    /// Setup horizontal wind acceleration.
    pub wind: f32,
    /// Setup vertical gravity acceleration.
    pub gravity: f32,
    /// Setup mix (`0` disables, `1` full).
    pub mix: f32,
    /// Whether a global inertia timeline drives this constraint.
    pub inertia_global: bool,
    /// Whether a global strength timeline drives this constraint.
    pub strength_global: bool,
    /// Whether a global damping timeline drives this constraint.
    pub damping_global: bool,
    /// Whether a global mass timeline drives this constraint.
    pub mass_global: bool,
    /// Whether a global wind timeline drives this constraint.
    pub wind_global: bool,
    /// Whether a global gravity timeline drives this constraint.
    pub gravity_global: bool,
    /// Whether a global mix timeline drives this constraint.
    pub mix_global: bool,
}

/// A runtime physics constraint: the mixable tunables plus simulation state
/// that persists across frames.
#[derive(Debug, Clone)]
pub struct PhysicsConstraint {
    /// Inertia (mixable by animation).
    pub inertia: f32,
    /// Spring strength (mixable).
    pub strength: f32,
    /// Damping (mixable).
    pub damping: f32,
    /// Inverse mass (mixable).
    pub mass_inverse: f32,
    /// Wind acceleration (mixable).
    pub wind: f32,
    /// Gravity acceleration (mixable).
    pub gravity: f32,
    /// Mix weight in `[0, 1]` (mixable).
    pub mix: f32,

    // Simulation state, carried across frames.
    reset: bool,
    pending_reset: bool,
    ux: f32,
    uy: f32,
    cx: f32,
    cy: f32,
    tx: f32,
    ty: f32,
    x_offset: f32,
    x_velocity: f32,
    y_offset: f32,
    y_velocity: f32,
    rotate_offset: f32,
    rotate_velocity: f32,
    scale_offset: f32,
    scale_velocity: f32,
    remaining: f32,
    last_time: f32,
}

impl PhysicsConstraint {
    /// Build a runtime constraint from setup data, at rest.
    pub(crate) fn from_data(data: &PhysicsConstraintData) -> Self {
        Self {
            inertia: data.inertia,
            strength: data.strength,
            damping: data.damping,
            mass_inverse: data.mass_inverse,
            wind: data.wind,
            gravity: data.gravity,
            mix: data.mix,
            reset: true,
            pending_reset: false,
            ux: 0.0,
            uy: 0.0,
            cx: 0.0,
            cy: 0.0,
            tx: 0.0,
            ty: 0.0,
            x_offset: 0.0,
            x_velocity: 0.0,
            y_offset: 0.0,
            y_velocity: 0.0,
            rotate_offset: 0.0,
            rotate_velocity: 0.0,
            scale_offset: 0.0,
            scale_velocity: 0.0,
            remaining: 0.0,
            last_time: 0.0,
        }
    }

    /// Reset accumulated simulation state, taking `time` as the new baseline.
    fn reset_state(&mut self, time: f32) {
        self.remaining = 0.0;
        self.last_time = time;
        self.reset = true;
        self.x_offset = 0.0;
        self.x_velocity = 0.0;
        self.y_offset = 0.0;
        self.y_velocity = 0.0;
        self.rotate_offset = 0.0;
        self.rotate_velocity = 0.0;
        self.scale_offset = 0.0;
        self.scale_velocity = 0.0;
    }

    /// Request a reset on the next `update_world_transform` (set by the physics
    /// reset timeline).
    pub(crate) fn set_pending_reset(&mut self) {
        self.pending_reset = true;
    }

    /// Take and clear the pending-reset flag.
    pub(crate) fn take_pending_reset(&mut self) -> bool {
        core::mem::take(&mut self.pending_reset)
    }
}

/// Advance and apply physics constraint `c`, reading and writing its bone's
/// world transform. `pose` carries simulation state, so it is mutated. An
/// out-of-range constraint or bone index skips the constraint.
pub(crate) fn solve(
    bones: &mut [Bone],
    data: &SkeletonData,
    c: usize,
    pose: &mut PhysicsConstraint,
    time: f32,
    reference_scale: f32,
    physics: Physics,
) {
    let Some(pd) = data.physics_constraints.get(c) else {
        return;
    };
    let mix = pose.mix;
    if mix == 0.0 || physics == Physics::None {
        return;
    }
    let use_x = pd.x > 0.0;
    let use_y = pd.y > 0.0;
    let rotate_or_shear = pd.rotate > 0.0 || pd.shear_x > 0.0;
    let use_scale = pd.scale_x > 0.0;
    let (Some(bone_data), Some(bone)) = (data.bones.get(pd.bone), bones.get_mut(pd.bone)) else {
        return;
    };
    let l = bone_data.length;

    if physics == Physics::Reset {
        pose.reset_state(time);
    }

    if physics == Physics::Update || physics == Physics::Reset {
        let delta = (time - pose.last_time).max(0.0);
        pose.remaining += delta;
        pose.last_time = time;

        let bx = bone.world_x();
        let by = bone.world_y();
        if pose.reset {
            pose.reset = false;
            pose.ux = bx;
            pose.uy = by;
        } else {
            let i = pose.inertia;
            let t = pd.step;
            let f = if reference_scale > 0.0 {
                reference_scale
            } else {
                DEFAULT_REFERENCE_SCALE
            };
            let q = pd.limit * delta;
            let mut a = pose.remaining;
            let mut d = -1.0_f32;

            if use_x || use_y {
                if use_x {
                    pose.x_offset += (pose.ux - bx) * i;
                    pose.ux = bx;
                }
                if use_y {
                    pose.y_offset += (pose.uy - by) * i;
                    pose.uy = by;
                }
                if can_step(a, t) {
                    d = pose.damping.powf(60.0 * t);
                    let m = pose.mass_inverse * t;
                    let e = pose.strength;
                    let w = pose.wind * f;
                    let g = pose.gravity * f;
                    let mut steps = 0;
                    loop {
                        if use_x {
                            pose.x_velocity += (w - pose.x_offset * e) * m;
                            pose.x_offset += pose.x_velocity * t;
                            pose.x_velocity *= d;
                        }
                        if use_y {
                            pose.y_velocity -= (g + pose.y_offset * e) * m;
                            pose.y_offset += pose.y_velocity * t;
                            pose.y_velocity *= d;
                        }
                        a -= t;
                        steps += 1;
                        if last_step(&mut a, t, steps) {
                            break;
                        }
                    }
                }
                if use_x {
                    bone.add_world_pos(pose.x_offset * mix * pd.x, 0.0);
                }
                if use_y {
                    bone.add_world_pos(0.0, pose.y_offset * mix * pd.y);
                }
            }

            if rotate_or_shear || use_scale {
                let (ba, bc, bwx, bwy) = (bone.a(), bone.c(), bone.world_x(), bone.world_y());
                let ca = bc.atan2(ba);
                let mut cc;
                let mut s;
                let mut mr = 0.0;
                // A negative or NaN limit, or an infinite delta, makes `q` a
                // bound that `f32::clamp` rejects with a panic.
                let dx = clamp(pose.cx - bwx, -q, q);
                let dy = clamp(pose.cy - bwy, -q, q);
                let world_scale_x = ba.hypot(bc);

                if rotate_or_shear {
                    mr = (pd.rotate + pd.shear_x) * mix;
                    let mut r = (dy + pose.ty).atan2(dx + pose.tx) - ca - pose.rotate_offset * mr;
                    pose.rotate_offset += (r - (r * INV_PI2 - 0.5).ceil() * PI2) * i;
                    r = pose.rotate_offset * mr + ca;
                    cc = r.cos();
                    s = r.sin();
                    if use_scale {
                        let rr = l * world_scale_x;
                        if rr > 0.0 {
                            pose.scale_offset += (dx * cc + dy * s) * i / rr;
                        }
                    }
                } else {
                    cc = ca.cos();
                    s = ca.sin();
                    let rr = l * world_scale_x;
                    if rr > 0.0 {
                        pose.scale_offset += (dx * cc + dy * s) * i / rr;
                    }
                }

                a = pose.remaining;
                if can_step(a, t) {
                    if d == -1.0 {
                        d = pose.damping.powf(60.0 * t);
                    }
                    let m = pose.mass_inverse * t;
                    let e = pose.strength;
                    let wind = pose.wind;
                    let gravity = pose.gravity;
                    let h = l / f;
                    let mut steps = 0;
                    loop {
                        a -= t;
                        steps += 1;
                        let last = last_step(&mut a, t, steps);
                        if use_scale {
                            pose.scale_velocity +=
                                (wind * cc - gravity * s - pose.scale_offset * e) * m;
                            pose.scale_offset += pose.scale_velocity * t;
                            pose.scale_velocity *= d;
                        }
                        if rotate_or_shear {
                            pose.rotate_velocity -=
                                ((wind * s + gravity * cc) * h + pose.rotate_offset * e) * m;
                            pose.rotate_offset += pose.rotate_velocity * t;
                            pose.rotate_velocity *= d;
                            if last {
                                break;
                            }
                            let r = pose.rotate_offset * mr + ca;
                            cc = r.cos();
                            s = r.sin();
                        } else if last {
                            break;
                        }
                    }
                }
            }
            pose.remaining = a;
        }
        pose.cx = bone.world_x();
        pose.cy = bone.world_y();
    } else if physics == Physics::Pose {
        if use_x {
            bone.add_world_pos(pose.x_offset * mix * pd.x, 0.0);
        }
        if use_y {
            bone.add_world_pos(0.0, pose.y_offset * mix * pd.y);
        }
    }

    if rotate_or_shear {
        apply_rotate_shear(bone, pd, pose.rotate_offset * mix);
    }
    if use_scale {
        apply_scale(bone, pd, pose.scale_offset * mix);
    }

    if physics != Physics::Pose {
        pose.tx = l * bone.a();
        pose.ty = l * bone.c();
    }
}

/// Whether `remaining` time holds at least one fixed step of length `t`. A
/// step that is not positive never drains the remaining time, so it never
/// steps.
fn can_step(remaining: f32, t: f32) -> bool {
    t > 0.0 && remaining >= t
}

/// Whether the step loop ends after step number `steps`: less than one step of
/// time remains, or the loop reached [`MAX_STEPS`]. At the cap the rest of the
/// backlog is dropped, so `remaining` is set to zero.
fn last_step(remaining: &mut f32, t: f32, steps: u32) -> bool {
    if *remaining < t {
        return true;
    }
    if steps >= MAX_STEPS {
        *remaining = 0.0;
        return true;
    }
    false
}

/// Apply the accumulated rotation / shear offset `o` to the bone's world matrix.
fn apply_rotate_shear(bone: &mut Bone, pd: &PhysicsConstraintData, o: f32) {
    if pd.shear_x > 0.0 {
        let mut r = 0.0;
        if pd.rotate > 0.0 {
            r = o * pd.rotate;
            let (s, cc) = r.sin_cos();
            bone.rotate_b_d(s, cc);
        }
        r += o * pd.shear_x;
        let (s, cc) = r.sin_cos();
        bone.rotate_a_c(s, cc);
    } else {
        let (s, cc) = (o * pd.rotate).sin_cos();
        bone.rotate_a_c(s, cc);
        bone.rotate_b_d(s, cc);
    }
}

/// Apply the accumulated scale offset to the bone's world matrix.
fn apply_scale(bone: &mut Bone, pd: &PhysicsConstraintData, scale_mix: f32) {
    let mut s = 1.0 + scale_mix * pd.scale_x;
    bone.scale_a_c(s);
    match pd.scale_y_mode {
        ScaleYMode::None => {}
        ScaleYMode::Uniform => bone.scale_b_d(s),
        ScaleYMode::Volume => {
            s = s.abs();
            s = if s >= 0.7 {
                1.0 / s
            } else {
                4.0 - 3.673_47 * s
            };
            bone.scale_b_d(s);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use glam::Vec2;

    use super::*;
    use crate::data::BoneData;
    use crate::skel::Skeleton;

    /// A physics constraint on `bone` that drives every property.
    fn physics(bone: usize) -> PhysicsConstraintData {
        PhysicsConstraintData {
            name: "jiggle".into(),
            order: 0,
            skin_required: false,
            bone,
            x: 1.0,
            y: 1.0,
            rotate: 1.0,
            scale_x: 1.0,
            shear_x: 1.0,
            limit: 5000.0,
            step: 1.0 / 60.0,
            scale_y_mode: ScaleYMode::Volume,
            inertia: 0.5,
            strength: 100.0,
            damping: 0.9,
            mass_inverse: 1.0,
            wind: 1.0,
            gravity: 1.0,
            mix: 1.0,
            inertia_global: false,
            strength_global: false,
            damping_global: false,
            mass_global: false,
            wind_global: false,
            gravity_global: false,
            mix_global: false,
        }
    }

    /// A root with one 10-long child bone.
    fn bones() -> Vec<BoneData> {
        vec![
            BoneData {
                index: 0,
                name: "root".into(),
                ..Default::default()
            },
            BoneData {
                index: 1,
                name: "tail".into(),
                parent: Some(0),
                length: 10.0,
                position: Vec2::new(5.0, 0.0),
                ..Default::default()
            },
        ]
    }

    /// Pose the rig once per `dt`, moving the skeleton so the simulation has
    /// motion to respond to. Returns the skeleton for inspection.
    fn run(pd: PhysicsConstraintData, dts: &[f32]) -> Skeleton {
        let data = SkeletonData {
            bones: bones(),
            physics_constraints: vec![pd],
            ..Default::default()
        };
        let mut sk = Skeleton::new(Arc::new(data));
        for (i, &dt) in dts.iter().enumerate() {
            sk.set_bones_to_setup_pose();
            sk.x = i as f32 * 3.0;
            sk.update(dt);
            sk.update_world_transform();
        }
        sk
    }

    #[test]
    fn moving_the_skeleton_moves_the_bone_off_its_rest_pose() {
        let sk = run(physics(1), &[1.0 / 60.0; 4]);
        let tail = sk.bone(1).unwrap();
        // Without physics the tail would sit at (sk.x + 5, 0) = (14, 0).
        let moved = (tail.world_x() - 14.0).abs() + tail.world_y().abs();
        assert!(
            moved > 1e-3,
            "tail at ({}, {})",
            tail.world_x(),
            tail.world_y()
        );
    }

    #[test]
    fn non_positive_step_does_not_hang() {
        for step in [0.0, -1.0, f32::NAN, f32::NEG_INFINITY] {
            let mut pd = physics(1);
            pd.step = step;
            run(pd, &[1.0 / 60.0, 1.0, 1.0]);
        }
    }

    #[test]
    fn tiny_step_does_not_hang() {
        // A huge JSON `fps` gives a step too small to drain 1/60 s.
        for step in [1e-30, f32::MIN_POSITIVE / 4.0] {
            let mut pd = physics(1);
            pd.step = step;
            run(pd, &[1.0 / 60.0, 1.0 / 60.0, 1.0 / 60.0]);
        }
    }

    #[test]
    fn huge_or_non_finite_time_does_not_hang() {
        for dt in [1e10, f32::MAX, f32::INFINITY, f32::NAN, -1.0] {
            run(physics(1), &[1.0 / 60.0, dt, 1.0 / 60.0, 1.0 / 60.0]);
        }
    }

    #[test]
    fn step_cap_drops_the_backlog() {
        let data = SkeletonData {
            bones: bones(),
            physics_constraints: vec![physics(1)],
            ..Default::default()
        };
        let mut sk = Skeleton::new(Arc::new(data.clone()));
        sk.update_world_transform();
        let mut bones = sk.bones().to_vec();
        let mut pose = PhysicsConstraint::from_data(&data.physics_constraints[0]);
        // The first solve only records the rest position.
        solve(&mut bones, &data, 0, &mut pose, 0.0, 100.0, Physics::Update);
        solve(&mut bones, &data, 0, &mut pose, 1e9, 100.0, Physics::Update);
        assert!(
            pose.remaining < 1.0 / 60.0,
            "remaining = {}",
            pose.remaining
        );
    }

    #[test]
    fn bad_limit_does_not_panic() {
        for limit in [-1.0, f32::NAN, f32::NEG_INFINITY, f32::INFINITY, 0.0] {
            let mut pd = physics(1);
            pd.limit = limit;
            run(pd, &[1.0 / 60.0, 1.0 / 60.0, f32::INFINITY, 1.0 / 60.0]);
        }
    }

    #[test]
    fn out_of_range_bone_or_constraint_is_skipped() {
        let data = SkeletonData {
            bones: bones(),
            ..Default::default()
        };
        let mut sk = Skeleton::new(Arc::new(data.clone()));
        sk.update_world_transform();
        let before: Vec<[f32; 6]> = sk
            .bones()
            .iter()
            .map(|b| [b.a(), b.b(), b.c(), b.d(), b.world_x(), b.world_y()])
            .collect();
        let bad = SkeletonData {
            physics_constraints: vec![physics(99)],
            ..data
        };
        let mut bones = sk.bones().to_vec();
        let mut pose = PhysicsConstraint::from_data(&bad.physics_constraints[0]);
        for time in [0.0, 1.0, 2.0] {
            // The last index is past the constraint list.
            for c in 0..2 {
                solve(&mut bones, &bad, c, &mut pose, time, 100.0, Physics::Update);
            }
        }
        let after: Vec<[f32; 6]> = bones
            .iter()
            .map(|b| [b.a(), b.b(), b.c(), b.d(), b.world_x(), b.world_y()])
            .collect();
        assert_eq!(after, before);
    }
}
