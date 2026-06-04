//! Physics constraints: a spring-damper simulation that adds secondary motion.
//!
//! A [`PhysicsConstraint`] makes a bone lag, swing, and settle in response to
//! the skeleton's movement (hair, cloth, antennae, and similar jiggle). Unlike
//! the IK, transform, and path constraints it is *stateful*: it carries velocity
//! and offset across frames and integrates over a time delta, and it reads and
//! writes the bone's world transform directly.
//!
//! The integrator is adapted from Spine 4.3 `PhysicsConstraint`, using the
//! fixed-timestep formulation with scalar wind and gravity and no inter-frame
//! lag interpolation (chine drives a fixed delta, so lag smoothing is moot).

use super::ScaleYMode;
use crate::data::SkeletonData;
use crate::skel::Bone;

/// `2 * PI`.
const PI2: f32 = core::f32::consts::PI * 2.0;
/// `1 / (2 * PI)`.
const INV_PI2: f32 = 1.0 / PI2;
/// Spine's default `referenceScale` when none is set.
const DEFAULT_REFERENCE_SCALE: f32 = 100.0;

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
    /// Fixed integration timestep, in seconds.
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
/// world transform. `pose` carries simulation state, so it is mutated.
pub(crate) fn solve(
    bones: &mut [Bone],
    data: &SkeletonData,
    c: usize,
    pose: &mut PhysicsConstraint,
    time: f32,
    reference_scale: f32,
    physics: Physics,
) {
    let pd = &data.physics_constraints[c];
    let mix = pose.mix;
    if mix == 0.0 || physics == Physics::None {
        return;
    }
    let use_x = pd.x > 0.0;
    let use_y = pd.y > 0.0;
    let rotate_or_shear = pd.rotate > 0.0 || pd.shear_x > 0.0;
    let use_scale = pd.scale_x > 0.0;
    let bi = pd.bone;
    let l = data.bones[bi].length;

    if physics == Physics::Reset {
        pose.reset_state(time);
    }

    if physics == Physics::Update || physics == Physics::Reset {
        let delta = (time - pose.last_time).max(0.0);
        pose.remaining += delta;
        pose.last_time = time;

        let bx = bones[bi].world_x();
        let by = bones[bi].world_y();
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
                if a >= t {
                    d = pose.damping.powf(60.0 * t);
                    let m = pose.mass_inverse * t;
                    let e = pose.strength;
                    let w = pose.wind * f;
                    let g = pose.gravity * f;
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
                        if a < t {
                            break;
                        }
                    }
                }
                if use_x {
                    bones[bi].add_world_pos(pose.x_offset * mix * pd.x, 0.0);
                }
                if use_y {
                    bones[bi].add_world_pos(0.0, pose.y_offset * mix * pd.y);
                }
            }

            if rotate_or_shear || use_scale {
                let (ba, bc, bwx, bwy) = {
                    let b = &bones[bi];
                    (b.a(), b.c(), b.world_x(), b.world_y())
                };
                let ca = bc.atan2(ba);
                let mut cc;
                let mut s;
                let mut mr = 0.0;
                let dx = (pose.cx - bwx).clamp(-q, q);
                let dy = (pose.cy - bwy).clamp(-q, q);
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
                if a >= t {
                    if d == -1.0 {
                        d = pose.damping.powf(60.0 * t);
                    }
                    let m = pose.mass_inverse * t;
                    let e = pose.strength;
                    let wind = pose.wind;
                    let gravity = pose.gravity;
                    let h = l / f;
                    loop {
                        a -= t;
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
                            if a < t {
                                break;
                            }
                            let r = pose.rotate_offset * mr + ca;
                            cc = r.cos();
                            s = r.sin();
                        } else if a < t {
                            break;
                        }
                    }
                }
            }
            pose.remaining = a;
        }
        pose.cx = bones[bi].world_x();
        pose.cy = bones[bi].world_y();
    } else if physics == Physics::Pose {
        if use_x {
            bones[bi].add_world_pos(pose.x_offset * mix * pd.x, 0.0);
        }
        if use_y {
            bones[bi].add_world_pos(0.0, pose.y_offset * mix * pd.y);
        }
    }

    if rotate_or_shear {
        apply_rotate_shear(&mut bones[bi], pd, pose.rotate_offset * mix);
    }
    if use_scale {
        apply_scale(&mut bones[bi], pd, pose.scale_offset * mix);
    }

    if physics != Physics::Pose {
        pose.tx = l * bones[bi].a();
        pose.ty = l * bones[bi].c();
    }
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
