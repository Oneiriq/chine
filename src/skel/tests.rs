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
