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

fn bone(index: usize, parent: Option<usize>) -> BoneData {
    BoneData {
        index,
        name: format!("b{index}"),
        parent,
        position: Vec2::new(5.0, 0.0),
        ..Default::default()
    }
}

fn ik(order: usize, bones: Vec<usize>, target: usize) -> IkConstraintData {
    IkConstraintData {
        name: "ik".into(),
        order,
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

fn transform(
    order: usize,
    source: usize,
    bones: Vec<usize>,
    local: bool,
) -> TransformConstraintData {
    TransformConstraintData {
        name: "tc".into(),
        order,
        bones,
        source,
        offsets: [0.0; 6],
        local_source: local,
        local_target: local,
        additive: false,
        clamp: false,
        properties: Vec::new(),
        mix_rotate: 1.0,
        mix_x: 1.0,
        mix_y: 1.0,
        mix_scale_x: 1.0,
        mix_scale_y: 1.0,
        mix_shear_y: 1.0,
    }
}

fn path(order: usize, slot: usize, bones: Vec<usize>) -> PathConstraintData {
    PathConstraintData {
        name: "path".into(),
        order,
        bones,
        slot,
        position_mode: Default::default(),
        spacing_mode: Default::default(),
        rotate_mode: Default::default(),
        offset_rotation: 0.0,
        position: 0.0,
        spacing: 0.0,
        mix_rotate: 1.0,
        mix_x: 1.0,
        mix_y: 1.0,
    }
}

fn physics(order: usize, bone: usize) -> PhysicsConstraintData {
    let mut pd = physics_skel(1.0).data().physics_constraints[0].clone();
    pd.order = order;
    pd.bone = bone;
    pd
}

/// Pose `data` for a few physics steps and render it, which must not panic.
fn pose_and_render(data: SkeletonData) -> Skeleton {
    let mut sk = Skeleton::new(Arc::new(data));
    for _ in 0..3 {
        step_frame(&mut sk);
        let _ = crate::render::render(&sk);
    }
    sk
}

fn constraint_entries(sk: &Skeleton) -> usize {
    sk.update_cache
        .iter()
        .filter(|u| !matches!(u, Updatable::Bone(_)))
        .count()
}

// A binary export stores bone parents as raw indices. A parent past the bone
// list used to index out of bounds while building the update cache.
#[test]
fn out_of_range_bone_parent_poses_as_a_root() {
    let data = SkeletonData {
        bones: vec![bone(0, None), bone(1, Some(u32::MAX as usize))],
        ..Default::default()
    };
    let sk = pose_and_render(data);
    let b = sk.bone(1).unwrap();
    assert_eq!(b.parent(), None);
    assert!(close(b.world_x(), 5.0) && close(b.world_y(), 0.0));
}

// Each constraint below names a bone or slot the rig does not have. Building
// the update cache used to index out of bounds on every one of them (the path
// slot lookup, `sort_bone`, and `sort_reset`). Such a constraint is now left
// out of the cache, so its solver never runs.
#[test]
fn constraints_naming_missing_bones_or_slots_are_skipped() {
    let slot = SlotData {
        index: 0,
        name: "s".into(),
        bone: 7,
        color: Color::WHITE,
        dark_color: None,
        attachment: None,
        blend: crate::data::BlendMode::Normal,
    };
    let variants: Vec<SkeletonData> = vec![
        SkeletonData {
            ik_constraints: vec![ik(0, vec![0], 9)],
            ..Default::default()
        },
        SkeletonData {
            ik_constraints: vec![ik(0, vec![9], 0)],
            ..Default::default()
        },
        SkeletonData {
            transform_constraints: vec![transform(0, 9, vec![0], false)],
            ..Default::default()
        },
        SkeletonData {
            transform_constraints: vec![transform(0, 0, vec![9], true)],
            ..Default::default()
        },
        SkeletonData {
            path_constraints: vec![path(0, 3, vec![0])],
            ..Default::default()
        },
        SkeletonData {
            slots: vec![slot.clone()],
            path_constraints: vec![path(0, 0, vec![0])],
            ..Default::default()
        },
        SkeletonData {
            slots: vec![SlotData { bone: 0, ..slot }],
            path_constraints: vec![path(0, 0, vec![9])],
            ..Default::default()
        },
        SkeletonData {
            physics_constraints: vec![physics(0, 9)],
            ..Default::default()
        },
    ];
    for mut data in variants {
        data.bones = vec![bone(0, None)];
        let sk = pose_and_render(data);
        assert_eq!(constraint_entries(&sk), 0);
        assert_eq!(sk.update_cache.len(), 1, "only the root bone is updated");
    }
}

// A slider whose bone is missing still sorts: it reads no bone and applies
// nothing. An unknown slider index reads as a zero-mix pose.
#[test]
fn slider_on_a_missing_bone_is_harmless() {
    let data = SkeletonData {
        bones: vec![bone(0, None)],
        sliders: vec![SliderData {
            bone: Some(9),
            property: Some(crate::constraint::slider::SliderProperty::X),
            animation_index: Some(0),
            ..Default::default()
        }],
        ..Default::default()
    };
    let sk = pose_and_render(data);
    assert_eq!(constraint_entries(&sk), 1);
    assert_eq!(sk.slider_pose(5).mix, 0.0);
}

// Bones 0 and 1 are each other's parent and bone 2 is its own parent. The
// recursive parent walk never ended (a stack overflow), and neither did the
// recursive descendant reset that the constraints trigger. Both walks now stop
// where the cycle closes.
#[test]
fn cyclic_bone_parents_terminate() {
    let data = SkeletonData {
        bones: vec![bone(0, Some(1)), bone(1, Some(0)), bone(2, Some(2))],
        ik_constraints: vec![ik(0, vec![2], 1)],
        physics_constraints: vec![physics(1, 0), physics(2, 2)],
        ..Default::default()
    };
    let sk = pose_and_render(data);
    assert_eq!(constraint_entries(&sk), 3);
    for i in 0..3 {
        assert!(
            sk.update_cache
                .iter()
                .any(|u| matches!(u, Updatable::Bone(b) if *b == i)),
            "bone {i} is updated"
        );
    }
}

// A 100,000-bone chain from a large export: the recursive parent walk and
// descendant reset used one stack frame per bone and overflowed the stack.
#[test]
fn deep_bone_chain_sorts_without_recursion() {
    let n = 100_000;
    let bones = (0..n).map(|i| bone(i, i.checked_sub(1))).collect();
    let data = SkeletonData {
        bones,
        // The IK sorts the whole chain from its leaf, then the physics on the
        // root resets every descendant.
        ik_constraints: vec![ik(0, vec![n - 1], 0)],
        physics_constraints: vec![physics(1, 0)],
        ..Default::default()
    };
    let mut sk = Skeleton::new(Arc::new(data));
    sk.update_world_transform();
    assert!(sk.bone(n - 1).unwrap().world_x().is_finite());
}

// Alternating constraints that re-sort a deep chain and then reset it make the
// cache grow with bones times constraints (about 1.1 million entries here, and
// gigabytes for a large hostile file). Past `MAX_CONSTRAINT_CACHE` entries no
// further constraint is sorted in.
#[test]
fn update_cache_growth_is_bounded() {
    let n = 1_100;
    let pairs = 1_000;
    let bones = (0..n).map(|i| bone(i, i.checked_sub(1))).collect();
    let data = SkeletonData {
        bones,
        ik_constraints: (0..pairs).map(|k| ik(2 * k, vec![n - 1], 0)).collect(),
        physics_constraints: (0..pairs).map(|k| physics(2 * k + 1, 0)).collect(),
        ..Default::default()
    };
    let sk = Skeleton::new(Arc::new(data));
    assert!(
        sk.update_cache.len() <= cache::MAX_CONSTRAINT_CACHE + 2 * n,
        "cache len {}",
        sk.update_cache.len()
    );
    assert!(constraint_entries(&sk) < 2 * pairs);
}
