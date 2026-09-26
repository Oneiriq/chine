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

/// Slot 0 of `rig(diagonal(), ..)` showing `path` under the name "path".
fn with_path(path: PathAttachment) -> SkeletonData {
    let mut data = rig(diagonal(), 6, false, 0);
    data.default_skin.set(0, "path", Attachment::Path(path));
    data
}

/// The follower's world position once `pc` is applied at full mix.
fn follower_at(data: SkeletonData, mut pc: PathConstraintData) -> (f32, f32) {
    pc.mix_rotate = 0.0;
    let data = SkeletonData {
        path_constraints: vec![pc],
        ..data
    };
    let mut sk = Skeleton::new(Arc::new(data));
    sk.update_world_transform();
    let f = sk.bone(1).unwrap();
    (f.world_x(), f.world_y())
}

fn close(actual: (f32, f32), expected: (f32, f32)) -> bool {
    (actual.0 - expected.0).abs() < 1e-3 && (actual.1 - expected.1).abs() < 1e-3
}

// A path without constant speed was sampled by arc length, and its exported
// curve lengths were ignored. Spine moves along each curve by its Bezier
// parameter there. This curve puts both handles on its start, so it is
// x = 30 * t^3, and halfway along the path is t = 0.5, x = 3.75.
#[test]
fn path_without_constant_speed_follows_its_exported_lengths() {
    let vertices = MeshVertices::Unweighted(vec![
        -10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 30.0, 0.0, 40.0, 0.0,
    ]);
    let path = |constant_speed| {
        PathAttachment::new(
            "path",
            vertices.clone(),
            6,
            vec![30.0, 0.0],
            false,
            constant_speed,
        )
    };
    let pc = follow(vec![1], RotateMode::Chain);
    let exported = follower_at(with_path(path(false)), pc.clone());
    assert!(close(exported, (3.75, 0.0)), "{exported:?}");
    // With constant speed the same path is measured, which puts the bone
    // near the middle of the line.
    let measured = follower_at(with_path(path(true)), pc);
    assert!((measured.0 - 15.0).abs() < 1.0, "{measured:?}");
}

// The last curve of a closed path runs from the last knot back to the first.
// The exported lengths put 15 of the path's 30 a quarter of the way along
// that curve, whose handles bend it up to y = 10.
#[test]
fn closed_path_without_constant_speed_wraps_its_last_curve() {
    let third = 10.0 / 3.0;
    let vertices = MeshVertices::Unweighted(vec![
        0.0,
        10.0,
        0.0,
        0.0,
        third,
        0.0,
        2.0 * third,
        0.0,
        10.0,
        0.0,
        10.0,
        10.0,
    ]);
    let path = PathAttachment::new("path", vertices, 6, vec![10.0, 30.0], true, false);
    let at = follower_at(with_path(path), follow(vec![1], RotateMode::Chain));
    assert!(close(at, (8.4375, 5.625)), "{at:?}");
}

// Exported lengths with fewer entries than the path has curves fall back to
// measuring the path.
#[test]
fn short_exported_lengths_fall_back_to_constant_speed() {
    let path = PathAttachment::new("path", diagonal(), 6, Vec::new(), false, false);
    let at = follower_at(with_path(path), follow(vec![1], RotateMode::Chain));
    assert!(
        (at.0 - 15.0).abs() < 1.0 && (at.1 - 15.0).abs() < 1.0,
        "{at:?}"
    );
}

// The constraint read the path from the slot's setup attachment in the
// default skin. Spine follows the path the slot currently shows, found
// through the active skin.
#[test]
fn constraint_follows_the_slots_current_path() {
    let vertical = MeshVertices::Unweighted(vec![
        0.0, -10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 20.0, 0.0, 30.0, 0.0, 40.0,
    ]);
    let mut data = rig(diagonal(), 6, false, 0);
    data.default_skin.set(
        0,
        "vertical",
        Attachment::Path(PathAttachment::new(
            "vertical",
            vertical.clone(),
            6,
            Vec::new(),
            false,
            true,
        )),
    );
    let mut skin = Skin::new("alt");
    skin.set(
        0,
        "path",
        Attachment::Path(PathAttachment::new(
            "path",
            vertical,
            6,
            Vec::new(),
            false,
            true,
        )),
    );
    data.skins.push(skin);
    let mut pc = follow(vec![1], RotateMode::Chain);
    pc.mix_rotate = 0.0;
    data.path_constraints.push(pc);
    let mut sk = Skeleton::new(Arc::new(data));
    let follower = |sk: &Skeleton| {
        let f = sk.bone(1).unwrap();
        (f.world_x(), f.world_y())
    };

    sk.update_world_transform();
    assert!(close(follower(&sk), (15.0, 15.0)), "{:?}", follower(&sk));

    // The slot now shows the vertical path.
    sk.slot_pose_and_setup(0).unwrap().0.attachment = Some("vertical".into());
    sk.update_world_transform();
    assert!(close(follower(&sk), (0.0, 15.0)), "{:?}", follower(&sk));

    // The active skin's "path" replaces the default skin's.
    sk.set_slots_to_setup_pose();
    sk.set_skin("alt");
    sk.update_world_transform();
    assert!(close(follower(&sk), (0.0, 15.0)), "{:?}", follower(&sk));

    // A slot that shows nothing leaves the bone at its own pose.
    sk.slot_pose_and_setup(0).unwrap().0.attachment = None;
    sk.set_bones_to_setup_pose();
    sk.update_world_transform();
    assert_eq!(follower(&sk), (0.0, 0.0));
}
