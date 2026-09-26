//! Malformed attachment data: vertex arrays that disagree with their counts,
//! atlas values near the integer limits, and hostile sequence fields.

use std::sync::Arc;

use super::*;
use crate::data::{BoneData, SkeletonData};

fn one_bone() -> Skeleton {
    let data = SkeletonData {
        bones: vec![BoneData {
            index: 0,
            name: "root".into(),
            position: Vec2::new(10.0, 0.0),
            ..Default::default()
        }],
        ..Default::default()
    };
    let mut sk = Skeleton::new(Arc::new(data));
    sk.update_world_transform();
    sk
}

fn weighted_mesh(bones: Vec<usize>, vertices: Vec<f32>, vertex_count: usize) -> MeshAttachment {
    MeshAttachment::new(
        "m",
        "m",
        MeshVertices::Weighted { bones, vertices },
        vec![0.0; vertex_count * 2],
        vec![0, 1, 2],
    )
}

// A weighted mesh whose UVs describe three vertices while its bone list is
// empty, too short for its influence counts, or its bind values too short for
// its influences (the JSON fuzz crashes). Each read past the end of an array.
// Only the fully described vertices are emitted now.
#[test]
fn weighted_mesh_shorter_than_its_uvs_emits_the_described_vertices() {
    let sk = one_bone();
    let cases = [
        (vec![], vec![], 0),
        (vec![1, 0, 2, 0], vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0], 1),
        (vec![1, 0, 1, 0], vec![0.0, 0.0, 1.0, 0.0, 0.0], 1),
        (vec![1, 0, 1, 0, 1, 0], vec![0.0, 0.0, 1.0], 1),
    ];
    for (bones, vertices, expected) in cases {
        let mesh = weighted_mesh(bones, vertices, 3);
        let world = mesh.compute_world_vertices(&sk, 0, &[]);
        assert_eq!(world.len(), expected);
        for p in world {
            assert!((p.x - 10.0).abs() < 1e-4 && p.y.abs() < 1e-4, "{p:?}");
        }
    }
}

// An influence count of `usize::MAX` used to overflow the index arithmetic.
// The vertex is not described, so nothing is emitted.
#[test]
fn weighted_mesh_with_a_huge_influence_count_emits_nothing() {
    let sk = one_bone();
    let mesh = weighted_mesh(vec![usize::MAX, 0], vec![0.0, 0.0, 1.0], 3);
    assert!(mesh.compute_world_vertices(&sk, 0, &[]).is_empty());
}

// An unweighted mesh with one vertex position but three UV pairs used to read
// past its vertex array.
#[test]
fn unweighted_mesh_shorter_than_its_uvs_emits_the_described_vertices() {
    let sk = one_bone();
    let mesh = MeshAttachment::new(
        "m",
        "m",
        MeshVertices::Unweighted(vec![1.0, 2.0, 3.0]),
        vec![0.0; 6],
        vec![0, 1, 2],
    );
    let world = mesh.compute_world_vertices(&sk, 0, &[]);
    assert_eq!(world, vec![Vec2::new(11.0, 2.0)]);
}

// Path, bounding-box, and clipping attachments store a vertex count read from
// the file. `usize::MAX` used to reach `Vec::with_capacity` and panic with a
// capacity overflow. The count now only caps what the vertex data holds.
#[test]
fn huge_declared_vertex_count_is_capped_by_the_data() {
    let sk = one_bone();
    let v = || MeshVertices::Unweighted(vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0]);
    let path = PathAttachment::new("p", v(), usize::MAX, vec![], false, false);
    assert_eq!(path.compute_world_vertices(&sk, 0).len(), 3);
    let bb = BoundingBoxAttachment::new("bb", v(), usize::MAX);
    assert_eq!(bb.compute_world_vertices(&sk, 0).len(), 3);
    let clip = ClippingAttachment::new("c", "s", v(), usize::MAX);
    assert_eq!(clip.compute_world_vertices(&sk, 0).len(), 3);
    let weighted = ClippingAttachment::new(
        "c",
        "s",
        MeshVertices::Weighted {
            bones: vec![1, 0],
            vertices: vec![0.0, 0.0, 1.0],
        },
        usize::MAX,
    );
    assert_eq!(weighted.compute_world_vertices(&sk, 0).len(), 1);
}

// An atlas region at `y: 4294967295` (the atlas fuzz crash) overflowed
// `y + height` while computing its UVs.
#[test]
fn region_edges_near_u32_max_do_not_overflow() {
    let region = AtlasRegion {
        name: "r".into(),
        page: 0,
        x: u32::MAX,
        y: u32::MAX,
        width: u32::MAX,
        height: 10,
        degrees: 90,
        offset_x: 0.0,
        offset_y: 0.0,
        original_width: u32::MAX,
        original_height: 0,
        index: -1,
    };
    let mut r = RegionAttachment::new("r", "r");
    r.width = 10.0;
    r.height = 10.0;
    r.update(&region, 0, 0);
    assert!(r.uvs.iter().all(|u| u.is_finite()));
}

// `start + index` overflowed for a large `start`, and a hostile `digits`
// padded the name with that many zeros (a 4 GB allocation in the pose fuzz
// OOM, a capacity overflow panic for `usize::MAX`).
#[test]
fn sequence_region_name_survives_hostile_fields() {
    let seq = Sequence::new(1, usize::MAX, 0, 0);
    let name = seq.region_name("a", usize::MAX);
    assert_eq!(name, format!("a{}", u128::from(u64::MAX) * 2));
    let seq = Sequence::new(1, 0, usize::MAX, 0);
    let name = seq.region_name("a", 7);
    assert_eq!(name.len(), 1 + MAX_SEQUENCE_DIGITS);
    assert!(name.ends_with("007"));
}

#[test]
fn frame_index_inverts_region_name() {
    for start in [0, 1, 7, 99, usize::MAX - 3] {
        for digits in [0, 1, 2, 4, MAX_SEQUENCE_DIGITS + 5] {
            let seq = Sequence::new(10, start, digits, 0);
            for index in [0, 1, 2, 9, 10, 1234] {
                let name = seq.region_name("base1", index);
                assert_eq!(
                    seq.frame_index("base1", &name),
                    Some(index),
                    "{name} start={start} digits={digits}"
                );
            }
        }
    }
    let seq = Sequence::new(10, 1, 2, 0);
    assert_eq!(seq.frame_index("f", "f01"), Some(0));
    assert_eq!(seq.frame_index("f", "f10"), Some(9));
    assert_eq!(seq.frame_index("f", "f123"), Some(122));
    // Padding must match the digit count exactly.
    assert_eq!(seq.frame_index("f", "f1"), None);
    assert_eq!(seq.frame_index("f", "f001"), None);
    assert_eq!(seq.frame_index("f", "f0123"), None);
    // Below `start`, not digits, or another base.
    assert_eq!(seq.frame_index("f", "f00"), None);
    assert_eq!(seq.frame_index("f", "f0x"), None);
    assert_eq!(seq.frame_index("f", "f"), None);
    assert_eq!(seq.frame_index("f", "g01"), None);
    // A zero frame number is written "0" before padding.
    let zero = Sequence::new(1, 0, 0, 0);
    assert_eq!(zero.frame_index("f", "f0"), Some(0));
    assert_eq!(zero.frame_index("f", "f00"), None);
    // Too many digits for any frame number.
    assert_eq!(zero.frame_index("f", &format!("f{}", "9".repeat(60))), None);
}

#[test]
fn sequence_frame_runs_cover_their_ranges() {
    let mut seq = Sequence::new(10, 0, 0, 0);
    assert!(seq.frame(0).is_none());
    seq.push_frame(0, vec![0.0], 1);
    seq.push_frame(3, vec![3.0], 2);
    seq.push_frame(4, vec![4.0], 3);
    let page = |i| seq.frame(i).map(|(_, page)| page);
    assert_eq!(
        (0..6).map(page).collect::<Vec<_>>(),
        [1, 1, 1, 2, 3, 3].map(Some)
    );
    assert_eq!(page(usize::MAX), Some(3));
}
