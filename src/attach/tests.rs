use super::*;
use crate::data::{BoneData, SkeletonData};
use std::sync::Arc;

const EPS: f32 = 1e-3;

fn close(p: Vec2, x: f32, y: f32) -> bool {
    (p.x - x).abs() < EPS && (p.y - y).abs() < EPS
}

fn one_bone_at(x: f32, y: f32) -> Skeleton {
    let data = SkeletonData {
        bones: vec![BoneData {
            index: 0,
            name: "root".into(),
            position: Vec2::new(x, y),
            ..Default::default()
        }],
        ..Default::default()
    };
    let mut sk = Skeleton::new(Arc::new(data));
    sk.update_world_transform();
    sk
}

#[test]
fn region_quad_corners_center_on_the_bone() {
    let sk = one_bone_at(100.0, 50.0);
    let region = AtlasRegion {
        name: "r".into(),
        page: 0,
        x: 0,
        y: 0,
        width: 20,
        height: 10,
        degrees: 0,
        offset_x: 0.0,
        offset_y: 0.0,
        original_width: 20,
        original_height: 10,
        index: -1,
    };
    let mut att = RegionAttachment::new("r", "r");
    att.width = 20.0;
    att.height = 10.0;
    att.update(&region, 64, 64);
    let v = att.compute_world_vertices(sk.bone(0).unwrap());
    // 20x10 quad centered on (100,50): BL(90,45) UL(90,55) UR(110,55) BR(110,45).
    assert!(close(v[0], 90.0, 45.0), "BL {:?}", v[0]);
    assert!(close(v[1], 90.0, 55.0), "UL {:?}", v[1]);
    assert!(close(v[2], 110.0, 55.0), "UR {:?}", v[2]);
    assert!(close(v[3], 110.0, 45.0), "BR {:?}", v[3]);
}

#[test]
fn unweighted_mesh_follows_its_bone() {
    let sk = one_bone_at(10.0, 20.0);
    let mesh = MeshAttachment::new(
        "m",
        "m",
        MeshVertices::Unweighted(vec![0.0, 0.0, 5.0, 0.0, 0.0, 5.0]),
        vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
        vec![0, 1, 2],
    );
    let w = mesh.compute_world_vertices(&sk, 0, &[]);
    // identity bone at (10,20): each local vertex offset by (10,20).
    assert!(close(w[0], 10.0, 20.0));
    assert!(close(w[1], 15.0, 20.0));
    assert!(close(w[2], 10.0, 25.0));
}

#[test]
fn weighted_mesh_blends_two_bones() {
    // Two roots at (0,0) and (100,0). A vertex weighted 50/50 lands at the
    // midpoint of where each bone places its local origin.
    let data = SkeletonData {
        bones: vec![
            BoneData {
                index: 0,
                name: "a".into(),
                ..Default::default()
            },
            BoneData {
                index: 1,
                name: "b".into(),
                position: Vec2::new(100.0, 0.0),
                ..Default::default()
            },
        ],
        ..Default::default()
    };
    let mut sk = Skeleton::new(Arc::new(data));
    sk.update_world_transform();
    let mesh = MeshAttachment::new(
        "m",
        "m",
        MeshVertices::Weighted {
            bones: vec![2, 0, 1], // 1 vertex, influenced by bones 0 and 1
            vertices: vec![0.0, 0.0, 0.5, 0.0, 0.0, 0.5], // (0,0)w.5 via bone0; (0,0)w.5 via bone1
        },
        vec![0.0, 0.0],
        vec![],
    );
    let w = mesh.compute_world_vertices(&sk, 0, &[]);
    assert_eq!(w.len(), 1);
    assert!(close(w[0], 50.0, 0.0), "midpoint {:?}", w[0]);
}

#[test]
fn path_control_points_follow_their_bone() {
    let sk = one_bone_at(10.0, 0.0);
    // 4 control points (one cubic Bezier), unweighted.
    let path = PathAttachment::new(
        "p",
        MeshVertices::Unweighted(vec![0.0, 0.0, 10.0, 0.0, 20.0, 0.0, 30.0, 0.0]),
        4,
        vec![30.0],
        false,
        true,
    );
    assert_eq!(path.vertex_count(), 4);
    let w = path.compute_world_vertices(&sk, 0);
    assert_eq!(w.len(), 4);
    // bone at (10,0): control points offset by +10 in x.
    assert!(close(w[0], 10.0, 0.0));
    assert!(close(w[3], 40.0, 0.0));
}

#[test]
fn rotated_region_remaps_mesh_uvs() {
    let region = AtlasRegion {
        name: "r".into(),
        page: 0,
        x: 10,
        y: 20,
        width: 30,
        height: 40,
        degrees: 90,
        offset_x: 0.0,
        offset_y: 0.0,
        original_width: 40,
        original_height: 30,
        index: -1,
    };
    let mut m = MeshAttachment::new(
        "m",
        "m",
        MeshVertices::Unweighted(vec![0.0, 0.0, 5.0, 5.0]),
        vec![0.0, 0.0, 1.0, 1.0], // mesh UVs (0,0) and (1,1)
        vec![],
    );
    m.remap_uvs(&region, 100, 100);
    // degrees 90: u = (rx + (1 - mv) * rw) / pw, v = (ry + mu * rh) / ph.
    // (0,0) -> u=(10+30)/100=0.4, v=(20+0)/100=0.2.
    assert!((m.uvs[0] - 0.4).abs() < 1e-4, "u0={}", m.uvs[0]);
    assert!((m.uvs[1] - 0.2).abs() < 1e-4, "v0={}", m.uvs[1]);
    // (1,1) -> u=(10+0)/100=0.1, v=(20+40)/100=0.6.
    assert!((m.uvs[2] - 0.1).abs() < 1e-4, "u1={}", m.uvs[2]);
    assert!((m.uvs[3] - 0.6).abs() < 1e-4, "v1={}", m.uvs[3]);
}

#[test]
fn bounding_box_polygon_follows_its_bone() {
    let sk = one_bone_at(10.0, 0.0);
    let bb = BoundingBoxAttachment::new(
        "bb",
        MeshVertices::Unweighted(vec![0.0, 0.0, 10.0, 0.0, 10.0, 10.0]),
        3,
    );
    assert_eq!(bb.vertex_count(), 3);
    let w = bb.compute_world_vertices(&sk, 0);
    assert_eq!(w.len(), 3);
    // bone at (10,0): each polygon vertex offset by +10 in x.
    assert!(close(w[0], 10.0, 0.0), "{:?}", w[0]);
    assert!(close(w[1], 20.0, 0.0), "{:?}", w[1]);
    assert!(close(w[2], 20.0, 10.0), "{:?}", w[2]);
}

#[test]
fn point_attachment_transforms_with_its_bone() {
    let sk = one_bone_at(10.0, 20.0);
    let bone = sk.bone(0).unwrap();
    let p = PointAttachment::new("p", 5.0, 0.0, 90.0);
    let pos = p.compute_world_position(bone);
    assert!(close(pos, 15.0, 20.0), "pos {:?}", pos);
    // identity bone: world rotation equals the point's local rotation.
    assert!((p.compute_world_rotation(bone) - 90.0).abs() < 1e-3);
}

#[test]
fn linked_mesh_borrows_parent_geometry() {
    let mut parent = MeshAttachment::new(
        "wing",
        "wing",
        MeshVertices::Unweighted(vec![0.0, 0.0, 10.0, 0.0, 0.0, 10.0]),
        vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
        vec![0, 1, 2],
    );
    parent.hull_length = 3;
    let link =
        LinkedMeshAttachment::new("wing-blue", "wing-blue", None, "wing", Color::WHITE, true);
    let m = link.resolve(&parent, &AttachmentKey::default());
    // Identity stays the link's own. Geometry is borrowed from the parent.
    assert_eq!(m.name, "wing-blue");
    assert_eq!(m.path, "wing-blue");
    assert_eq!(m.triangles, parent.triangles);
    assert_eq!(m.hull_length, 3);
    assert_eq!(m.vertex_count(), 3);
}
