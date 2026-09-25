//! Malformed rigs and atlases through binding and rendering: missing pages,
//! hostile sequence fields, meshes whose parts disagree, and dangling slot
//! references.

use std::sync::Arc;

use super::*;
use crate::atlas::AtlasPage;
use crate::attach::{ClippingAttachment, MeshVertices};
use crate::data::BoneData;

fn page(name: &str) -> AtlasPage {
    let mut atlas = Atlas::parse(&format!("{name}\nsize: 64, 32\n"));
    atlas.pages.remove(0)
}

fn region(name: &str, page: usize, x: u32) -> AtlasRegion {
    AtlasRegion {
        name: name.into(),
        page,
        x,
        y: 2,
        width: 8,
        height: 4,
        degrees: if x.is_multiple_of(3) { 90 } else { 0 },
        offset_x: 1.0,
        offset_y: 0.5,
        original_width: 10,
        original_height: 6,
        index: -1,
    }
}

fn slot(index: usize, attachment: &str) -> SlotData {
    SlotData {
        index,
        name: format!("s{index}"),
        bone: 0,
        color: Color::WHITE,
        dark_color: None,
        attachment: Some(attachment.into()),
        blend: BlendMode::Normal,
    }
}

fn skeleton(slots: Vec<SlotData>, skin: Skin) -> Skeleton {
    let data = SkeletonData {
        bones: vec![BoneData {
            index: 0,
            name: "root".into(),
            ..Default::default()
        }],
        slots,
        default_skin: skin,
        ..Default::default()
    };
    let mut sk = Skeleton::new(Arc::new(data));
    sk.update_world_transform();
    sk
}

fn tri_mesh(uvs: Vec<f32>, triangles: Vec<u16>) -> Attachment {
    Attachment::Mesh(MeshAttachment::new(
        "m",
        "m",
        MeshVertices::Unweighted(vec![1.0, 1.0, 30.0, 1.0, 1.0, 30.0]),
        uvs,
        triangles,
    ))
}

fn sequenced_region(count: usize, start: usize, digits: usize, setup: usize) -> RegionAttachment {
    let mut r = RegionAttachment::new("s", "s");
    r.width = 8.0;
    r.height = 8.0;
    r.sequence = Some(Sequence::new(count, start, digits, setup));
    r
}

fn sequenced_mesh(count: usize, start: usize, digits: usize, setup: usize) -> MeshAttachment {
    let mut m = MeshAttachment::new(
        "t",
        "s",
        MeshVertices::Unweighted(vec![0.0, 0.0, 5.0, 0.0, 0.0, 5.0]),
        vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
        vec![0, 1, 2],
    );
    m.sequence = Some(Sequence::new(count, start, digits, setup));
    m
}

// `Atlas` has public fields, so a host can hand over a region whose page index
// is past the page list. Binding used to index the page list with it.
#[test]
fn region_on_a_missing_page_is_left_unbound() {
    let atlas = Atlas {
        pages: vec![],
        regions: vec![region("r", 5, 1), region("s0", 5, 1), region("s1", 5, 1)],
    };
    let mut skin = Skin::new("default");
    skin.set(0, "r", Attachment::Region(RegionAttachment::new("r", "r")));
    skin.set(1, "s", Attachment::Region(sequenced_region(2, 0, 0, 0)));
    skin.set(2, "t", Attachment::Mesh(sequenced_mesh(2, 0, 0, 1)));
    skin.set(
        3,
        "m",
        tri_mesh(vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0], vec![0, 1, 2]),
    );
    let mut data = SkeletonData {
        default_skin: skin,
        ..Default::default()
    };
    bind_atlas(&mut data, &atlas);
    for (_, _, att) in data.default_skin.iter() {
        match att {
            Attachment::Region(r) => assert_eq!(r.page, 0),
            Attachment::Mesh(m) => assert_eq!(m.page, 0),
            _ => {}
        }
    }
}

// A sequence count of `usize::MAX` used to loop that many times, pushing a
// frame each time. Binding now scans the atlas once and stores the frames as
// runs, so it is quick and small whatever the count.
#[test]
fn huge_sequence_count_binds_in_bounded_work() {
    let atlas = Atlas {
        pages: vec![page("p")],
        regions: vec![region("s0", 0, 1), region("s1", 0, 2)],
    };
    let mut r = sequenced_region(usize::MAX, 0, 1, usize::MAX);
    Binder::new(&atlas).bind_region_sequence(&mut r);
    let seq = r.sequence.as_ref().unwrap();
    assert_eq!(seq.frame(0).unwrap().1, 0);
    assert_ne!(seq.frame(0).unwrap().0, seq.frame(1).unwrap().0);
    assert_eq!(seq.frame(1).unwrap().1, 0);
    // Every later frame falls back to the last bound UVs.
    assert_eq!(seq.frame(2).unwrap().0, seq.frame(1).unwrap().0);
    assert_eq!(seq.frame(usize::MAX).unwrap().0, seq.frame(1).unwrap().0);

    let mut m = sequenced_mesh(usize::MAX, usize::MAX, 0, 0);
    Binder::new(&atlas).bind_mesh_sequence(&mut m);
    assert!(m.sequence.as_ref().unwrap().frame(7).is_some());
}

// The pose fuzz OOM: `digits` near `u32::MAX` padded each frame name with that
// many zeros. `usize::MAX` used to panic with a capacity overflow.
#[test]
fn huge_sequence_digits_bind_without_a_huge_name() {
    let atlas = Atlas {
        pages: vec![page("p")],
        regions: vec![region("s0", 0, 1)],
    };
    for digits in [4_294_967_285, usize::MAX] {
        let mut r = sequenced_region(3, 0, digits, 0);
        Binder::new(&atlas).bind_region_sequence(&mut r);
        let mut m = sequenced_mesh(3, 0, digits, 0);
        Binder::new(&atlas).bind_mesh_sequence(&mut m);
        assert!(m.sequence.unwrap().frame(0).is_some());
    }
}

// Many sequences over a large atlas, or long sequences of large meshes, would
// make binding slow and large. Binding pays from a budget and stops storing
// frames once it runs out: later frames read the last stored one.
#[test]
fn sequence_binding_stops_when_the_budget_runs_out() {
    let atlas = Atlas {
        pages: vec![page("p")],
        regions: vec![region("s0", 0, 1), region("s1", 0, 2), region("s2", 0, 4)],
    };
    let mut binder = Binder::new(&atlas);
    // Matching frames costs one atlas scan, and each frame of a 3-vertex mesh
    // costs 6.
    binder.budget = binder.scan_cost + 2 * 6;
    let mut m = sequenced_mesh(3, 0, 0, 0);
    binder.bind_mesh_sequence(&mut m);
    assert_eq!(binder.budget, 0);
    let seq = m.sequence.unwrap();
    assert_ne!(seq.frame(0).unwrap().0, seq.frame(1).unwrap().0);
    assert_eq!(seq.frame(2).unwrap().0, seq.frame(1).unwrap().0);

    let mut r = sequenced_region(3, 0, 0, 0);
    binder.bind_region_sequence(&mut r);
    assert!(r.sequence.unwrap().frame(0).is_none());
}

/// The region named `name` and its page, by a linear search of the atlas.
fn find_region<'a>(atlas: &'a Atlas, name: &str) -> Option<(&'a AtlasRegion, &'a AtlasPage)> {
    let region = atlas.find_region(name)?;
    Some((region, atlas.pages.get(region.page)?))
}

/// The dense frame list and final attachment state that the original
/// frame-by-frame binding produced for a region sequence.
fn reference_region_bind(
    r: &mut RegionAttachment,
    atlas: &Atlas,
    seq: &Sequence,
) -> Vec<(Vec<f32>, usize)> {
    let mut frames = Vec::new();
    for i in 0..seq.count {
        if let Some((region, page)) = find_region(atlas, &seq.region_name(&r.path, i)) {
            r.update(region, page.width, page.height);
            frames.push((r.uvs.to_vec(), region.page));
        } else {
            frames.push((r.uvs.to_vec(), r.page));
        }
    }
    if let Some((region, page)) = find_region(atlas, &seq.region_name(&r.path, seq.setup_index)) {
        r.page = region.page;
        r.update(region, page.width, page.height);
    }
    frames
}

/// The dense frame list and final attachment state that the original
/// frame-by-frame binding produced for a mesh sequence.
fn reference_mesh_bind(
    m: &mut MeshAttachment,
    atlas: &Atlas,
    seq: &Sequence,
) -> Vec<(Vec<f32>, usize)> {
    let mut frames = Vec::new();
    let original = m.uvs.clone();
    for i in 0..seq.count {
        if let Some((region, page)) = find_region(atlas, &seq.region_name(&m.path, i)) {
            m.uvs.clone_from(&original);
            m.remap_uvs(region, page.width, page.height);
            frames.push((m.uvs.clone(), region.page));
        } else {
            frames.push((original.clone(), m.page));
        }
    }
    m.uvs.clone_from(&original);
    if let Some((region, page)) = find_region(atlas, &seq.region_name(&m.path, seq.setup_index)) {
        m.page = region.page;
        m.remap_uvs(region, page.width, page.height);
    }
    frames
}

/// Every frame the bound sequence serves, for indices `0..count + 3`.
fn served(seq: &Sequence) -> Vec<(Vec<f32>, usize)> {
    (0..seq.count + 3)
        .filter_map(|i| seq.frame(i).map(|(uvs, page)| (uvs.to_vec(), page)))
        .collect()
}

// The run-based binding must serve exactly the frames the dense binding
// stored, and leave the attachment in the same state. Atlases mix gaps,
// duplicate names, names with the wrong padding, other bases, a second page,
// and the attachment's own page set to something else first.
#[test]
fn sequence_binding_matches_the_dense_reference() {
    let names = [
        "s0", "s1", "s01", "s3", "s03", "s004", "s7", "s9", "s10", "s11", "s12", "s1", "t2", "s",
        "sx", "s00", "s5",
    ];
    let mut seed = 12345_u32;
    let mut next = move || {
        seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
        seed >> 8
    };
    for _ in 0..300 {
        let mut regions = Vec::new();
        for name in names {
            if next().is_multiple_of(2) {
                let x = next() % 40;
                regions.push(region(name, (next() % 3) as usize, x));
            }
        }
        let atlas = Atlas {
            pages: vec![page("a"), page("b")],
            regions,
        };
        let count = (next() % 14) as usize;
        let start = (next() % 3) as usize;
        let digits = (next() % 4) as usize;
        let setup = (next() % 16) as usize;
        let seq = Sequence::new(count, start, digits, setup);

        let mut expected_r = sequenced_region(count, start, digits, setup);
        expected_r.page = 7;
        let mut actual_r = expected_r.clone();
        let dense = reference_region_bind(&mut expected_r, &atlas, &seq);
        Binder::new(&atlas).bind_region_sequence(&mut actual_r);
        assert_eq!(
            served(actual_r.sequence.as_ref().unwrap()),
            with_tail(&dense)
        );
        assert_eq!(actual_r.uvs, expected_r.uvs);
        assert_eq!(actual_r.page, expected_r.page);
        let bone = one_bone();
        assert_eq!(
            actual_r.compute_world_vertices(bone.bone(0).unwrap()),
            expected_r.compute_world_vertices(bone.bone(0).unwrap())
        );

        let mut expected_m = sequenced_mesh(count, start, digits, setup);
        expected_m.page = 7;
        let mut actual_m = expected_m.clone();
        let dense = reference_mesh_bind(&mut expected_m, &atlas, &seq);
        Binder::new(&atlas).bind_mesh_sequence(&mut actual_m);
        assert_eq!(
            served(actual_m.sequence.as_ref().unwrap()),
            with_tail(&dense)
        );
        assert_eq!(actual_m.uvs, expected_m.uvs);
        assert_eq!(actual_m.page, expected_m.page);
    }
}

/// `dense` plus the three past-the-end reads, which repeat the last frame.
fn with_tail(dense: &[(Vec<f32>, usize)]) -> Vec<(Vec<f32>, usize)> {
    let mut out = dense.to_vec();
    if let Some(last) = dense.last() {
        out.extend(std::iter::repeat_n(last.clone(), 3));
    }
    out
}

fn one_bone() -> Skeleton {
    skeleton(vec![], Skin::new("default"))
}

// A mesh whose triangles index past its vertices (or whose UVs are missing, as
// in the JSON fuzz inputs) used to reach the clipper, which indexed past the
// command's positions. Its commands now keep only complete triangles.
#[test]
fn mesh_triangles_past_the_vertices_are_dropped() {
    let clip = Attachment::Clipping(ClippingAttachment::new(
        "clip",
        "s1",
        MeshVertices::Unweighted(vec![0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0]),
        4,
    ));
    let cases = [
        (
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            vec![0, 1, 2, 0, 1, 5, 7],
        ),
        (vec![0.0, 0.0, 1.0, 0.0, 0.0], vec![0, 1, 2]),
        (vec![], vec![0, 1, 2]),
    ];
    for (uvs, triangles) in cases {
        let mut skin = Skin::new("default");
        skin.set(0, "m", tri_mesh(uvs.clone(), triangles.clone()));
        let plain = render(&skeleton(vec![slot(0, "m")], skin));
        let c = &plain[0];
        assert_eq!(c.uvs.len(), c.positions.len() * 2);
        assert!(c
            .triangles
            .iter()
            .all(|&t| usize::from(t) < c.positions.len()));

        let mut skin = Skin::new("default");
        skin.set(0, "clip", clip.clone());
        skin.set(1, "m", tri_mesh(uvs, triangles));
        let clipped = render(&skeleton(vec![slot(0, "clip"), slot(1, "m")], skin));
        for c in &clipped {
            assert!(c
                .triangles
                .iter()
                .all(|&t| usize::from(t) < c.positions.len()));
        }
    }
}

// The draw order and skins can name slots the rig does not have: a draw-order
// entry past the slot list, and a skin entry for slot 9. Both are ignored.
#[test]
fn dangling_slot_references_draw_nothing() {
    let mut skin = Skin::new("default");
    skin.set(
        0,
        "m",
        tri_mesh(vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0], vec![0, 1, 2]),
    );
    skin.set(
        9,
        "m",
        tri_mesh(vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0], vec![0, 1, 2]),
    );
    let mut sk = skeleton(vec![slot(0, "m")], skin);
    sk.set_draw_order(&[usize::MAX, 9, 0, 0]);
    assert_eq!(render(&sk).len(), 2);
}
