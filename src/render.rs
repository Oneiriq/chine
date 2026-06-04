//! Render command generation: turn a posed skeleton into renderer-agnostic draw
//! data.
//!
//! [`render`] walks a skeleton's slots in draw order and emits a
//! [`RenderCommand`] per visible attachment: world-space positions, page UVs,
//! triangles, tint, blend mode, and atlas page. The host uploads these to the
//! GPU; chine does no rendering itself. Call [`bind_atlas`] once after loading
//! so attachment UVs and page indices are resolved against the atlas.

use glam::Vec2;

use crate::atlas::Atlas;
use crate::attach::{Attachment, MeshAttachment, RegionAttachment, Sequence};
use crate::data::{BlendMode, Color, SkeletonData};
use crate::skel::Skeleton;
use crate::skin::Skin;

/// One batch of geometry: an attachment's triangles with its texture and tint.
#[derive(Debug, Clone)]
pub struct RenderCommand {
    /// World-space vertex positions.
    pub positions: Vec<Vec2>,
    /// Page-normalized UVs, two per vertex (`u, v`).
    pub uvs: Vec<f32>,
    /// Triangle indices into `positions`.
    pub triangles: Vec<u16>,
    /// Tint color (slot color times attachment color).
    pub color: Color,
    /// Dark tint for two-color (tint-black) rendering, if the slot uses it.
    pub dark_color: Option<Color>,
    /// Atlas page index (which texture to bind).
    pub page: usize,
    /// Blend mode for the attachment's slot.
    pub blend: BlendMode,
}

/// Bind every region / mesh attachment to its atlas region: compute region
/// corner offsets and UVs, remap mesh UVs into page space, and record the page
/// index. Call once after loading and before [`render`].
pub fn bind_atlas(data: &mut SkeletonData, atlas: &Atlas) {
    bind_skin(&mut data.default_skin, atlas);
    for skin in &mut data.skins {
        bind_skin(skin, atlas);
    }
}

fn bind_skin(skin: &mut Skin, atlas: &Atlas) {
    for att in skin.attachments_mut() {
        match att {
            Attachment::Region(r) => {
                if r.sequence.is_some() {
                    bind_region_sequence(r, atlas);
                } else if let Some(region) = atlas.find_region(&r.path) {
                    let page = &atlas.pages[region.page];
                    let (pw, ph) = (page.width, page.height);
                    r.page = region.page;
                    r.update(region, pw, ph);
                }
            }
            Attachment::Mesh(m) => {
                if m.sequence.is_some() {
                    bind_mesh_sequence(m, atlas);
                } else if let Some(region) = atlas.find_region(&m.path) {
                    let page = &atlas.pages[region.page];
                    let (pw, ph) = (page.width, page.height);
                    m.page = region.page;
                    m.remap_uvs(region, pw, ph);
                }
            }
            Attachment::Path(_)
            | Attachment::BoundingBox(_)
            | Attachment::Point(_)
            | Attachment::LinkedMesh(_)
            | Attachment::Clipping(_) => {}
        }
    }
}

/// Resolve a region attachment's sequence: bind each frame's UVs and page, then
/// bake the static corner offsets / UVs / page to the setup frame.
fn bind_region_sequence(r: &mut RegionAttachment, atlas: &Atlas) {
    let Some(mut seq) = r.sequence.take() else {
        return;
    };
    for i in 0..seq.count {
        if let Some(region) = atlas.find_region(&seq.region_name(&r.path, i)) {
            let page = &atlas.pages[region.page];
            r.update(region, page.width, page.height);
            seq.push_frame(r.uvs.to_vec(), region.page);
        } else {
            seq.push_frame(r.uvs.to_vec(), r.page);
        }
    }
    if let Some(region) = atlas.find_region(&seq.region_name(&r.path, seq.setup_index)) {
        let page = &atlas.pages[region.page];
        r.page = region.page;
        r.update(region, page.width, page.height);
    }
    r.sequence = Some(seq);
}

/// Resolve a mesh attachment's sequence: remap the original UVs into each
/// frame's region, then leave the static UVs at the setup frame.
fn bind_mesh_sequence(m: &mut MeshAttachment, atlas: &Atlas) {
    let Some(mut seq) = m.sequence.take() else {
        return;
    };
    let original = m.uvs.clone();
    for i in 0..seq.count {
        if let Some(region) = atlas.find_region(&seq.region_name(&m.path, i)) {
            let page = &atlas.pages[region.page];
            m.uvs.clone_from(&original);
            m.remap_uvs(region, page.width, page.height);
            seq.push_frame(m.uvs.clone(), region.page);
        } else {
            seq.push_frame(original.clone(), m.page);
        }
    }
    m.uvs.clone_from(&original);
    if let Some(region) = atlas.find_region(&seq.region_name(&m.path, seq.setup_index)) {
        let page = &atlas.pages[region.page];
        m.page = region.page;
        m.remap_uvs(region, page.width, page.height);
    }
    m.sequence = Some(seq);
}

/// The bound UVs and page of a sequenced attachment at the slot's frame index
/// (`-1` uses the attachment's setup index); `None` for a non-sequenced
/// attachment (so the caller falls back to the static UVs).
fn sequence_frame(seq: Option<&Sequence>, slot_index: i32) -> Option<(&[f32], usize)> {
    let seq = seq?;
    let index = if slot_index < 0 {
        seq.setup_index
    } else {
        slot_index as usize
    };
    seq.frame(index)
}

/// Produce the draw-order render command stream for a posed skeleton.
///
/// Call [`Skeleton::update_world_transform`] first (to pose the bones) and
/// [`bind_atlas`] once at load time (to resolve UVs). Walks the skeleton's
/// runtime draw order, reading each slot's current attachment and tint (driven
/// by slot timelines); the blend mode comes from the setup data.
#[must_use]
pub fn render(skeleton: &Skeleton) -> Vec<RenderCommand> {
    let mut out = Vec::new();
    let mut clip: Option<Clip> = None;
    for &slot_index in skeleton.draw_order() {
        if let Some(cmd) = build_command(skeleton, slot_index, &mut clip) {
            match &clip {
                Some(c) => out.extend(clip_command(cmd, &c.polygons)),
                None => out.push(cmd),
            }
        }
        if let Some(c) = &clip {
            if slot_index == c.end_slot {
                clip = None;
            }
        }
    }
    out
}

/// Build the render command for one slot, or `None` if it has no renderable
/// attachment. A clipping attachment instead starts masking via `clip`.
fn build_command(
    skeleton: &Skeleton,
    slot_index: usize,
    clip: &mut Option<Clip>,
) -> Option<RenderCommand> {
    let data = skeleton.data();
    let setup = data.slots.get(slot_index)?;
    let slot = skeleton.slot(slot_index)?;
    let name = slot.attachment.as_deref()?;
    let att = data.attachment(slot_index, name, skeleton.active_skin())?;
    match att {
        Attachment::Region(r) => {
            let bone = skeleton.bone(setup.bone)?;
            let (uvs, page) = sequence_frame(r.sequence.as_ref(), slot.sequence_index)
                .map_or_else(|| (r.uvs.to_vec(), r.page), |(u, p)| (u.to_vec(), p));
            Some(RenderCommand {
                positions: r.compute_world_vertices(bone).to_vec(),
                uvs,
                triangles: vec![0, 1, 2, 2, 3, 0],
                color: mul(slot.color, r.color),
                dark_color: slot.dark_color,
                page,
                blend: setup.blend,
            })
        }
        Attachment::Mesh(m) => {
            let (uvs, page) = sequence_frame(m.sequence.as_ref(), slot.sequence_index)
                .map_or_else(|| (m.uvs.clone(), m.page), |(u, p)| (u.to_vec(), p));
            Some(RenderCommand {
                positions: m.compute_world_vertices(skeleton, setup.bone, &slot.deform),
                uvs,
                triangles: m.triangles.clone(),
                color: mul(slot.color, m.color),
                dark_color: slot.dark_color,
                page,
                blend: setup.blend,
            })
        }
        Attachment::Clipping(c) => {
            let pieces = clip_pieces(c.compute_world_vertices(skeleton, setup.bone));
            if !pieces.is_empty() {
                let end = data.find_slot(&c.end_slot).unwrap_or(usize::MAX);
                *clip = Some(Clip {
                    polygons: pieces,
                    end_slot: end,
                });
            }
            None
        }
        Attachment::Path(_)
        | Attachment::BoundingBox(_)
        | Attachment::Point(_)
        | Attachment::LinkedMesh(_) => None,
    }
}

/// An active clip region: one or more convex (CCW) polygons in world space (a
/// concave clip is decomposed into several) and the draw-order slot index after
/// which clipping stops.
struct Clip {
    polygons: Vec<Vec<Vec2>>,
    end_slot: usize,
}

/// Clip a command's triangles against the convex `polygons` (a decomposed clip
/// region), interpolating UVs at the new edges and unioning the pieces. Returns
/// `None` if nothing survives.
fn clip_command(cmd: RenderCommand, polygons: &[Vec<Vec2>]) -> Option<RenderCommand> {
    let mut positions = Vec::new();
    let mut uvs = Vec::new();
    let mut triangles = Vec::new();
    for tri in cmd.triangles.chunks_exact(3) {
        let idx = [tri[0] as usize, tri[1] as usize, tri[2] as usize];
        let p = [
            cmd.positions[idx[0]],
            cmd.positions[idx[1]],
            cmd.positions[idx[2]],
        ];
        let uv = [
            (cmd.uvs[idx[0] * 2], cmd.uvs[idx[0] * 2 + 1]),
            (cmd.uvs[idx[1] * 2], cmd.uvs[idx[1] * 2 + 1]),
            (cmd.uvs[idx[2] * 2], cmd.uvs[idx[2] * 2 + 1]),
        ];
        for polygon in polygons {
            let clipped = clip_triangle(&p, &uv, polygon);
            if clipped.len() < 3 {
                continue;
            }
            let base = positions.len() as u16;
            for (pt, (u, v)) in &clipped {
                positions.push(*pt);
                uvs.push(*u);
                uvs.push(*v);
            }
            for k in 1..clipped.len() as u16 - 1 {
                triangles.extend_from_slice(&[base, base + k, base + k + 1]);
            }
        }
    }
    if triangles.is_empty() {
        return None;
    }
    Some(RenderCommand {
        positions,
        uvs,
        triangles,
        color: cmd.color,
        dark_color: cmd.dark_color,
        page: cmd.page,
        blend: cmd.blend,
    })
}

/// Sutherland-Hodgman clip of one triangle against a convex polygon (CCW),
/// returning the clipped polygon's vertices with interpolated UVs.
fn clip_triangle(p: &[Vec2; 3], uv: &[(f32, f32); 3], polygon: &[Vec2]) -> Vec<(Vec2, (f32, f32))> {
    let mut subject = vec![(p[0], uv[0]), (p[1], uv[1]), (p[2], uv[2])];
    let m = polygon.len();
    for i in 0..m {
        if subject.len() < 3 {
            break;
        }
        subject = clip_against_edge(&subject, polygon[i], polygon[(i + 1) % m]);
    }
    subject
}

/// Clip a subject polygon against the inside half-plane of directed edge
/// `e1 -> e2` (left side, for a CCW clip polygon), interpolating UVs at crossings.
fn clip_against_edge(
    subject: &[(Vec2, (f32, f32))],
    e1: Vec2,
    e2: Vec2,
) -> Vec<(Vec2, (f32, f32))> {
    let n = subject.len();
    let edge = e2 - e1;
    let inside = |pt: Vec2| edge.x * (pt.y - e1.y) - edge.y * (pt.x - e1.x) >= 0.0;
    let mut out = Vec::with_capacity(n + 1);
    for i in 0..n {
        let (cur, cur_uv) = subject[i];
        let (prev, prev_uv) = subject[(i + n - 1) % n];
        let cur_in = inside(cur);
        let prev_in = inside(prev);
        if cur_in {
            if !prev_in {
                out.push(edge_intersect(prev, prev_uv, cur, cur_uv, e1, e2));
            }
            out.push((cur, cur_uv));
        } else if prev_in {
            out.push(edge_intersect(prev, prev_uv, cur, cur_uv, e1, e2));
        }
    }
    out
}

/// The point (with interpolated UV) where segment `a -> b` crosses the line
/// through `e1 -> e2`.
fn edge_intersect(
    a: Vec2,
    a_uv: (f32, f32),
    b: Vec2,
    b_uv: (f32, f32),
    e1: Vec2,
    e2: Vec2,
) -> (Vec2, (f32, f32)) {
    let d = b - a;
    let edge = e2 - e1;
    let denom = edge.x * d.y - edge.y * d.x;
    let t = if denom.abs() > 1e-9 {
        (edge.x * (e1.y - a.y) - edge.y * (e1.x - a.x)) / denom
    } else {
        0.0
    };
    let pt = a + d * t;
    (
        pt,
        (
            a_uv.0 + (b_uv.0 - a_uv.0) * t,
            a_uv.1 + (b_uv.1 - a_uv.1) * t,
        ),
    )
}

/// Reverse `poly` to counter-clockwise winding (positive signed area) if needed,
/// so the clip half-plane tests treat its interior as "inside".
fn make_ccw(mut poly: Vec<Vec2>) -> Vec<Vec2> {
    let n = poly.len();
    let mut area = 0.0;
    for i in 0..n {
        let a = poly[i];
        let b = poly[(i + 1) % n];
        area += a.x * b.y - b.x * a.y;
    }
    if area < 0.0 {
        poly.reverse();
    }
    poly
}

/// Decompose a clip polygon into convex CCW pieces: the polygon itself when it is
/// already convex, otherwise an ear-clipping triangulation. Empty if degenerate.
fn clip_pieces(poly: Vec<Vec2>) -> Vec<Vec<Vec2>> {
    if poly.len() < 3 {
        return Vec::new();
    }
    let poly = make_ccw(poly);
    if is_convex(&poly) {
        vec![poly]
    } else {
        triangulate(&poly)
    }
}

/// Whether `poly`'s corners all turn the same way (it is convex).
fn is_convex(poly: &[Vec2]) -> bool {
    let n = poly.len();
    let mut sign = 0.0_f32;
    for i in 0..n {
        let a = poly[i];
        let b = poly[(i + 1) % n];
        let c = poly[(i + 2) % n];
        let cross = (b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x);
        if cross.abs() > 1e-6 {
            if sign == 0.0 {
                sign = cross.signum();
            } else if cross.signum() != sign {
                return false;
            }
        }
    }
    true
}

/// Ear-clipping triangulation of a simple CCW polygon into CCW triangles.
fn triangulate(poly: &[Vec2]) -> Vec<Vec<Vec2>> {
    let mut indices: Vec<usize> = (0..poly.len()).collect();
    let mut tris = Vec::new();
    while indices.len() > 3 {
        let m = indices.len();
        let mut ear = None;
        for i in 0..m {
            let a = poly[indices[(i + m - 1) % m]];
            let b = poly[indices[i]];
            let c = poly[indices[(i + 1) % m]];
            // A convex corner with no other vertex inside triangle a,b,c is an ear.
            if (b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x) <= 0.0 {
                continue;
            }
            let clear = indices.iter().enumerate().all(|(j, &vi)| {
                j == (i + m - 1) % m
                    || j == i
                    || j == (i + 1) % m
                    || !point_in_triangle(poly[vi], a, b, c)
            });
            if clear {
                ear = Some((i, [a, b, c]));
                break;
            }
        }
        let Some((i, tri)) = ear else {
            break; // degenerate polygon: stop with what we have
        };
        tris.push(tri.to_vec());
        indices.remove(i);
    }
    if indices.len() == 3 {
        tris.push(vec![poly[indices[0]], poly[indices[1]], poly[indices[2]]]);
    }
    tris
}

/// Whether point `p` lies within triangle `a, b, c` (inclusive of edges).
fn point_in_triangle(p: Vec2, a: Vec2, b: Vec2, c: Vec2) -> bool {
    let d1 = tri_sign(p, a, b);
    let d2 = tri_sign(p, b, c);
    let d3 = tri_sign(p, c, a);
    let has_neg = d1 < 0.0 || d2 < 0.0 || d3 < 0.0;
    let has_pos = d1 > 0.0 || d2 > 0.0 || d3 > 0.0;
    !(has_neg && has_pos)
}

/// Signed area (doubled) of triangle `p, a, b`; its sign gives the turn side.
fn tri_sign(p: Vec2, a: Vec2, b: Vec2) -> f32 {
    (p.x - b.x) * (a.y - b.y) - (a.x - b.x) * (p.y - b.y)
}

/// Component-wise color multiply (slot tint times attachment tint).
fn mul(a: Color, b: Color) -> Color {
    Color::new(a.r * b.r, a.g * b.g, a.b * b.b, a.a * b.a)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::atlas::AtlasRegion;
    use crate::attach::{ClippingAttachment, MeshAttachment, MeshVertices, RegionAttachment};
    use crate::data::{BoneData, SkeletonData, SlotData};

    fn one_bone_skeleton(x: f32, y: f32, attachment: &str, skin: Skin) -> Skeleton {
        let data = SkeletonData {
            bones: vec![BoneData {
                index: 0,
                name: "root".into(),
                position: Vec2::new(x, y),
                ..Default::default()
            }],
            slots: vec![SlotData {
                index: 0,
                name: "s".into(),
                bone: 0,
                color: Color::WHITE,
                dark_color: None,
                attachment: Some(attachment.into()),
                blend: BlendMode::Additive,
            }],
            default_skin: skin,
            ..Default::default()
        };
        let mut sk = Skeleton::new(Arc::new(data));
        sk.update_world_transform();
        sk
    }

    #[test]
    fn renders_a_mesh_slot() {
        let mut skin = Skin::new("default");
        skin.set(
            0,
            "tri",
            Attachment::Mesh(MeshAttachment::new(
                "tri",
                "tri",
                MeshVertices::Unweighted(vec![0.0, 0.0, 10.0, 0.0, 0.0, 10.0]),
                vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                vec![0, 1, 2],
            )),
        );
        let sk = one_bone_skeleton(10.0, 0.0, "tri", skin);
        let cmds = render(&sk);
        assert_eq!(cmds.len(), 1);
        let c = &cmds[0];
        assert_eq!(c.triangles, vec![0, 1, 2]);
        assert_eq!(c.positions.len(), 3);
        // root at (10, 0): local mesh vertices offset by +10 in x.
        assert!((c.positions[0].x - 10.0).abs() < 1e-3);
        assert!((c.positions[1].x - 20.0).abs() < 1e-3);
        assert_eq!(c.blend, BlendMode::Additive);
    }

    #[test]
    fn renders_a_region_slot() {
        let region = AtlasRegion {
            name: "r".into(),
            page: 2,
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
        let mut r = RegionAttachment::new("r", "r");
        r.width = 20.0;
        r.height = 10.0;
        r.update(&region, 64, 64);
        r.page = region.page;
        let mut skin = Skin::new("default");
        skin.set(0, "r", Attachment::Region(r));

        let sk = one_bone_skeleton(100.0, 50.0, "r", skin);
        let cmds = render(&sk);
        assert_eq!(cmds.len(), 1);
        let c = &cmds[0];
        assert_eq!(c.page, 2);
        assert_eq!(c.positions.len(), 4);
        assert_eq!(c.triangles, vec![0, 1, 2, 2, 3, 0]);
        assert_eq!(c.uvs.len(), 8);
        // 20x10 quad centered on (100, 50): BL corner at (90, 45).
        assert!(
            (c.positions[0].x - 90.0).abs() < 1e-2,
            "x={}",
            c.positions[0].x
        );
        assert!(
            (c.positions[0].y - 45.0).abs() < 1e-2,
            "y={}",
            c.positions[0].y
        );
    }

    #[test]
    fn clip_triangle_inside_is_unchanged() {
        let square = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(10.0, 0.0),
            Vec2::new(10.0, 10.0),
            Vec2::new(0.0, 10.0),
        ];
        let p = [
            Vec2::new(2.0, 2.0),
            Vec2::new(8.0, 2.0),
            Vec2::new(2.0, 8.0),
        ];
        let uv = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)];
        let clipped = clip_triangle(&p, &uv, &square);
        assert_eq!(clipped.len(), 3);
    }

    #[test]
    fn clip_triangle_crosses_right_edge() {
        let square = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(10.0, 0.0),
            Vec2::new(10.0, 10.0),
            Vec2::new(0.0, 10.0),
        ];
        // A=(5,5) inside, B=(15,5) outside (x>10), C=(5,8) inside.
        let p = [
            Vec2::new(5.0, 5.0),
            Vec2::new(15.0, 5.0),
            Vec2::new(5.0, 8.0),
        ];
        let uv = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)];
        let clipped = clip_triangle(&p, &uv, &square);
        // Clipped to four vertices including the two right-edge crossings.
        assert_eq!(clipped.len(), 4);
        // The (10,5) crossing lies halfway along A->B, so its UV is (0.5, 0).
        let at = clipped
            .iter()
            .find(|(pt, _)| (pt.x - 10.0).abs() < 1e-3 && (pt.y - 5.0).abs() < 1e-3)
            .expect("a vertex at (10,5)");
        let &(_, (u, v)) = at;
        assert!((u - 0.5).abs() < 1e-3, "u={u}");
        assert!(v.abs() < 1e-3, "v={v}");
        // Nothing escapes the clip square.
        for (pt, _) in &clipped {
            assert!(pt.x <= 10.0 + 1e-3, "{pt:?}");
        }
    }

    #[test]
    fn clipping_masks_a_following_slot() {
        // Slot 0: a clipping square [0,10]^2; slot 1: a mesh triangle that spills
        // past it. The clip ends on slot 1 ("m"), so the mesh is masked.
        let mut skin = Skin::new("default");
        skin.set(
            0,
            "clip",
            Attachment::Clipping(ClippingAttachment::new(
                "clip",
                "m",
                MeshVertices::Unweighted(vec![0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0]),
                4,
            )),
        );
        skin.set(
            1,
            "tri",
            Attachment::Mesh(MeshAttachment::new(
                "tri",
                "tri",
                MeshVertices::Unweighted(vec![5.0, 5.0, 30.0, 5.0, 5.0, 30.0]),
                vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                vec![0, 1, 2],
            )),
        );
        let data = SkeletonData {
            bones: vec![BoneData {
                index: 0,
                name: "root".into(),
                ..Default::default()
            }],
            slots: vec![
                SlotData {
                    index: 0,
                    name: "clipslot".into(),
                    bone: 0,
                    color: Color::WHITE,
                    dark_color: None,
                    attachment: Some("clip".into()),
                    blend: BlendMode::Additive,
                },
                SlotData {
                    index: 1,
                    name: "m".into(),
                    bone: 0,
                    color: Color::WHITE,
                    dark_color: None,
                    attachment: Some("tri".into()),
                    blend: BlendMode::Additive,
                },
            ],
            default_skin: skin,
            ..Default::default()
        };
        let mut sk = Skeleton::new(Arc::new(data));
        sk.update_world_transform();
        let cmds = render(&sk);
        // The clip attachment emits nothing; only the masked mesh remains.
        assert_eq!(cmds.len(), 1);
        let c = &cmds[0];
        // The mesh overflowed the square, so clipping bounded it and introduced
        // extra vertices (the original triangle had three).
        assert!(c.positions.len() > 3, "len={}", c.positions.len());
        for p in &c.positions {
            assert!(
                p.x <= 10.0 + 1e-3 && p.y <= 10.0 + 1e-3 && p.x >= -1e-3 && p.y >= -1e-3,
                "{p:?}"
            );
        }
    }

    #[test]
    fn convex_clip_stays_one_piece() {
        let square = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(10.0, 0.0),
            Vec2::new(10.0, 10.0),
            Vec2::new(0.0, 10.0),
        ];
        assert!(is_convex(&square));
        assert_eq!(clip_pieces(square).len(), 1);
    }

    #[test]
    fn concave_clip_decomposes_and_excludes_the_notch() {
        // L-shape (concave); the top-right region (x>4, y>4) is outside it.
        let l = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(10.0, 0.0),
            Vec2::new(10.0, 4.0),
            Vec2::new(4.0, 4.0),
            Vec2::new(4.0, 10.0),
            Vec2::new(0.0, 10.0),
        ];
        assert!(!is_convex(&l));
        let pieces = clip_pieces(l);
        assert!(
            pieces.len() >= 2,
            "concave clip should decompose: {}",
            pieces.len()
        );
        let uv = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)];
        // A triangle entirely in the notch survives against no piece.
        let notch = [
            Vec2::new(6.0, 6.0),
            Vec2::new(9.0, 6.0),
            Vec2::new(6.0, 9.0),
        ];
        let hits = pieces
            .iter()
            .filter(|poly| clip_triangle(&notch, &uv, poly).len() >= 3)
            .count();
        assert_eq!(hits, 0, "notch triangle must be fully clipped");
        // A triangle inside the L survives against at least one piece.
        let inside = [
            Vec2::new(1.0, 1.0),
            Vec2::new(3.0, 1.0),
            Vec2::new(1.0, 3.0),
        ];
        let hits2 = pieces
            .iter()
            .filter(|poly| clip_triangle(&inside, &uv, poly).len() >= 3)
            .count();
        assert!(hits2 >= 1, "inside triangle must survive");
    }
}
