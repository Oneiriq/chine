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
use crate::attach::Attachment;
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
                if let Some(region) = atlas.find_region(&r.path) {
                    let page = &atlas.pages[region.page];
                    let (pw, ph) = (page.width, page.height);
                    r.page = region.page;
                    r.update(region, pw, ph);
                }
            }
            Attachment::Mesh(m) => {
                if let Some(region) = atlas.find_region(&m.path) {
                    let page = &atlas.pages[region.page];
                    let (pw, ph) = (page.width, page.height);
                    m.page = region.page;
                    m.remap_uvs(region, pw, ph);
                }
            }
            Attachment::Path(_) => {}
        }
    }
}

/// Produce the draw-order render command stream for a posed skeleton.
///
/// Call [`Skeleton::update_world_transform`] first (to pose the bones) and
/// [`bind_atlas`] once at load time (to resolve UVs). Walks the skeleton's
/// runtime draw order, reading each slot's current attachment and tint (driven
/// by slot timelines); the blend mode comes from the setup data.
#[must_use]
pub fn render(skeleton: &Skeleton) -> Vec<RenderCommand> {
    let data = skeleton.data();
    let mut out = Vec::new();
    for &slot_index in skeleton.draw_order() {
        let (Some(setup), Some(slot)) = (data.slots.get(slot_index), skeleton.slot(slot_index))
        else {
            continue;
        };
        let Some(name) = slot.attachment.as_deref() else {
            continue;
        };
        let Some(att) = data.attachment(slot_index, name, None) else {
            continue;
        };
        match att {
            Attachment::Region(r) => {
                let Some(bone) = skeleton.bone(setup.bone) else {
                    continue;
                };
                out.push(RenderCommand {
                    positions: r.compute_world_vertices(bone).to_vec(),
                    uvs: r.uvs.to_vec(),
                    triangles: vec![0, 1, 2, 2, 3, 0],
                    color: mul(slot.color, r.color),
                    page: r.page,
                    blend: setup.blend,
                });
            }
            Attachment::Mesh(m) => {
                out.push(RenderCommand {
                    positions: m.compute_world_vertices(skeleton, setup.bone, &slot.deform),
                    uvs: m.uvs.clone(),
                    triangles: m.triangles.clone(),
                    color: mul(slot.color, m.color),
                    page: m.page,
                    blend: setup.blend,
                });
            }
            Attachment::Path(_) => {}
        }
    }
    out
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
    use crate::attach::{MeshAttachment, MeshVertices, RegionAttachment};
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
}
