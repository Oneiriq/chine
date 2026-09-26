//! Render command generation: turn a posed skeleton into renderer-agnostic draw
//! data.
//!
//! [`render`] walks a skeleton's slots in draw order and emits a
//! [`RenderCommand`] per visible attachment: world-space positions, page UVs,
//! triangles, tint, blend mode, and atlas page. The host uploads these to the
//! GPU. chine does no rendering itself. Call [`bind_atlas`] once after loading
//! so attachment UVs and page indices are resolved against the atlas.
//!
//! A render loop should hold a [`RenderScratch`] and call [`render_with`]
//! every frame: the commands and the clipper's internal buffers are then
//! reused, so steady-state rendering performs no heap allocation.

use std::collections::HashMap;

use glam::Vec2;

use crate::atlas::{Atlas, AtlasPage, AtlasRegion};
use crate::attach::{Attachment, MeshAttachment, RegionAttachment, Sequence};
use crate::clip::{clip_into, clip_outside_into, replace_clip_within, spend, CLIP_WORK_BUDGET};
use crate::data::{BlendMode, Color, SkeletonData, SlotData};
use crate::skel::{Skeleton, Slot};
use crate::skin::Skin;

pub use crate::clip::RenderScratch;

/// One batch of geometry: an attachment's triangles with its texture and tint.
#[derive(Debug, Clone, Default, PartialEq)]
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
///
/// Sequence binding draws on a fixed work budget, far above what real rigs
/// use: many sequences over a large atlas, or long sequences of large meshes,
/// could otherwise take unbounded time and memory. Once the budget is spent,
/// the remaining sequence frames are left unbound and show their last bound
/// frame, or the attachment's static UVs when none was bound.
pub fn bind_atlas(data: &mut SkeletonData, atlas: &Atlas) {
    let mut binder = Binder::new(atlas);
    binder.bind_skin(&mut data.default_skin);
    for skin in &mut data.skins {
        binder.bind_skin(skin);
    }
}

/// The work one [`bind_atlas`] call may spend on sequences, counted in name
/// bytes built or scanned while matching frames to regions, plus UV values
/// stored.
const SEQUENCE_BIND_BUDGET: usize = 1 << 26;

/// An atlas indexed by region name, and the sequence binding budget left.
struct Binder<'a> {
    atlas: &'a Atlas,
    /// The first region declared under each name, which is the one
    /// [`Atlas::find_region`] returns.
    by_name: HashMap<&'a str, &'a AtlasRegion>,
    /// The cost of scanning every region name once.
    scan_cost: usize,
    budget: usize,
}

impl<'a> Binder<'a> {
    fn new(atlas: &'a Atlas) -> Self {
        let mut by_name = HashMap::with_capacity(atlas.regions.len());
        let mut scan_cost = 0_usize;
        for region in &atlas.regions {
            by_name.entry(region.name.as_str()).or_insert(region);
            scan_cost = scan_cost.saturating_add(region.name.len().saturating_add(1));
        }
        Self {
            atlas,
            by_name,
            scan_cost,
            budget: SEQUENCE_BIND_BUDGET,
        }
    }

    fn bind_skin(&mut self, skin: &mut Skin) {
        for att in skin.attachments_mut() {
            match att {
                Attachment::Region(r) => {
                    if r.sequence.is_some() {
                        self.bind_region_sequence(r);
                    } else if let Some((region, page)) = self.find(&r.path) {
                        r.page = region.page;
                        r.update(region, page.width, page.height);
                    }
                }
                Attachment::Mesh(m) => {
                    if m.sequence.is_some() {
                        self.bind_mesh_sequence(m);
                    } else if let Some((region, page)) = self.find(&m.path) {
                        m.page = region.page;
                        m.remap_uvs(region, page.width, page.height);
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

    /// The atlas region named `name` and its page. A region whose page index
    /// is outside the atlas (possible only in a hand-built `Atlas`) counts as
    /// missing.
    fn find(&self, name: &str) -> Option<(&'a AtlasRegion, &'a AtlasPage)> {
        let region: &'a AtlasRegion = self.by_name.get(name)?;
        let page = self.atlas.pages.get(region.page)?;
        Some((region, page))
    }

    /// Every sequence frame below `seq.count` whose region exists in the atlas,
    /// in frame order, with its region and page.
    ///
    /// This either looks each frame's name up or scans the atlas once for names
    /// that parse as the sequence's frames, whichever costs less, so the work
    /// is bounded by the atlas even when `count` is huge. Both give the same
    /// frames. The work is paid from the budget. Unpaid, no frame is found.
    fn sequence_regions(
        &mut self,
        seq: &Sequence,
        base: &str,
    ) -> Vec<(usize, &'a AtlasRegion, &'a AtlasPage)> {
        let atlas = self.atlas;
        let regions = &atlas.regions;
        let lookup_cost = seq
            .count
            .saturating_mul(seq.max_name_len(base).saturating_add(1));
        let lookup = lookup_cost <= self.scan_cost;
        if !spend(&mut self.budget, lookup_cost.min(self.scan_cost)) {
            return Vec::new();
        }
        if lookup {
            return (0..seq.count)
                .filter_map(|i| {
                    let (region, page) = self.find(&seq.region_name(base, i))?;
                    Some((i, region, page))
                })
                .collect();
        }
        let mut found: Vec<(usize, &str)> = regions
            .iter()
            .filter_map(|region| Some((seq.frame_index(base, &region.name)?, region.name.as_str())))
            .filter(|&(index, _)| index < seq.count)
            .collect();
        // Regions sharing a name share a frame index. Keep one entry per frame.
        found.sort_by_key(|&(index, _)| index);
        found.dedup_by_key(|&mut (index, _)| index);
        found
            .into_iter()
            .filter_map(|(index, name)| {
                let (region, page) = self.find(name)?;
                Some((index, region, page))
            })
            .collect()
    }

    /// Store `uvs` and `page` for the sequence frames from `first` on, paying
    /// one unit of budget per UV value. `false`, storing nothing, when the
    /// budget cannot pay.
    fn push_frames(&mut self, seq: &mut Sequence, first: usize, uvs: &[f32], page: usize) -> bool {
        if !spend(&mut self.budget, uvs.len()) {
            return false;
        }
        seq.push_frame(first, uvs.to_vec(), page);
        true
    }

    /// Resolve a region attachment's sequence: bind each frame's UVs and page,
    /// then bake the static corner offsets / UVs / page to the setup frame.
    fn bind_region_sequence(&mut self, r: &mut RegionAttachment) {
        let Some(mut seq) = r.sequence.take() else {
            return;
        };
        self.bind_region_frames(&mut seq, r);
        if let Some((region, page)) = self.find(&seq.region_name(&r.path, seq.setup_index)) {
            r.page = region.page;
            r.update(region, page.width, page.height);
        }
        r.sequence = Some(seq);
    }

    /// Bind a region sequence's frames. A frame with no atlas region reuses the
    /// UVs of the latest frame before it that has one (the attachment's own
    /// UVs if none) with the attachment's page. Such frames are stored as one
    /// run per gap.
    fn bind_region_frames(&mut self, seq: &mut Sequence, r: &mut RegionAttachment) {
        let fallback_page = r.page;
        let mut next = 0;
        for (index, region, page) in self.sequence_regions(seq, &r.path) {
            if index > next && !self.push_frames(seq, next, &r.uvs, fallback_page) {
                return;
            }
            r.update(region, page.width, page.height);
            if !self.push_frames(seq, index, &r.uvs, region.page) {
                return;
            }
            next = index + 1;
        }
        if next < seq.count {
            self.push_frames(seq, next, &r.uvs, fallback_page);
        }
    }

    /// Resolve a mesh attachment's sequence: remap the original UVs into each
    /// frame's region, then leave the static UVs at the setup frame.
    fn bind_mesh_sequence(&mut self, m: &mut MeshAttachment) {
        let Some(mut seq) = m.sequence.take() else {
            return;
        };
        let original = m.uvs.clone();
        self.bind_mesh_frames(&mut seq, m, &original);
        m.uvs.clone_from(&original);
        if let Some((region, page)) = self.find(&seq.region_name(&m.path, seq.setup_index)) {
            m.page = region.page;
            m.remap_uvs(region, page.width, page.height);
        }
        m.sequence = Some(seq);
    }

    /// Bind a mesh sequence's frames from the `original` UVs. A frame with no
    /// atlas region keeps the original UVs and the attachment's page. Such
    /// frames are stored as one run per gap.
    fn bind_mesh_frames(&mut self, seq: &mut Sequence, m: &mut MeshAttachment, original: &[f32]) {
        let mut next = 0;
        for (index, region, page) in self.sequence_regions(seq, &m.path) {
            if index > next && !self.push_frames(seq, next, original, m.page) {
                return;
            }
            m.uvs.clear();
            m.uvs.extend_from_slice(original);
            m.remap_uvs(region, page.width, page.height);
            if !self.push_frames(seq, index, &m.uvs, region.page) {
                return;
            }
            next = index + 1;
        }
        if next < seq.count {
            self.push_frames(seq, next, original, m.page);
        }
    }
}

/// The bound UVs and page of a sequenced attachment at the slot's frame index
/// (`-1` uses the attachment's setup index), or `None` for a non-sequenced
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
/// by slot timelines). The blend mode comes from the setup data.
#[must_use]
pub fn render(skeleton: &Skeleton) -> Vec<RenderCommand> {
    let mut out = Vec::new();
    render_into(skeleton, &mut out);
    out
}

/// Build the skeleton's draw-order render commands into `out`, which is cleared
/// first. A render loop should keep one buffer and call this every frame to
/// reuse its capacity, rather than calling [`render`] and allocating a fresh
/// `Vec` each frame. To also reuse the commands' own buffers and the clipper's
/// internals, hold a [`RenderScratch`] and call [`render_with`] instead.
pub fn render_into(skeleton: &Skeleton, out: &mut Vec<RenderCommand>) {
    out.clear();
    let mut scratch = RenderScratch::new();
    let emitted = render_with(skeleton, &mut scratch).len();
    out.extend(scratch.commands.drain(..emitted));
}

/// Build the skeleton's draw-order render commands in `scratch`, returning the
/// frame's commands. Equivalent to [`render_into`], but every buffer involved
/// (the commands themselves and the clipper's intermediates) lives in
/// `scratch` and is rebuilt in place: keep one scratch per render loop and,
/// once its buffers have grown to the skeleton's working sizes, steady-state
/// rendering, clipped or not, performs no heap allocation.
///
/// Clipping draws on a fixed work budget per frame, far above what real rigs
/// use, so a hostile rig cannot make a frame arbitrarily slow. Once the budget
/// is spent, the frame's remaining clipped slots draw nothing.
#[must_use]
pub fn render_with<'a>(skeleton: &Skeleton, scratch: &'a mut RenderScratch) -> &'a [RenderCommand] {
    let RenderScratch {
        commands,
        staging,
        clip_world,
        poly_points,
        poly_ranges,
        ear,
        subject,
        clipped,
        outside,
    } = scratch;
    poly_points.clear();
    poly_ranges.clear();
    let mut emitted = 0;
    let mut budget = CLIP_WORK_BUDGET;
    // The active clip's end slot: clipping stops once that slot is drawn.
    let mut clip_end: Option<usize> = None;
    // Whether the active clip keeps what lies outside its polygon.
    let mut inverse = false;
    for &slot_index in skeleton.draw_order() {
        if let Some((setup, slot, att)) = resolve_attachment(skeleton, slot_index) {
            if let Attachment::Clipping(c) = att {
                // Start masking: decompose the clip polygon into convex
                // pieces, or take its convex hull for a convex or inverse
                // clip. A degenerate polygon leaves any active clip running.
                clip_world.clear();
                c.compute_world_vertices_into(skeleton, setup.bone, clip_world);
                let hull = c.convex || c.inverse;
                if replace_clip_within(clip_world, hull, ear, poly_points, poly_ranges, &mut budget)
                {
                    // The end slot lookup scans the slot list, so it is paid
                    // from the budget too. Unpaid, it counts as not found.
                    let slots = &skeleton.data().slots;
                    let end = if spend(&mut budget, slots.len()) {
                        skeleton.data().find_slot(&c.end_slot)
                    } else {
                        None
                    };
                    clip_end = Some(end.unwrap_or(usize::MAX));
                    inverse = c.inverse;
                }
            } else {
                // Build a drawable slot into the next output command, or into
                // the staging command first when a clip is active.
                let dst = if clip_end.is_some() {
                    &mut *staging
                } else {
                    command_slot(commands, emitted)
                };
                if fill_command(skeleton, setup, slot, att, dst) {
                    if clip_end.is_some() {
                        let dst = command_slot(commands, emitted);
                        // An inverse clip has one piece, the polygon's hull.
                        let drawn = if inverse {
                            clip_outside_into(
                                staging,
                                poly_ranges
                                    .first()
                                    .and_then(|&(start, len)| {
                                        poly_points.get(start..start.checked_add(len)?)
                                    })
                                    .unwrap_or_default(),
                                subject,
                                clipped,
                                outside,
                                dst,
                                &mut budget,
                            )
                        } else {
                            clip_into(
                                staging,
                                poly_points,
                                poly_ranges,
                                subject,
                                clipped,
                                dst,
                                &mut budget,
                            )
                        };
                        if drawn {
                            emitted += 1;
                        }
                    } else {
                        emitted += 1;
                    }
                }
            }
        }
        if clip_end == Some(slot_index) {
            clip_end = None;
        }
    }
    &commands[..emitted]
}

/// Resolve a slot's drawable state: its setup data, runtime state, and current
/// attachment. `None` if the slot shows nothing or a reference is dangling.
fn resolve_attachment(
    skeleton: &Skeleton,
    slot_index: usize,
) -> Option<(&SlotData, &Slot, &Attachment)> {
    let data = skeleton.data();
    let setup = data.slots.get(slot_index)?;
    let slot = skeleton.slot(slot_index)?;
    let name = slot.attachment.as_deref()?;
    let att = data.attachment(slot_index, name, skeleton.active_skin())?;
    Some((setup, slot, att))
}

/// The command at `index` in the scratch's grow-only pool, pushing an empty
/// command first when the pool is shorter. A reused command is overwritten in
/// place, so its buffers keep their capacity across frames.
fn command_slot(commands: &mut Vec<RenderCommand>, index: usize) -> &mut RenderCommand {
    if index == commands.len() {
        commands.push(RenderCommand::default());
    }
    &mut commands[index]
}

/// Fill `dst` (clearing and reusing its buffers) with one slot's draw data.
/// `false`, leaving `dst` untouched, for attachments that draw nothing:
/// paths, bounding boxes, points, and unresolved linked meshes. Clipping
/// attachments are handled by the caller, not here.
fn fill_command(
    skeleton: &Skeleton,
    setup: &SlotData,
    slot: &Slot,
    att: &Attachment,
    dst: &mut RenderCommand,
) -> bool {
    match att {
        Attachment::Region(r) => {
            let Some(bone) = skeleton.bone(setup.bone) else {
                return false;
            };
            dst.positions.clear();
            dst.positions
                .extend_from_slice(&r.compute_world_vertices(bone));
            dst.uvs.clear();
            match sequence_frame(r.sequence.as_ref(), slot.sequence_index) {
                Some((uvs, page)) => {
                    dst.uvs.extend_from_slice(uvs);
                    dst.page = page;
                }
                None => {
                    dst.uvs.extend_from_slice(&r.uvs);
                    dst.page = r.page;
                }
            }
            dst.triangles.clear();
            dst.triangles.extend_from_slice(&[0, 1, 2, 2, 3, 0]);
            dst.color = mul(slot.color, r.color);
            dst.dark_color = slot.dark_color;
            dst.blend = setup.blend;
            true
        }
        Attachment::Mesh(m) => {
            m.compute_world_vertices_into(skeleton, setup.bone, &slot.deform, &mut dst.positions);
            dst.uvs.clear();
            match sequence_frame(m.sequence.as_ref(), slot.sequence_index) {
                Some((uvs, page)) => {
                    dst.uvs.extend_from_slice(uvs);
                    dst.page = page;
                }
                None => {
                    dst.uvs.extend_from_slice(&m.uvs);
                    dst.page = m.page;
                }
            }
            // Malformed mesh data can leave the positions, UVs, and triangles
            // out of step. Keep the vertices that have both a position and a
            // UV pair, and the triangles whose corners are all among them, so
            // every command is self-consistent. Valid meshes pass unchanged.
            let vertices = dst.positions.len().min(dst.uvs.len() / 2);
            dst.positions.truncate(vertices);
            dst.uvs.truncate(vertices * 2);
            dst.triangles.clear();
            for tri in m.triangles.chunks(3) {
                if tri.len() == 3 && tri.iter().all(|&t| usize::from(t) < vertices) {
                    dst.triangles.extend_from_slice(tri);
                }
            }
            dst.color = mul(slot.color, m.color);
            dst.dark_color = slot.dark_color;
            dst.blend = setup.blend;
            true
        }
        Attachment::Clipping(_)
        | Attachment::Path(_)
        | Attachment::BoundingBox(_)
        | Attachment::Point(_)
        | Attachment::LinkedMesh(_) => false,
    }
}

/// Component-wise color multiply (slot tint times attachment tint).
fn mul(a: Color, b: Color) -> Color {
    Color::new(a.r * b.r, a.g * b.g, a.b * b.b, a.a * b.a)
}

#[cfg(test)]
mod robustness;

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
    fn render_into_clears_and_reuses_the_buffer() {
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

        // A buffer holding stale commands is cleared, then refilled to match a
        // fresh render.
        let mut buf = render(&sk);
        buf.extend(render(&sk)); // now two stale commands
        render_into(&sk, &mut buf);
        let fresh = render(&sk);
        assert_eq!(buf.len(), 1, "stale commands were not cleared");
        assert_eq!(buf.len(), fresh.len());
        assert_eq!(buf[0].page, fresh[0].page);
        assert_eq!(buf[0].positions.len(), fresh[0].positions.len());
        assert_eq!(buf[0].triangles, fresh[0].triangles);
    }

    /// Slot 0: a clipping square `[0,10]^2`. Slot 1: a mesh triangle that
    /// spills past it. The clip ends on slot 1 ("m"), so the mesh is masked.
    fn clipped_skeleton() -> Skeleton {
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
        sk
    }

    #[test]
    fn clipping_masks_a_following_slot() {
        let sk = clipped_skeleton();
        let cmds = render(&sk);
        // The clip attachment emits nothing. Only the masked mesh remains.
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

    // A reused scratch must be invisible in the output: rendering the same
    // clipped skeleton a second time through one scratch (whose buffers now
    // hold the first frame) yields exactly the commands of a fresh scratch
    // and of the allocating wrappers.
    #[test]
    fn reused_scratch_render_matches_a_fresh_render() {
        let sk = clipped_skeleton();
        let mut reused = RenderScratch::new();
        let first = render_with(&sk, &mut reused).to_vec();
        let second = render_with(&sk, &mut reused);

        let mut fresh = RenderScratch::new();
        let expected = render_with(&sk, &mut fresh);
        assert_eq!(second, expected);
        assert_eq!(second, first.as_slice());
        assert_eq!(second, render(&sk).as_slice());
    }

    /// The four blend modes, in the slot order the fixtures declare them.
    const ALL_BLENDS: [BlendMode; 4] = [
        BlendMode::Normal,
        BlendMode::Additive,
        BlendMode::Multiply,
        BlendMode::Screen,
    ];

    // End to end through the JSON loader: one slot per blend mode, each
    // emitted RenderCommand must carry its slot's mode.
    #[cfg(feature = "json")]
    #[test]
    fn json_blend_modes_reach_their_render_commands() {
        let json = r#"{
            "bones": [ { "name": "root" } ],
            "slots": [
                { "name": "n", "bone": "root", "attachment": "n" },
                { "name": "a", "bone": "root", "attachment": "a", "blend": "additive" },
                { "name": "m", "bone": "root", "attachment": "m", "blend": "multiply" },
                { "name": "s", "bone": "root", "attachment": "s", "blend": "screen" }
            ],
            "skins": [ {
                "name": "default",
                "attachments": {
                    "n": { "n": { "width": 10, "height": 10 } },
                    "a": { "a": { "width": 10, "height": 10 } },
                    "m": { "m": { "width": 10, "height": 10 } },
                    "s": { "s": { "width": 10, "height": 10 } }
                }
            } ]
        }"#;
        let data = crate::load::from_json(json).unwrap();
        let mut sk = Skeleton::new(Arc::new(data));
        sk.update_world_transform();
        let cmds = render(&sk);
        assert_eq!(cmds.len(), 4);
        for (cmd, expected) in cmds.iter().zip(ALL_BLENDS) {
            assert_eq!(cmd.blend, expected);
            assert_eq!(cmd.positions.len(), 4, "a region quad per slot");
        }
    }

    // End to end through the binary loader: a hand-built `.skel` with one
    // region-attachment slot per blend ordinal (0..=3), rendered, must emit
    // each RenderCommand with its slot's mode.
    #[cfg(feature = "binary")]
    #[test]
    fn binary_blend_modes_reach_their_render_commands() {
        fn put_string(out: &mut Vec<u8>, s: &str) {
            out.push(u8::try_from(s.len() + 1).unwrap());
            out.extend_from_slice(s.as_bytes());
        }
        fn put_f32(out: &mut Vec<u8>, v: f32) {
            out.extend_from_slice(&v.to_be_bytes());
        }

        let names = ["n", "a", "m", "s"];
        let mut b = Vec::new();
        // Header: hash, version, bounds, reference scale, essential.
        b.extend_from_slice(&[0; 8]);
        put_string(&mut b, "4.3.00");
        for v in [0.0_f32, 0.0, 0.0, 0.0, 1.0] {
            put_f32(&mut b, v);
        }
        b.push(0);
        // String table: the four attachment names.
        b.push(4);
        for name in names {
            put_string(&mut b, name);
        }
        // One root bone.
        b.push(1);
        put_string(&mut b, "root");
        for v in [0.0_f32, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0] {
            put_f32(&mut b, v);
        }
        b.push(0); // inherit = Normal
        b.push(0); // skin required = false
                   // Four slots: blend ordinals 0..=3, attachment = string ref i + 1.
        b.push(4);
        for (i, name) in names.iter().enumerate() {
            let i = u8::try_from(i).unwrap();
            put_string(&mut b, name);
            b.push(0); // bone 0
            b.extend_from_slice(&0xFFFF_FFFF_u32.to_be_bytes()); // color
            b.extend_from_slice(&0xFFFF_FFFF_u32.to_be_bytes()); // dark (none)
            b.push(i + 1); // attachment string ref
            b.push(i); // blend ordinal
        }
        b.push(0); // constraint count
                   // Default skin: one plain region attachment per slot.
        b.push(4);
        for i in 0..4_u8 {
            b.push(i); // slot index
            b.push(1); // one attachment
            b.push(i + 1); // placeholder string ref
            b.push(0); // flags: region, name = placeholder, no extras
            for v in [0.0_f32, 0.0, 1.0, 1.0, 10.0, 10.0] {
                put_f32(&mut b, v); // x, y, scaleX, scaleY, width, height
            }
        }
        b.extend_from_slice(&[0; 3]); // named skins, events, animations

        let data = crate::binary::from_binary(&b).expect("blend fixture parses");
        let mut sk = Skeleton::new(Arc::new(data));
        sk.update_world_transform();
        let cmds = render(&sk);
        assert_eq!(cmds.len(), 4);
        for (cmd, expected) in cmds.iter().zip(ALL_BLENDS) {
            assert_eq!(cmd.blend, expected);
            assert_eq!(cmd.positions.len(), 4, "a region quad per slot");
        }
    }
}
