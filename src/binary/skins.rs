//! The skin and attachment sections of a `.skel` export.
//!
//! A skin lists, per slot, the attachments it defines. Each attachment is a
//! flags byte (the type in the low three bits) followed by that type's fields.
//! Every index a skin stores (slots, skin bones and constraints, weighted
//! vertex bones, mesh triangles, the clipping end slot, string table
//! references) is checked against the table it points into.

use super::*;

/// A linked mesh whose source skin is known only by its index in Spine's
/// skin list until every skin is read.
pub(super) struct PendingLink {
    /// The skin holding the link: `None` for the default skin, else its index
    /// in `data.skins`.
    pub(super) owner: Option<usize>,
    /// The link's slot.
    pub(super) slot: usize,
    /// The link's name in its slot.
    pub(super) name: String,
    /// The source skin's index in Spine's skin list.
    pub(super) skin_index: usize,
}

/// Spine's skin list, which binary data indexes skins by: the default skin
/// first when the export has one, then the named skins.
#[derive(Debug, Clone, Copy)]
pub(super) struct SkinList {
    /// Whether the export has a default skin (one that lists slots).
    pub(super) has_default: bool,
    /// The number of named skins.
    pub(super) named: usize,
}

impl SkinList {
    /// The skin at `index` in Spine's list: `Some(None)` for the default
    /// skin, `Some(Some(i))` for `data.skins[i]`, and `None` past the list.
    pub(super) fn get(self, index: usize) -> Option<Option<usize>> {
        let named = if self.has_default {
            match index.checked_sub(1) {
                None => return Some(None),
                Some(named) => named,
            }
        } else {
            index
        };
        (named < self.named).then_some(Some(named))
    }
}

/// Name the source skin of each linked mesh in `links`, now that every skin
/// is read. A skin index past Spine's skin list is corrupt.
pub(super) fn name_link_skins(
    r: &mut BinaryReader,
    data: &mut SkeletonData,
    skins: SkinList,
    links: Vec<PendingLink>,
) {
    for link in links {
        let Some(source) = skins.get(link.skin_index) else {
            corrupt(r);
            continue;
        };
        let name = source
            .and_then(|i| data.skins.get(i))
            .map(|skin| skin.name.clone());
        let owner = match link.owner {
            None => Some(&mut data.default_skin),
            Some(i) => data.skins.get_mut(i),
        };
        if let Some(Attachment::LinkedMesh(mesh)) =
            owner.and_then(|skin| skin.attachment_mut(link.slot, &link.name))
        {
            mesh.skin = name;
        }
    }
}

/// Read a skin: the default skin (`is_default`, a slot count then attachments)
/// or a named skin (name, bone/constraint index lists, then attachments).
/// `data` holds the bones, slots, constraints, and skins read so far. The
/// default skin reads as `None` when it lists no slots, and Spine's skin list
/// then leaves it out. Each linked mesh read is added to `links`.
pub(super) fn read_skin(
    r: &mut BinaryReader,
    strings: &[String],
    data: &SkeletonData,
    is_default: bool,
    nonessential: bool,
    links: &mut Vec<PendingLink>,
) -> Option<Skin> {
    let mut skin;
    let slot_count;
    let owner = (!is_default).then_some(data.skins.len());
    if is_default {
        slot_count = r.count();
        skin = Skin::new("default");
        if slot_count == 0 {
            return None;
        }
    } else {
        skin = Skin::new(r.string().unwrap_or_default());
        if nonessential {
            let _ = r.u32();
        }
        // The skin's bone and constraint lists are not kept, but they are
        // still indices into those tables.
        let bones = r.count();
        for _ in 0..bones {
            if r.var_usize() >= data.bones.len() {
                corrupt(r);
            }
        }
        let constraints = constraint_count(data);
        let listed = r.count();
        for _ in 0..listed {
            if r.var_usize() >= constraints {
                corrupt(r);
            }
        }
        slot_count = r.count();
    }
    for _ in 0..slot_count {
        let slot = r.var_usize();
        if slot >= data.slots.len() {
            corrupt(r);
        }
        let att_count = r.count();
        for _ in 0..att_count {
            let placeholder = string_ref(r, strings).unwrap_or_default();
            let Some((att, link_skin)) =
                read_attachment(r, strings, data, &placeholder, nonessential)
            else {
                continue;
            };
            if let Some(skin_index) = link_skin {
                links.push(PendingLink {
                    owner,
                    slot,
                    name: placeholder.clone(),
                    skin_index,
                });
            }
            skin.set(slot, placeholder, att);
        }
    }
    Some(skin)
}

/// Read a list of slot indices, each checked against the slot table.
fn read_slot_list(r: &mut BinaryReader, slot_count: usize) -> Vec<usize> {
    let n = r.count();
    let mut slots: Vec<usize> = (0..n)
        .map(|_| {
            let slot = r.var_usize();
            if slot >= slot_count {
                corrupt(r);
            }
            slot
        })
        .collect();
    slots.sort_unstable();
    slots.dedup();
    slots
}

/// Read one attachment (Spine 4.3): a flags byte selects the type (low 3 bits)
/// and which optional fields follow. A linked mesh also returns its source
/// skin's index in Spine's skin list, which is resolved once every skin is
/// read.
fn read_attachment(
    r: &mut BinaryReader,
    strings: &[String],
    data: &SkeletonData,
    placeholder: &str,
    nonessential: bool,
) -> Option<(Attachment, Option<usize>)> {
    let bones = data.bones.len();
    let flags = r.byte();
    let name = if flags & 8 != 0 {
        string_ref(r, strings).unwrap_or_default()
    } else {
        placeholder.to_string()
    };
    let mut link_skin = None;
    let attachment = match flags & 0b111 {
        0 => {
            let path = if flags & 16 != 0 {
                string_ref(r, strings).unwrap_or_else(|| name.clone())
            } else {
                name.clone()
            };
            let color = read_att_color(r, flags & 32 != 0);
            let sequence = read_sequence(r, flags & 64 != 0);
            let rotation = if flags & 128 != 0 { r.float() } else { 0.0 };
            let mut reg = RegionAttachment::new(name, path);
            reg.x = r.float();
            reg.y = r.float();
            reg.scale_x = r.float();
            reg.scale_y = r.float();
            reg.width = r.float();
            reg.height = r.float();
            reg.rotation = rotation;
            reg.color = color;
            reg.sequence = sequence;
            Attachment::Region(reg)
        }
        1 => {
            let (vertices, count) = read_vertices(r, flags & 16 != 0, bones);
            skip_nonessential_color(r, nonessential);
            Attachment::BoundingBox(BoundingBoxAttachment::new(name, vertices, count))
        }
        2 => {
            let path = if flags & 16 != 0 {
                string_ref(r, strings).unwrap_or_else(|| name.clone())
            } else {
                name.clone()
            };
            let color = read_att_color(r, flags & 32 != 0);
            let sequence = read_sequence(r, flags & 64 != 0);
            let hull = r.var_usize();
            let (vertices, count) = read_vertices(r, flags & 128 != 0, bones);
            let uvs = read_float_array(r, count.saturating_mul(2));
            let tri_count = count
                .saturating_mul(2)
                .saturating_sub(hull.saturating_add(2))
                .saturating_mul(3);
            let triangles = read_triangles(r, tri_count, count);
            // The other slots that show linked meshes inheriting this mesh's
            // timelines.
            let timeline_slots = read_slot_list(r, data.slots.len());
            if nonessential {
                let edges = r.count();
                for _ in 0..edges {
                    r.var_usize();
                }
                let _ = r.float();
                let _ = r.float();
            }
            let mut m = MeshAttachment::new(name, path, vertices, uvs, triangles);
            m.hull_length = hull;
            m.color = color;
            m.sequence = sequence;
            m.timeline_slots = timeline_slots.into();
            Attachment::Mesh(m)
        }
        3 => {
            let path = if flags & 16 != 0 {
                string_ref(r, strings).unwrap_or_else(|| name.clone())
            } else {
                name.clone()
            };
            let color = read_att_color(r, flags & 32 != 0);
            let sequence = read_sequence(r, flags & 64 != 0);
            let inherit = flags & 128 != 0;
            let source_slot = r.var_usize();
            if source_slot >= data.slots.len() {
                corrupt(r);
            }
            link_skin = Some(r.var_usize());
            let source = string_ref(r, strings).unwrap_or_default();
            if nonessential {
                let _ = r.float();
                let _ = r.float();
            }
            // The source skin is named once every skin is read.
            let mut link = LinkedMeshAttachment::new(name, path, None, source, color, inherit);
            link.source_slot = Some(source_slot);
            link.sequence = sequence;
            Attachment::LinkedMesh(link)
        }
        4 => {
            let closed = flags & 16 != 0;
            let constant_speed = flags & 32 != 0;
            let (vertices, count) = read_vertices(r, flags & 64 != 0, bones);
            let lengths = read_float_array(r, count / 3);
            skip_nonessential_color(r, nonessential);
            Attachment::Path(PathAttachment::new(
                name,
                vertices,
                count,
                lengths,
                closed,
                constant_speed,
            ))
        }
        5 => {
            let rotation = r.float();
            let x = r.float();
            let y = r.float();
            skip_nonessential_color(r, nonessential);
            Attachment::Point(PointAttachment::new(name, x, y, rotation))
        }
        6 => {
            let end = r.var_usize();
            let (vertices, count) = read_vertices(r, flags & 16 != 0, bones);
            skip_nonessential_color(r, nonessential);
            let end_slot = match data.slots.get(end) {
                Some(slot) => slot.name.clone(),
                None => {
                    corrupt(r);
                    String::new()
                }
            };
            Attachment::Clipping(ClippingAttachment::new(name, end_slot, vertices, count))
        }
        // Type 7 is not an attachment type, and its fields cannot be skipped.
        _ => {
            corrupt(r);
            return None;
        }
    };
    Some((attachment, link_skin))
}

/// Read an attachment color (RGBA8888) when present, else opaque white.
fn read_att_color(r: &mut BinaryReader, present: bool) -> Color {
    if present {
        color_rgba(r.u32())
    } else {
        Color::WHITE
    }
}

/// Consume the trailing editor-only color int present on some attachments.
fn skip_nonessential_color(r: &mut BinaryReader, nonessential: bool) {
    if nonessential {
        let _ = r.u32();
    }
}

/// Consume an animation `Sequence` (4 varints) when the attachment has one.
fn read_sequence(r: &mut BinaryReader, present: bool) -> Option<Sequence> {
    if !present {
        return None;
    }
    let count = r.count();
    let start = r.var_usize();
    let digits = r.var_usize();
    let setup_index = r.var_usize();
    Some(Sequence::new(count, start, digits, setup_index))
}

/// Read mesh/polygon vertices: unweighted (`2 * vertexCount` floats) or weighted
/// (per-vertex bone influences, each naming one of `bone_count` bones).
/// Returns the vertices and the vertex count.
fn read_vertices(r: &mut BinaryReader, weighted: bool, bone_count: usize) -> (MeshVertices, usize) {
    let count = r.count();
    if !weighted {
        return (
            MeshVertices::Unweighted(read_float_array(r, count.saturating_mul(2))),
            count,
        );
    }
    // `total` is the length of the `[influences, bone...]` layout. Spine sizes
    // the layout from it and the weights from `total - count`, so the layout
    // must end exactly at `total` and describe at least `count` vertices.
    let total = r.count();
    let mut bones = Vec::new();
    let mut vertices = Vec::new();
    let mut described = 0_usize;
    while bones.len() < total {
        let influences = r.count();
        bones.push(influences);
        for _ in 0..influences {
            let bone = r.var_usize();
            if bone >= bone_count {
                corrupt(r);
            }
            bones.push(bone);
            vertices.push(r.float());
            vertices.push(r.float());
            vertices.push(r.float());
        }
        described += 1;
    }
    if bones.len() != total || described < count {
        corrupt(r);
    }
    (MeshVertices::Weighted { bones, vertices }, count)
}

/// Read `n` big-endian floats. A count the remaining data cannot hold is
/// corrupt, and reads as an empty array.
fn read_float_array(r: &mut BinaryReader, n: usize) -> Vec<f32> {
    if n > r.remaining() / 4 {
        corrupt(r);
        return Vec::new();
    }
    (0..n).map(|_| r.float()).collect()
}

/// Read `n` var-uint triangle indices into a mesh of `vertex_count` vertices.
/// A list longer than the remaining data, or an index that names no vertex
/// (or does not fit the `u16` index type), is corrupt.
fn read_triangles(r: &mut BinaryReader, n: usize, vertex_count: usize) -> Vec<u16> {
    if n > r.remaining() {
        corrupt(r);
        return Vec::new();
    }
    (0..n)
        .map(|_| {
            let index = r.var_usize();
            match u16::try_from(index) {
                Ok(t) if index < vertex_count => t,
                _ => {
                    corrupt(r);
                    0
                }
            }
        })
        .collect()
}
