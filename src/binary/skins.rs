//! The skin and attachment sections of a `.skel` export.
//!
//! A skin lists, per slot, the attachments it defines. Each attachment is a
//! flags byte (the type in the low three bits) followed by that type's fields.
//! Every index a skin stores (slots, skin bones and constraints, weighted
//! vertex bones, mesh triangles, the clipping end slot, string table
//! references) is checked against the table it points into.

use super::*;

/// Read a skin: the default skin (`is_default`, a slot count then attachments)
/// or a named skin (name, bone/constraint index lists, then attachments).
/// `data` holds the bones, slots, and constraints read so far.
pub(super) fn read_skin(
    r: &mut BinaryReader,
    strings: &[String],
    data: &SkeletonData,
    is_default: bool,
    nonessential: bool,
) -> Skin {
    let mut skin;
    let slot_count;
    if is_default {
        slot_count = r.count();
        skin = Skin::new("default");
        if slot_count == 0 {
            return skin;
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
            if let Some(att) = read_attachment(r, strings, data, &placeholder, nonessential) {
                skin.set(slot, placeholder, att);
            }
        }
    }
    skin
}

/// Read one attachment (Spine 4.3): a flags byte selects the type (low 3 bits)
/// and which optional fields follow.
fn read_attachment(
    r: &mut BinaryReader,
    strings: &[String],
    data: &SkeletonData,
    placeholder: &str,
    nonessential: bool,
) -> Option<Attachment> {
    let bones = data.bones.len();
    let flags = r.byte();
    let name = if flags & 8 != 0 {
        string_ref(r, strings).unwrap_or_default()
    } else {
        placeholder.to_string()
    };
    match flags & 0b111 {
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
            Some(Attachment::Region(reg))
        }
        1 => {
            let (vertices, count) = read_vertices(r, flags & 16 != 0, bones);
            skip_nonessential_color(r, nonessential);
            Some(Attachment::BoundingBox(BoundingBoxAttachment::new(
                name, vertices, count,
            )))
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
            let timeline_slots = r.count();
            for _ in 0..timeline_slots {
                r.var_usize();
            }
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
            Some(Attachment::Mesh(m))
        }
        3 => {
            let path = if flags & 16 != 0 {
                string_ref(r, strings).unwrap_or_else(|| name.clone())
            } else {
                name.clone()
            };
            let color = read_att_color(r, flags & 32 != 0);
            let _ = read_sequence(r, flags & 64 != 0);
            let inherit = flags & 128 != 0;
            let _source_index = r.var_usize();
            let _skin_index = r.var_usize();
            let parent = string_ref(r, strings).unwrap_or_default();
            if nonessential {
                let _ = r.float();
                let _ = r.float();
            }
            Some(Attachment::LinkedMesh(LinkedMeshAttachment::new(
                name, path, None, parent, color, inherit,
            )))
        }
        4 => {
            let closed = flags & 16 != 0;
            let constant_speed = flags & 32 != 0;
            let (vertices, count) = read_vertices(r, flags & 64 != 0, bones);
            let lengths = read_float_array(r, count / 3);
            skip_nonessential_color(r, nonessential);
            Some(Attachment::Path(PathAttachment::new(
                name,
                vertices,
                count,
                lengths,
                closed,
                constant_speed,
            )))
        }
        5 => {
            let rotation = r.float();
            let x = r.float();
            let y = r.float();
            skip_nonessential_color(r, nonessential);
            Some(Attachment::Point(PointAttachment::new(
                name, x, y, rotation,
            )))
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
            Some(Attachment::Clipping(ClippingAttachment::new(
                name, end_slot, vertices, count,
            )))
        }
        // Type 7 is not an attachment type, and its fields cannot be skipped.
        _ => {
            corrupt(r);
            None
        }
    }
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
