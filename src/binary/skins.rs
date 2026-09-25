//! The skin and attachment sections of a `.skel` export.
//!
//! A skin lists, per slot, the attachments it defines. Each attachment is a
//! flags byte (the type in the low three bits) followed by that type's fields.

use super::*;

/// Read a skin: the default skin (`is_default`, a slot count then attachments)
/// or a named skin (name, bone/constraint index lists, then attachments).
pub(super) fn read_skin(
    r: &mut BinaryReader,
    strings: &[String],
    slot_names: &[String],
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
        let bone_count = r.count();
        for _ in 0..bone_count {
            r.var_usize();
        }
        let constraint_count = r.count();
        for _ in 0..constraint_count {
            r.var_usize();
        }
        slot_count = r.count();
    }
    for _ in 0..slot_count {
        let slot = r.var_usize();
        let att_count = r.count();
        for _ in 0..att_count {
            let placeholder = string_ref(r, strings).unwrap_or_default();
            if let Some(att) = read_attachment(r, strings, slot_names, &placeholder, nonessential) {
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
    slot_names: &[String],
    placeholder: &str,
    nonessential: bool,
) -> Option<Attachment> {
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
            let (vertices, count) = read_vertices(r, flags & 16 != 0);
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
            let (vertices, count) = read_vertices(r, flags & 128 != 0);
            let uvs = read_float_array(r, count * 2);
            let tri_count = (count * 2).saturating_sub(hull + 2) * 3;
            let triangles = read_short_array(r, tri_count);
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
            let (vertices, count) = read_vertices(r, flags & 64 != 0);
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
            let (vertices, count) = read_vertices(r, flags & 16 != 0);
            skip_nonessential_color(r, nonessential);
            let end_slot = slot_names.get(end).cloned().unwrap_or_default();
            Some(Attachment::Clipping(ClippingAttachment::new(
                name, end_slot, vertices, count,
            )))
        }
        _ => None,
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
/// (per-vertex bone influences). Returns the vertices and the vertex count.
fn read_vertices(r: &mut BinaryReader, weighted: bool) -> (MeshVertices, usize) {
    let count = r.count();
    if !weighted {
        return (
            MeshVertices::Unweighted(read_float_array(r, count * 2)),
            count,
        );
    }
    let total = r.count();
    let mut bones = Vec::new();
    let mut vertices = Vec::new();
    while bones.len() < total {
        let influences = r.count();
        bones.push(influences);
        for _ in 0..influences {
            bones.push(r.var_usize());
            vertices.push(r.float());
            vertices.push(r.float());
            vertices.push(r.float());
        }
    }
    (MeshVertices::Weighted { bones, vertices }, count)
}

/// Read `n` big-endian floats.
fn read_float_array(r: &mut BinaryReader, n: usize) -> Vec<f32> {
    (0..n).map(|_| r.float()).collect()
}

/// Read `n` var-uint shorts (triangle / edge indices).
fn read_short_array(r: &mut BinaryReader, n: usize) -> Vec<u16> {
    (0..n).map(|_| r.var_usize() as u16).collect()
}
