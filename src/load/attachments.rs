//! Skin attachment parsing for the JSON loader.
//!
//! The runtime indexes vertex, UV, triangle, and bone arrays by each other's
//! counts, so every attachment is checked here: the arrays must agree with
//! the counts the attachment declares, and weighted vertices must name
//! existing bones. The Spine editor always writes consistent data, so these
//! checks only reject corrupt exports.

use serde_json::Value;

use super::{bool_or, f, f_array, f_or, parse_color, LoadError};
use crate::attach::{
    Attachment, BoundingBoxAttachment, ClippingAttachment, LinkedMeshAttachment, MeshAttachment,
    MeshVertices, PathAttachment, PointAttachment, RegionAttachment, Sequence,
};
use crate::data::Color;

/// Parse one skin attachment stored under `placeholder`, its key in the
/// skin. Returns `Ok(None)` for an unknown attachment type, which the loader
/// skips. `bone_count` bounds the bone indices of weighted vertices. A linked
/// mesh's source slot is a slot name, which the caller resolves.
///
/// # Errors
/// Returns [`LoadError::Schema`] when the vertex, UV, triangle, hull, or
/// sequence data is inconsistent, and [`LoadError::BadReference`] when a
/// weighted vertex names a bone that does not exist.
pub(super) fn read_attachment(
    placeholder: &str,
    v: &Value,
    bone_count: usize,
) -> Result<Option<Attachment>, LoadError> {
    // An attachment's name defaults to its key, and its path to its name.
    let name = v.get("name").and_then(Value::as_str).unwrap_or(placeholder);
    let path = v
        .get("path")
        .and_then(Value::as_str)
        .unwrap_or(name)
        .to_string();
    let kind = v.get("type").and_then(Value::as_str).unwrap_or("region");
    // Spine 4.3 marks a linked mesh by naming its source, on a "mesh" or a
    // "linkedmesh". Earlier exports name the source "parent".
    let source = v
        .get("source")
        .or_else(|| v.get("parent"))
        .and_then(Value::as_str);
    if kind == "linkedmesh" || (kind == "mesh" && source.is_some()) {
        let mut link = LinkedMeshAttachment::new(
            name,
            path,
            v.get("skin").and_then(Value::as_str).map(str::to_string),
            source.unwrap_or(placeholder),
            parse_color(v.get("color").and_then(Value::as_str), Color::WHITE),
            v.get("timelines")
                .or_else(|| v.get("deform"))
                .and_then(Value::as_bool)
                .unwrap_or(true),
        );
        link.sequence = parse_sequence(name, v)?;
        return Ok(Some(Attachment::LinkedMesh(link)));
    }
    let attachment = match kind {
        "region" => {
            let mut r = RegionAttachment::new(name, path);
            r.x = f(v, "x");
            r.y = f(v, "y");
            r.scale_x = f_or(v, "scaleX", 1.0);
            r.scale_y = f_or(v, "scaleY", 1.0);
            r.rotation = f(v, "rotation");
            r.width = f(v, "width");
            r.height = f(v, "height");
            r.color = parse_color(v.get("color").and_then(Value::as_str), Color::WHITE);
            r.sequence = parse_sequence(name, v)?;
            Attachment::Region(r)
        }
        "mesh" => {
            let uvs = f_array(v, "uvs");
            if uvs.len() % 2 == 1 {
                return Err(schema(name, "has an odd number of UV values"));
            }
            let vertex_count = uvs.len() / 2;
            let vertices = read_vertices(name, v, uvs.len(), bone_count)?;
            let triangles = read_triangles(name, v, vertex_count)?;
            let hull_length = read_count(name, v, "hull")?;
            if hull_length > vertex_count {
                return Err(schema(name, "has a hull longer than its vertex list"));
            }
            let mut m = MeshAttachment::new(name, path, vertices, uvs, triangles);
            m.color = parse_color(v.get("color").and_then(Value::as_str), Color::WHITE);
            m.hull_length = hull_length;
            m.sequence = parse_sequence(name, v)?;
            Attachment::Mesh(m)
        }
        "path" => {
            let (vertices, count) = read_polygon(name, v, bone_count)?;
            Attachment::Path(PathAttachment::new(
                name,
                vertices,
                count,
                f_array(v, "lengths"),
                bool_or(v, "closed", false),
                bool_or(v, "constantSpeed", true),
            ))
        }
        "boundingbox" => {
            let (vertices, count) = read_polygon(name, v, bone_count)?;
            Attachment::BoundingBox(BoundingBoxAttachment::new(name, vertices, count))
        }
        "point" => Attachment::Point(PointAttachment::new(
            name,
            f(v, "x"),
            f(v, "y"),
            f(v, "rotation"),
        )),
        "clipping" => {
            let (vertices, count) = read_polygon(name, v, bone_count)?;
            let end = v
                .get("end")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            let mut clip = ClippingAttachment::new(name, end, vertices, count);
            clip.convex = bool_or(v, "convex", false);
            clip.inverse = bool_or(v, "inverse", false);
            Attachment::Clipping(clip)
        }
        _ => return Ok(None),
    };
    Ok(Some(attachment))
}

/// A schema error for the attachment `name`.
fn schema(name: &str, problem: &str) -> LoadError {
    LoadError::Schema(format!("attachment '{name}' {problem}"))
}

/// Read a count field. A field that is absent or not a non-negative integer
/// reads as 0. A value that does not fit in `usize` is an error.
fn read_count(name: &str, v: &Value, key: &str) -> Result<usize, LoadError> {
    match v.get(key).and_then(Value::as_u64) {
        None => Ok(0),
        Some(n) => usize::try_from(n).map_err(|_| schema(name, &format!("has too large a {key}"))),
    }
}

/// Read the vertices of a path, bounding box, or clipping polygon, sized by
/// its `vertexCount`. Returns the vertices and the vertex count.
fn read_polygon(
    name: &str,
    v: &Value,
    bone_count: usize,
) -> Result<(MeshVertices, usize), LoadError> {
    let count = read_count(name, v, "vertexCount")?;
    let world_len = count
        .checked_mul(2)
        .ok_or_else(|| schema(name, "has too large a vertexCount"))?;
    Ok((read_vertices(name, v, world_len, bone_count)?, count))
}

/// Read a vertex attachment's `vertices` for `world_len` world coordinates
/// (two per vertex). As in Spine's `readVertices`, an array of exactly
/// `world_len` floats is unweighted and anything else is weighted.
fn read_vertices(
    name: &str,
    v: &Value,
    world_len: usize,
    bone_count: usize,
) -> Result<MeshVertices, LoadError> {
    let raw = f_array(v, "vertices");
    if raw.len() == world_len {
        return Ok(MeshVertices::Unweighted(raw));
    }
    parse_weighted(name, &raw, world_len / 2, bone_count)
}

/// Decode Spine's weighted vertex array: per vertex a bone count, then
/// `[bone, x, y, weight]` per bone. The array must hold exactly
/// `vertex_count` complete vertices, each bound to bones that exist.
fn parse_weighted(
    name: &str,
    raw: &[f32],
    vertex_count: usize,
    bone_count: usize,
) -> Result<MeshVertices, LoadError> {
    let mut bones = Vec::new();
    let mut vertices = Vec::new();
    let mut groups = 0_usize;
    let mut rest = raw;
    while let Some((&count, tail)) = rest.split_first() {
        rest = tail;
        // Spine truncates the count to an integer, and a negative count binds
        // no bones. The cast saturates, so a huge count runs out of data.
        let count = count as usize;
        bones.push(count);
        for _ in 0..count {
            let Some((&[bone, x, y, weight], tail)) = rest.split_first_chunk::<4>() else {
                return Err(schema(name, "has a truncated weighted vertex"));
            };
            rest = tail;
            let bone = bone_index(bone, bone_count)
                .ok_or_else(|| LoadError::BadReference(format!("bone index {bone}")))?;
            bones.push(bone);
            vertices.extend([x, y, weight]);
        }
        groups += 1;
    }
    if groups != vertex_count {
        return Err(schema(
            name,
            "has weighted vertices that do not match its vertex count",
        ));
    }
    Ok(MeshVertices::Weighted { bones, vertices })
}

/// The bone index a weighted vertex stores as a float, if it names one of
/// `bone_count` bones. Spine truncates toward zero, so `-0.5` is bone 0.
fn bone_index(value: f32, bone_count: usize) -> Option<usize> {
    if value.is_nan() || value <= -1.0 {
        return None;
    }
    let index = value as usize;
    (index < bone_count).then_some(index)
}

/// Read a mesh's triangle list. Entries that are not non-negative integers
/// are skipped. Every other entry must index one of the `vertex_count`
/// vertices.
fn read_triangles(name: &str, v: &Value, vertex_count: usize) -> Result<Vec<u16>, LoadError> {
    let Some(values) = v.get("triangles").and_then(Value::as_array) else {
        return Ok(Vec::new());
    };
    values
        .iter()
        .filter_map(Value::as_u64)
        .map(|index| {
            usize::try_from(index)
                .ok()
                .filter(|&i| i < vertex_count)
                .and_then(|i| u16::try_from(i).ok())
                .ok_or_else(|| schema(name, "has a triangle index past its vertex list"))
        })
        .collect()
}

/// Parse the optional `sequence` (flipbook) object on a region or mesh
/// attachment: `count` regions starting at `start`, zero-padded to `digits`,
/// showing `setup` at rest. A field that is absent or not a non-negative
/// integer takes its default. Spine stores each field as a signed 32-bit
/// integer, so larger values are rejected, which also keeps `start + index`
/// in range.
fn parse_sequence(name: &str, v: &Value) -> Result<Option<Sequence>, LoadError> {
    let Some(s) = v.get("sequence").and_then(Value::as_object) else {
        return Ok(None);
    };
    let field = |key: &str, default: usize| match s.get(key).and_then(Value::as_u64) {
        None => Ok(default),
        Some(n) => i32::try_from(n)
            .ok()
            .and_then(|n| usize::try_from(n).ok())
            .ok_or_else(|| schema(name, &format!("has a sequence {key} out of range"))),
    };
    Ok(Some(Sequence::new(
        field("count", 0)?,
        field("start", 1)?,
        field("digits", 0)?,
        field("setup", 0)?,
    )))
}

/// Bytes that binding an atlas allocates for `seq`: per region, a frame of
/// `floats` UVs plus the region name built to look it up.
fn sequence_cost(seq: &Sequence, floats: usize, path: &str) -> usize {
    let per_frame = floats
        .saturating_mul(size_of::<f32>())
        .saturating_add(size_of::<Vec<f32>>() + size_of::<usize>())
        .saturating_add(path.len())
        .saturating_add(seq.digits);
    seq.count.saturating_add(1).saturating_mul(per_frame)
}

/// Bytes an attachment expands to beyond its own JSON: a sequenced region or
/// mesh gets per-frame UVs when an atlas is bound.
pub(super) fn attachment_cost(attachment: &Attachment) -> usize {
    match attachment {
        Attachment::Region(r) => r
            .sequence
            .as_ref()
            .map_or(0, |s| sequence_cost(s, 8, &r.path)),
        Attachment::Mesh(m) => m
            .sequence
            .as_ref()
            .map_or(0, |s| sequence_cost(s, m.uvs.len(), &m.path)),
        _ => 0,
    }
}

/// Bytes a resolved linked mesh takes: a copy of its source's UVs,
/// triangles, and vertices (sized from the deform length, which counts two
/// floats per weighted influence), plus the frames of its own sequence.
pub(super) fn linked_copy_cost(link: &LinkedMeshAttachment, source: &MeshAttachment) -> usize {
    source
        .uvs
        .len()
        .saturating_mul(size_of::<f32>())
        .saturating_add(source.triangles.len().saturating_mul(size_of::<u16>()))
        .saturating_add(source.deform_len().saturating_mul(size_of::<usize>()))
        .saturating_add(
            link.sequence
                .as_ref()
                .map_or(0, |s| sequence_cost(s, source.uvs.len(), &link.path)),
        )
}
