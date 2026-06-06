//! Loading skeleton data from Spine exports.
//!
//! Behind the `json` feature, [`from_json`] parses a Spine `.json` export into
//! a [`SkeletonData`] rig: bones, slots, skins, region / mesh / path
//! attachments, animations, and IK / transform / path / physics / slider
//! constraints. (The binary `.skel` loader lives in the `binary` module.)
//!
//! Region attachments are parsed with their transform but without UVs; those
//! are filled in once an [`crate::atlas::Atlas`] is bound (the UV/offset layout
//! depends on the packed region).

use std::error::Error;
use std::fmt;
use std::sync::Arc;

use glam::Vec2;
use serde_json::Value;

use crate::attach::{
    Attachment, BoundingBoxAttachment, ClippingAttachment, LinkedMeshAttachment, MeshAttachment,
    MeshVertices, PathAttachment, PointAttachment, RegionAttachment, Sequence,
};
use crate::constraint::ik::IkConstraintData;
use crate::constraint::path::{PathConstraintData, PositionMode, RotateMode, SpacingMode};
use crate::constraint::physics::PhysicsConstraintData;
use crate::constraint::transform::{
    FromMapping, FromProp, ToMapping, ToProp, TransformConstraintData,
};
use crate::constraint::ScaleYMode;
use crate::data::{BlendMode, BoneData, Color, Inherit, SkeletonData, SlotData};
use crate::event::EventData;
use crate::skin::Skin;

/// An error parsing a Spine export.
#[derive(Debug)]
pub enum LoadError {
    /// The JSON text was malformed.
    Json(serde_json::Error),
    /// A referenced bone or slot name was not found.
    BadReference(String),
    /// A required field was missing or had the wrong type.
    Schema(String),
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Json(e) => write!(f, "invalid JSON: {e}"),
            Self::BadReference(name) => write!(f, "unknown reference: {name}"),
            Self::Schema(msg) => write!(f, "schema error: {msg}"),
        }
    }
}

impl Error for LoadError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Json(e) => Some(e),
            _ => None,
        }
    }
}

/// Parse a Spine `.json` skeleton export into a [`SkeletonData`] rig.
///
/// # Errors
/// Returns [`LoadError`] if the JSON is malformed, a required field is missing,
/// or a bone/slot reference can't be resolved.
pub fn from_json(text: &str) -> Result<SkeletonData, LoadError> {
    let root: Value = serde_json::from_str(text).map_err(LoadError::Json)?;
    let mut data = SkeletonData::default();

    if let Some(skel) = root.get("skeleton") {
        data.spine_version = skel.get("spine").and_then(Value::as_str).map(String::from);
        data.position = Vec2::new(f(skel, "x"), f(skel, "y"));
        data.size = Vec2::new(f(skel, "width"), f(skel, "height"));
        data.reference_scale = f_or(skel, "referenceScale", 100.0);
    }

    if let Some(bones) = root.get("bones").and_then(Value::as_array) {
        for (index, b) in bones.iter().enumerate() {
            let name = str_field(b, "name")?;
            let parent = match b.get("parent").and_then(Value::as_str) {
                Some(p) => Some(
                    data.find_bone(p)
                        .ok_or_else(|| LoadError::BadReference(p.to_string()))?,
                ),
                None => None,
            };
            data.bones.push(BoneData {
                index,
                name,
                parent,
                length: f(b, "length"),
                position: Vec2::new(f(b, "x"), f(b, "y")),
                rotation: f(b, "rotation"),
                scale: Vec2::new(f_or(b, "scaleX", 1.0), f_or(b, "scaleY", 1.0)),
                shear: Vec2::new(f(b, "shearX"), f(b, "shearY")),
                inherit: parse_inherit(b.get("inherit").or(b.get("transform"))),
            });
        }
    }

    if let Some(slots) = root.get("slots").and_then(Value::as_array) {
        for (index, s) in slots.iter().enumerate() {
            let name = str_field(s, "name")?;
            let bone_name = str_field(s, "bone")?;
            let bone = data
                .find_bone(&bone_name)
                .ok_or(LoadError::BadReference(bone_name))?;
            data.slots.push(SlotData {
                index,
                name,
                bone,
                color: parse_color(s.get("color").and_then(Value::as_str), Color::WHITE),
                dark_color: s
                    .get("dark")
                    .and_then(Value::as_str)
                    .map(|h| parse_color(Some(h), Color::WHITE)),
                attachment: s
                    .get("attachment")
                    .and_then(Value::as_str)
                    .map(String::from),
                blend: parse_blend(s.get("blend").and_then(Value::as_str)),
            });
        }
    }

    if let Some(skins) = root.get("skins").and_then(Value::as_array) {
        for sk in skins {
            let skin_name = sk
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or("default")
                .to_string();
            let mut skin = Skin::new(skin_name.clone());
            if let Some(slot_map) = sk.get("attachments").and_then(Value::as_object) {
                for (slot_name, atts) in slot_map {
                    let slot = data
                        .find_slot(slot_name)
                        .ok_or_else(|| LoadError::BadReference(slot_name.clone()))?;
                    if let Some(atts) = atts.as_object() {
                        for (att_name, att) in atts {
                            if let Some(mut attachment) = parse_attachment(att_name, att) {
                                if let Attachment::Mesh(m) = &mut attachment {
                                    if skin_name != "default" {
                                        m.deform_skin = Some(skin_name.clone());
                                    }
                                }
                                skin.set(slot, att_name.clone(), attachment);
                            }
                        }
                    }
                }
            }
            if skin_name == "default" {
                data.default_skin = skin;
            } else {
                data.skins.push(skin);
            }
        }
    }

    if let Some(constraints) = root.get("constraints").and_then(Value::as_array) {
        for (order, cm) in constraints.iter().enumerate() {
            match cm.get("type").and_then(Value::as_str) {
                Some("ik") => {
                    let c = parse_ik(cm, order, &data)?;
                    data.ik_constraints.push(c);
                }
                Some("transform") => {
                    let c = parse_transform(cm, order, &data)?;
                    data.transform_constraints.push(c);
                }
                Some("path") => {
                    let c = parse_path(cm, order, &data)?;
                    data.path_constraints.push(c);
                }
                Some("physics") => {
                    let c = parse_physics(cm, order, &data)?;
                    data.physics_constraints.push(c);
                }
                _ => {}
            }
        }
    }

    if let Some(events) = root.get("events").and_then(Value::as_object) {
        for (name, e) in events {
            data.events.push(EventData {
                name: name.clone(),
                int_value: e.get("int").and_then(Value::as_i64).unwrap_or(0) as i32,
                float_value: f(e, "float"),
                string_value: e
                    .get("string")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string(),
                audio_path: e.get("audio").and_then(Value::as_str).map(String::from),
                volume: f_or(e, "volume", 1.0),
                balance: f(e, "balance"),
            });
        }
    }

    // Resolve linked meshes before animations so deform timelines bind to the
    // resolved (parent-shared) geometry rather than unresolved links.
    crate::link::resolve_linked_meshes(&mut data);

    if let Some(anims) = root.get("animations").and_then(Value::as_object) {
        for (name, anim) in anims {
            let animation = parse_animation(name, anim, &data)?;
            data.animations.push(Arc::new(animation));
        }
    }
    Ok(data)
}

/// Parse the optional `sequence` (flipbook) object on a region or mesh
/// attachment: `count` regions starting at `start`, zero-padded to `digits`,
/// showing `setup` at rest.
fn parse_sequence(v: &Value) -> Option<Sequence> {
    let s = v.get("sequence")?.as_object()?;
    let u = |key: &str, default: u64| s.get(key).and_then(Value::as_u64).unwrap_or(default) as usize;
    Some(Sequence::new(
        u("count", 0),
        u("start", 1),
        u("digits", 0),
        u("setup", 0),
    ))
}

fn parse_attachment(name: &str, v: &Value) -> Option<Attachment> {
    let path = v
        .get("path")
        .and_then(Value::as_str)
        .unwrap_or(name)
        .to_string();
    match v.get("type").and_then(Value::as_str).unwrap_or("region") {
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
            r.sequence = parse_sequence(v);
            Some(Attachment::Region(r))
        }
        "mesh" => {
            let uvs = f_array(v, "uvs");
            let triangles = u16_array(v, "triangles");
            let raw = f_array(v, "vertices");
            let vertices = if raw.len() == uvs.len() {
                MeshVertices::Unweighted(raw)
            } else {
                parse_weighted(&raw)
            };
            let mut m = MeshAttachment::new(name, path, vertices, uvs, triangles);
            m.color = parse_color(v.get("color").and_then(Value::as_str), Color::WHITE);
            m.hull_length = v.get("hull").and_then(Value::as_u64).unwrap_or(0) as usize;
            m.sequence = parse_sequence(v);
            Some(Attachment::Mesh(m))
        }
        "path" => {
            let count = v.get("vertexCount").and_then(Value::as_u64).unwrap_or(0) as usize;
            let raw = f_array(v, "vertices");
            let vertices = if raw.len() == count * 2 {
                MeshVertices::Unweighted(raw)
            } else {
                parse_weighted(&raw)
            };
            Some(Attachment::Path(PathAttachment::new(
                name,
                vertices,
                count,
                f_array(v, "lengths"),
                bool_or(v, "closed", false),
                bool_or(v, "constantSpeed", true),
            )))
        }
        "boundingbox" => {
            let count = v.get("vertexCount").and_then(Value::as_u64).unwrap_or(0) as usize;
            let raw = f_array(v, "vertices");
            let vertices = if raw.len() == count * 2 {
                MeshVertices::Unweighted(raw)
            } else {
                parse_weighted(&raw)
            };
            Some(Attachment::BoundingBox(BoundingBoxAttachment::new(
                name, vertices, count,
            )))
        }
        "point" => Some(Attachment::Point(PointAttachment::new(
            name,
            f(v, "x"),
            f(v, "y"),
            f(v, "rotation"),
        ))),
        "linkedmesh" => Some(Attachment::LinkedMesh(LinkedMeshAttachment::new(
            name,
            path,
            v.get("skin").and_then(Value::as_str).map(str::to_string),
            v.get("parent").and_then(Value::as_str).unwrap_or(name),
            parse_color(v.get("color").and_then(Value::as_str), Color::WHITE),
            v.get("deform").and_then(Value::as_bool).unwrap_or(true),
        ))),
        "clipping" => {
            let count = v.get("vertexCount").and_then(Value::as_u64).unwrap_or(0) as usize;
            let raw = f_array(v, "vertices");
            let vertices = if raw.len() == count * 2 {
                MeshVertices::Unweighted(raw)
            } else {
                parse_weighted(&raw)
            };
            let end = v
                .get("end")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            Some(Attachment::Clipping(ClippingAttachment::new(
                name, end, vertices, count,
            )))
        }
        _ => None,
    }
}

/// Decode Spine's weighted-vertex array: `[count, boneIdx, x, y, weight, ...]`.
fn parse_weighted(raw: &[f32]) -> MeshVertices {
    let mut bones = Vec::new();
    let mut vertices = Vec::new();
    let mut i = 0;
    while i < raw.len() {
        let count = raw[i] as usize;
        i += 1;
        bones.push(count);
        for _ in 0..count {
            if i + 4 > raw.len() {
                break;
            }
            bones.push(raw[i] as usize); // bone index
            vertices.push(raw[i + 1]); // x
            vertices.push(raw[i + 2]); // y
            vertices.push(raw[i + 3]); // weight
            i += 4;
        }
    }
    MeshVertices::Weighted { bones, vertices }
}


/// Resolve a constraint's named constrained bones to indices.
fn constraint_bones(cm: &Value, kind: &str, data: &SkeletonData) -> Result<Vec<usize>, LoadError> {
    let mut bones = Vec::new();
    if let Some(bs) = cm.get("bones").and_then(Value::as_array) {
        for bn in bs {
            let bn = bn
                .as_str()
                .ok_or_else(|| LoadError::Schema(format!("{kind} bone name")))?;
            bones.push(
                data.find_bone(bn)
                    .ok_or_else(|| LoadError::BadReference(bn.to_string()))?,
            );
        }
    }
    Ok(bones)
}

/// Parse an `ik` constraint entry.
fn parse_ik(cm: &Value, order: usize, data: &SkeletonData) -> Result<IkConstraintData, LoadError> {
    let name = str_field(cm, "name")?;
    let bones = constraint_bones(cm, "ik", data)?;
    let target_name = str_field(cm, "target")?;
    let target = data
        .find_bone(&target_name)
        .ok_or(LoadError::BadReference(target_name))?;
    let scale_y_mode = match cm.get("scaleY").and_then(Value::as_str) {
        Some("uniform") => ScaleYMode::Uniform,
        Some("volume") => ScaleYMode::Volume,
        _ => ScaleYMode::None,
    };
    Ok(IkConstraintData {
        name,
        order,
        bones,
        target,
        scale_y_mode,
        mix: f_or(cm, "mix", 1.0),
        softness: f(cm, "softness"),
        bend_direction: if bool_or(cm, "bendPositive", true) {
            1
        } else {
            -1
        },
        compress: bool_or(cm, "compress", false),
        stretch: bool_or(cm, "stretch", false),
    })
}

/// Parse a `transform` constraint entry, including its source-to-target property
/// map.
fn parse_transform(
    cm: &Value,
    order: usize,
    data: &SkeletonData,
) -> Result<TransformConstraintData, LoadError> {
    let name = str_field(cm, "name")?;
    let bones = constraint_bones(cm, "transform", data)?;
    let source_name = str_field(cm, "source")?;
    let source = data
        .find_bone(&source_name)
        .ok_or(LoadError::BadReference(source_name))?;
    let offsets = [
        f(cm, "rotation"),
        f(cm, "x"),
        f(cm, "y"),
        f(cm, "scaleX"),
        f(cm, "scaleY"),
        f(cm, "shearY"),
    ];
    let mut properties = Vec::new();
    if let Some(props) = cm.get("properties").and_then(Value::as_object) {
        for (from_name, from_val) in props {
            let Some(property) = from_prop(from_name) else {
                continue;
            };
            let mut to = Vec::new();
            if let Some(tos) = from_val.get("to").and_then(Value::as_object) {
                for (to_name, to_val) in tos {
                    if let Some(tprop) = to_prop(to_name) {
                        to.push(ToMapping {
                            property: tprop,
                            offset: f(to_val, "offset"),
                            max: f_or(to_val, "max", 1.0),
                            scale: f_or(to_val, "scale", 1.0),
                        });
                    }
                }
            }
            if !to.is_empty() {
                properties.push(FromMapping {
                    property,
                    offset: f(from_val, "offset"),
                    to,
                });
            }
        }
    }
    let mix_x = f_or(cm, "mixX", 1.0);
    let mix_scale_x = f_or(cm, "mixScaleX", 1.0);
    Ok(TransformConstraintData {
        name,
        order,
        bones,
        source,
        offsets,
        local_source: bool_or(cm, "localSource", false),
        local_target: bool_or(cm, "localTarget", false),
        additive: bool_or(cm, "additive", false),
        clamp: bool_or(cm, "clamp", false),
        properties,
        mix_rotate: f_or(cm, "mixRotate", 1.0),
        mix_x,
        mix_y: f_or(cm, "mixY", mix_x),
        mix_scale_x,
        mix_scale_y: f_or(cm, "mixScaleY", mix_scale_x),
        mix_shear_y: f_or(cm, "mixShearY", 1.0),
    })
}

/// Parse a `path` constraint entry.
fn parse_path(
    cm: &Value,
    order: usize,
    data: &SkeletonData,
) -> Result<PathConstraintData, LoadError> {
    let name = str_field(cm, "name")?;
    let bones = constraint_bones(cm, "path", data)?;
    let slot_name = str_field(cm, "slot")?;
    let slot = data
        .find_slot(&slot_name)
        .ok_or(LoadError::BadReference(slot_name))?;
    let position_mode = match cm.get("positionMode").and_then(Value::as_str) {
        Some("fixed") => PositionMode::Fixed,
        _ => PositionMode::Percent,
    };
    let spacing_mode = match cm.get("spacingMode").and_then(Value::as_str) {
        Some("fixed") => SpacingMode::Fixed,
        Some("percent") => SpacingMode::Percent,
        Some("proportional") => SpacingMode::Proportional,
        _ => SpacingMode::Length,
    };
    let rotate_mode = match cm.get("rotateMode").and_then(Value::as_str) {
        Some("chain") => RotateMode::Chain,
        Some("chainScale") => RotateMode::ChainScale,
        _ => RotateMode::Tangent,
    };
    let mix_x = f_or(cm, "mixX", 1.0);
    Ok(PathConstraintData {
        name,
        order,
        bones,
        slot,
        position_mode,
        spacing_mode,
        rotate_mode,
        offset_rotation: f(cm, "rotation"),
        position: f(cm, "position"),
        spacing: f(cm, "spacing"),
        mix_rotate: f_or(cm, "mixRotate", 1.0),
        mix_x,
        mix_y: f_or(cm, "mixY", mix_x),
    })
}

/// Parse a `physics` constraint entry. `fps` becomes a fixed `step` of `1/fps`
/// and `mass` is stored inverted; the per-property "global" flags (used only by
/// physics timelines) are not read yet.
fn parse_physics(
    cm: &Value,
    order: usize,
    data: &SkeletonData,
) -> Result<PhysicsConstraintData, LoadError> {
    let name = str_field(cm, "name")?;
    let bone_name = str_field(cm, "bone")?;
    let bone = data
        .find_bone(&bone_name)
        .ok_or(LoadError::BadReference(bone_name))?;
    let scale_y_mode = match cm.get("scaleY").and_then(Value::as_str) {
        Some("uniform") => ScaleYMode::Uniform,
        Some("volume") => ScaleYMode::Volume,
        _ => ScaleYMode::None,
    };
    let fps = cm.get("fps").and_then(Value::as_f64).unwrap_or(60.0);
    let mass = f_or(cm, "mass", 1.0);
    Ok(PhysicsConstraintData {
        name,
        order,
        bone,
        x: f(cm, "x"),
        y: f(cm, "y"),
        rotate: f(cm, "rotate"),
        scale_x: f(cm, "scaleX"),
        shear_x: f(cm, "shearX"),
        limit: f_or(cm, "limit", 5000.0),
        step: 1.0 / (fps.max(1.0) as f32),
        scale_y_mode,
        inertia: f_or(cm, "inertia", 0.5),
        strength: f_or(cm, "strength", 100.0),
        damping: f_or(cm, "damping", 0.85),
        mass_inverse: 1.0 / if mass != 0.0 { mass } else { 1.0 },
        wind: f(cm, "wind"),
        gravity: f(cm, "gravity"),
        mix: f_or(cm, "mix", 1.0),
        inertia_global: bool_or(cm, "inertiaGlobal", false),
        strength_global: bool_or(cm, "strengthGlobal", false),
        damping_global: bool_or(cm, "dampingGlobal", false),
        mass_global: bool_or(cm, "massGlobal", false),
        wind_global: bool_or(cm, "windGlobal", false),
        gravity_global: bool_or(cm, "gravityGlobal", false),
        mix_global: bool_or(cm, "mixGlobal", false),
    })
}

fn from_prop(name: &str) -> Option<FromProp> {
    Some(match name {
        "rotate" => FromProp::Rotate,
        "x" => FromProp::X,
        "y" => FromProp::Y,
        "scaleX" => FromProp::ScaleX,
        "scaleY" => FromProp::ScaleY,
        "shearY" => FromProp::ShearY,
        _ => return None,
    })
}

fn to_prop(name: &str) -> Option<ToProp> {
    Some(match name {
        "rotate" => ToProp::Rotate,
        "x" => ToProp::X,
        "y" => ToProp::Y,
        "scaleX" => ToProp::ScaleX,
        "scaleY" => ToProp::ScaleY,
        "shearY" => ToProp::ShearY,
        _ => return None,
    })
}

fn bool_or(v: &Value, key: &str, default: bool) -> bool {
    v.get(key).and_then(Value::as_bool).unwrap_or(default)
}

fn f(v: &Value, key: &str) -> f32 {
    v.get(key).and_then(Value::as_f64).unwrap_or(0.0) as f32
}

fn f_or(v: &Value, key: &str, default: f32) -> f32 {
    v.get(key)
        .and_then(Value::as_f64)
        .map_or(default, |x| x as f32)
}

fn str_field(v: &Value, key: &str) -> Result<String, LoadError> {
    v.get(key)
        .and_then(Value::as_str)
        .map(String::from)
        .ok_or_else(|| LoadError::Schema(format!("missing string field '{key}'")))
}

fn f_array(v: &Value, key: &str) -> Vec<f32> {
    v.get(key)
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(Value::as_f64)
                .map(|x| x as f32)
                .collect()
        })
        .unwrap_or_default()
}

fn u16_array(v: &Value, key: &str) -> Vec<u16> {
    v.get(key)
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(Value::as_u64)
                .map(|x| x as u16)
                .collect()
        })
        .unwrap_or_default()
}

fn parse_inherit(v: Option<&Value>) -> Inherit {
    match v.and_then(Value::as_str).unwrap_or("normal") {
        "onlyTranslation" => Inherit::OnlyTranslation,
        "noRotationOrReflection" => Inherit::NoRotationOrReflection,
        "noScale" => Inherit::NoScale,
        "noScaleOrReflection" => Inherit::NoScaleOrReflection,
        _ => Inherit::Normal,
    }
}

fn parse_blend(s: Option<&str>) -> BlendMode {
    match s.unwrap_or("normal") {
        "additive" => BlendMode::Additive,
        "multiply" => BlendMode::Multiply,
        "screen" => BlendMode::Screen,
        _ => BlendMode::Normal,
    }
}

/// Parse a Spine hex color (`"rrggbbaa"` or `"rrggbb"`), defaulting on failure.
fn parse_color(s: Option<&str>, default: Color) -> Color {
    let Some(s) = s.map(|s| s.trim_start_matches('#')) else {
        return default;
    };
    if s.len() < 6 {
        return default;
    }
    let byte = |i: usize| {
        u8::from_str_radix(s.get(i..i + 2).unwrap_or("ff"), 16).unwrap_or(255) as f32 / 255.0
    };
    Color::new(
        byte(0),
        byte(2),
        byte(4),
        if s.len() >= 8 { byte(6) } else { 1.0 },
    )
}


mod timelines;
use timelines::parse_animation;

#[cfg(test)]
mod tests;
