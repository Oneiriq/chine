//! Loading skeleton data from Spine exports.
//!
//! Behind the `json` feature, [`from_json`] parses a Spine `.json` export into
//! a [`SkeletonData`] rig (bones, slots, skins, region/mesh attachments). The
//! `.skel` binary loader and the animation / constraint sections arrive in
//! later milestones; unknown sections are ignored.
//!
//! Region attachments are parsed with their transform but without UVs — those
//! are filled in once an [`crate::atlas::Atlas`] is bound (the UV/offset layout
//! depends on the packed region).

use std::error::Error;
use std::fmt;
use std::sync::Arc;

use glam::Vec2;
use serde_json::Value;

use crate::anim::{Animation, BoneTimeline, Timeline};
use crate::attach::{Attachment, MeshAttachment, MeshVertices, RegionAttachment};
use crate::constraint::ik::IkConstraintData;
use crate::constraint::ScaleYMode;
use crate::data::{BlendMode, BoneData, Color, Inherit, SkeletonData, SlotData};
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
                            if let Some(attachment) = parse_attachment(att_name, att) {
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
        for cm in constraints {
            if cm.get("type").and_then(Value::as_str) != Some("ik") {
                continue;
            }
            let name = str_field(cm, "name")?;
            let mut cbones = Vec::new();
            if let Some(bs) = cm.get("bones").and_then(Value::as_array) {
                for bn in bs {
                    let bn = bn
                        .as_str()
                        .ok_or_else(|| LoadError::Schema("ik bone name".into()))?;
                    cbones.push(
                        data.find_bone(bn)
                            .ok_or_else(|| LoadError::BadReference(bn.to_string()))?,
                    );
                }
            }
            let target_name = str_field(cm, "target")?;
            let target = data
                .find_bone(&target_name)
                .ok_or(LoadError::BadReference(target_name))?;
            let scale_y_mode = match cm.get("scaleY").and_then(Value::as_str) {
                Some("uniform") => ScaleYMode::Uniform,
                Some("volume") => ScaleYMode::Volume,
                _ => ScaleYMode::None,
            };
            data.ik_constraints.push(IkConstraintData {
                name,
                bones: cbones,
                target,
                scale_y_mode,
                mix: f_or(cm, "mix", 1.0),
                softness: f(cm, "softness"),
                bend_direction: if cm
                    .get("bendPositive")
                    .and_then(Value::as_bool)
                    .unwrap_or(true)
                {
                    1
                } else {
                    -1
                },
                compress: cm.get("compress").and_then(Value::as_bool).unwrap_or(false),
                stretch: cm.get("stretch").and_then(Value::as_bool).unwrap_or(false),
            });
        }
    }

    if let Some(anims) = root.get("animations").and_then(Value::as_object) {
        for (name, anim) in anims {
            let animation = parse_animation(name, anim, &data)?;
            data.animations.push(Arc::new(animation));
        }
    }

    Ok(data)
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
            Some(Attachment::Mesh(m))
        }
        // linkedmesh / boundingbox / clipping / path / point arrive later.
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

/// Parse one animation: bone rotate / translate / scale timelines. Other
/// channels (slot, deform, event, constraint timelines) arrive in later
/// milestones and are skipped.
fn parse_animation(name: &str, anim: &Value, data: &SkeletonData) -> Result<Animation, LoadError> {
    let mut timelines = Vec::new();
    let mut duration = 0.0_f32;
    if let Some(bones) = anim.get("bones").and_then(Value::as_object) {
        for (bone_name, props) in bones {
            let bone = data
                .find_bone(bone_name)
                .ok_or_else(|| LoadError::BadReference(bone_name.clone()))?;
            let Some(props) = props.as_object() else {
                continue;
            };
            for (prop, keys) in props {
                let Some(keys) = keys.as_array() else {
                    continue;
                };
                if keys.is_empty() {
                    continue;
                }
                let (timeline, dur) = match prop.as_str() {
                    "rotate" => {
                        let (tl, d) = read_timeline1(keys, bone, 0.0);
                        (Timeline::Rotate(tl), d)
                    }
                    "translate" => {
                        let (tl, d) = read_timeline2(keys, bone, "x", "y", 0.0);
                        (Timeline::Translate(tl), d)
                    }
                    "scale" => {
                        let (tl, d) = read_timeline2(keys, bone, "x", "y", 1.0);
                        (Timeline::Scale(tl), d)
                    }
                    _ => continue, // shear / x-y splits / constraints arrive later
                };
                duration = duration.max(dur);
                timelines.push(timeline);
            }
        }
    }
    Ok(Animation::new(name, duration, timelines))
}

/// Read a one-value (rotate) timeline. Returns the timeline and its last
/// keyframe time. Mirrors Spine's `readTimeline` for `CurveTimeline1`.
fn read_timeline1(keys: &[Value], bone: usize, default_value: f32) -> (BoneTimeline, f32) {
    let n = keys.len();
    let mut tl = BoneTimeline::one_value(bone, n, n);
    let mut bezier = 0;
    let mut duration = 0.0_f32;
    let mut frame = 0;
    while frame < n {
        let k = &keys[frame];
        let time = f(k, "time");
        let value = f_or(k, "value", default_value);
        tl.set_frame1(frame, time, value);
        duration = duration.max(time);
        if frame + 1 < n {
            if let Some(curve) = k.get("curve") {
                let next = &keys[frame + 1];
                let time2 = f(next, "time");
                let value2 = f_or(next, "value", default_value);
                bezier = read_curve(curve, &mut tl, bezier, frame, 0, time, time2, value, value2);
            }
        }
        frame += 1;
    }
    (tl, duration)
}

/// Read a two-value (translate / scale) timeline. Mirrors Spine's `readTimeline`
/// for `BoneTimeline2`.
fn read_timeline2(
    keys: &[Value],
    bone: usize,
    name1: &str,
    name2: &str,
    default_value: f32,
) -> (BoneTimeline, f32) {
    let n = keys.len();
    let mut tl = BoneTimeline::two_value(bone, n, n * 2);
    let mut bezier = 0;
    let mut duration = 0.0_f32;
    let mut frame = 0;
    while frame < n {
        let k = &keys[frame];
        let time = f(k, "time");
        let v1 = f_or(k, name1, default_value);
        let v2 = f_or(k, name2, default_value);
        tl.set_frame2(frame, time, v1, v2);
        duration = duration.max(time);
        if frame + 1 < n {
            if let Some(curve) = k.get("curve") {
                let next = &keys[frame + 1];
                let time2 = f(next, "time");
                let nv1 = f_or(next, name1, default_value);
                let nv2 = f_or(next, name2, default_value);
                bezier = read_curve(curve, &mut tl, bezier, frame, 0, time, time2, v1, nv1);
                bezier = read_curve(curve, &mut tl, bezier, frame, 1, time, time2, v2, nv2);
            }
        }
        frame += 1;
    }
    (tl, duration)
}

/// Apply one keyframe's `curve` field (absent = linear, `"stepped"`, or a Bezier
/// array; the array holds 4 floats per value channel at offset `value_ord * 4`).
/// Mirrors Spine's `readCurve`.
#[allow(clippy::too_many_arguments)]
fn read_curve(
    curve: &Value,
    tl: &mut BoneTimeline,
    bezier: usize,
    frame: usize,
    value_ord: usize,
    time1: f32,
    time2: f32,
    value1: f32,
    value2: f32,
) -> usize {
    if let Some(s) = curve.as_str() {
        if s == "stepped" {
            tl.set_stepped(frame);
        }
        return bezier;
    }
    let Some(arr) = curve.as_array() else {
        return bezier;
    };
    let base = value_ord * 4;
    let at = |i: usize| arr.get(base + i).and_then(Value::as_f64).unwrap_or(0.0) as f32;
    tl.set_bezier(
        bezier,
        frame,
        value_ord,
        time1,
        value1,
        at(0),
        at(1),
        at(2),
        at(3),
        time2,
        value2,
    );
    bezier + 1
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attach::Attachment;

    const JSON: &str = r#"{
        "skeleton": { "spine": "4.3.11", "x": -10, "y": 0, "width": 200, "height": 300 },
        "bones": [
            { "name": "root" },
            { "name": "torso", "parent": "root", "length": 100, "y": 20, "rotation": 90, "inherit": "noScale" }
        ],
        "slots": [
            { "name": "head", "bone": "torso", "attachment": "head", "color": "ff8800ff", "blend": "additive" }
        ],
        "skins": [
            {
                "name": "default",
                "attachments": {
                    "head": {
                        "head": { "x": 5, "y": 6, "width": 40, "height": 50, "rotation": 15 },
                        "head-mesh": {
                            "type": "mesh",
                            "uvs": [0,0, 1,0, 0,1],
                            "triangles": [0,1,2],
                            "vertices": [10,10, 20,10, 10,20],
                            "hull": 3
                        }
                    }
                }
            }
        ]
    }"#;

    #[test]
    fn parses_header_bones_slots() {
        let data = from_json(JSON).unwrap();
        assert_eq!(data.spine_version.as_deref(), Some("4.3.11"));
        assert_eq!(data.size, Vec2::new(200.0, 300.0));

        assert_eq!(data.bones.len(), 2);
        let torso = &data.bones[data.find_bone("torso").unwrap()];
        assert_eq!(torso.parent, Some(0));
        assert!((torso.position.y - 20.0).abs() < 1e-6);
        assert!((torso.rotation - 90.0).abs() < 1e-6);
        assert_eq!(torso.inherit, Inherit::NoScale);
        // defaults filled where omitted.
        assert!((torso.scale.x - 1.0).abs() < 1e-6);

        let head = &data.slots[data.find_slot("head").unwrap()];
        assert_eq!(head.bone, 1);
        assert_eq!(head.blend, BlendMode::Additive);
        assert!((head.color.r - 1.0).abs() < 1e-3 && (head.color.g - 0.533).abs() < 1e-2);
        assert_eq!(head.attachment.as_deref(), Some("head"));
    }

    #[test]
    fn parses_skin_region_and_mesh_attachments() {
        let data = from_json(JSON).unwrap();
        let slot = data.find_slot("head").unwrap();

        match data.attachment(slot, "head", None) {
            Some(Attachment::Region(r)) => {
                assert!((r.width - 40.0).abs() < 1e-6 && (r.rotation - 15.0).abs() < 1e-6);
                assert!((r.x - 5.0).abs() < 1e-6);
            }
            other => panic!("expected region, got {other:?}"),
        }

        match data.attachment(slot, "head-mesh", None) {
            Some(Attachment::Mesh(m)) => {
                assert_eq!(m.vertex_count(), 3);
                assert_eq!(m.triangles, vec![0, 1, 2]);
                assert_eq!(m.hull_length, 3);
            }
            other => panic!("expected mesh, got {other:?}"),
        }
    }

    #[test]
    fn unknown_bone_reference_errors() {
        let bad = r#"{ "bones": [ { "name": "a", "parent": "ghost" } ] }"#;
        assert!(matches!(from_json(bad), Err(LoadError::BadReference(_))));
    }

    #[test]
    fn parses_and_plays_a_rotate_animation() {
        let json = r#"{
            "bones": [ { "name": "root" }, { "name": "arm", "parent": "root" } ],
            "animations": {
                "wave": {
                    "bones": {
                        "arm": {
                            "rotate": [ { "time": 0, "value": 0 }, { "time": 1, "value": 90 } ]
                        }
                    }
                }
            }
        }"#;
        let data = from_json(json).unwrap();
        assert_eq!(data.animations.len(), 1);
        let anim = data.find_animation("wave").unwrap().clone();
        assert!((anim.duration() - 1.0).abs() < 1e-6);

        let arm = data.find_bone("arm").unwrap();
        let mut sk = crate::skel::Skeleton::new(std::sync::Arc::new(data));
        let mut state = crate::anim::AnimationState::new();
        state.set_animation(anim, false);
        state.update(0.5);
        state.apply(&mut sk);
        // setup rotation 0 + interpolated 45 at the halfway point.
        assert!((sk.bone(arm).unwrap().rotation - 45.0).abs() < 1e-4);
    }

    #[test]
    fn parses_ik_constraint() {
        let json = r#"{
            "bones": [
                { "name": "root" }, { "name": "thigh", "parent": "root" },
                { "name": "shin", "parent": "thigh" }, { "name": "goal", "parent": "root" }
            ],
            "constraints": [
                { "type": "ik", "name": "leg", "bones": ["thigh", "shin"],
                  "target": "goal", "bendPositive": false, "mix": 0.8 }
            ]
        }"#;
        let data = from_json(json).unwrap();
        assert_eq!(data.ik_constraints.len(), 1);
        let c = &data.ik_constraints[0];
        assert_eq!(c.bones, vec![1, 2]);
        assert_eq!(c.target, 3);
        assert_eq!(c.bend_direction, -1);
        assert!((c.mix - 0.8).abs() < 1e-6);
    }
}
