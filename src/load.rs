//! Loading skeleton data from Spine exports.
//!
//! Behind the `json` feature, [`from_json`] parses a Spine `.json` export into
//! a [`SkeletonData`] rig: bones, slots, skins, region / mesh / path
//! attachments, animations, and IK / transform / path / physics constraints.
//! The binary `.skel` loader arrives later; unknown sections are ignored.
//!
//! Region attachments are parsed with their transform but without UVs; those
//! are filled in once an [`crate::atlas::Atlas`] is bound (the UV/offset layout
//! depends on the packed region).

use std::error::Error;
use std::fmt;
use std::sync::Arc;

use glam::Vec2;
use serde_json::Value;

use crate::anim::{
    Animation, AttachmentTimeline, BoneAxis, BoneTimeline, ConstraintTimeline, DrawOrderTimeline,
    EventTimeline, PhysicsProperty, PhysicsResetTimeline, Timeline, GLOBAL_PHYSICS,
};
use crate::attach::{Attachment, MeshAttachment, MeshVertices, PathAttachment, RegionAttachment};
use crate::constraint::ik::IkConstraintData;
use crate::constraint::path::{PathConstraintData, PositionMode, RotateMode, SpacingMode};
use crate::constraint::physics::PhysicsConstraintData;
use crate::constraint::transform::{
    FromMapping, FromProp, ToMapping, ToProp, TransformConstraintData,
};
use crate::constraint::ScaleYMode;
use crate::data::{BlendMode, BoneData, Color, Inherit, SkeletonData, SlotData};
use crate::event::{Event, EventData};
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
        // linkedmesh / boundingbox / clipping / point arrive later.
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
                    "shear" => {
                        let (tl, d) = read_timeline2(keys, bone, "x", "y", 0.0);
                        (Timeline::Shear(tl), d)
                    }
                    "translatex" => {
                        let (tl, d) = read_timeline1(keys, bone, 0.0);
                        (Timeline::BoneAxis(tl, BoneAxis::TranslateX), d)
                    }
                    "translatey" => {
                        let (tl, d) = read_timeline1(keys, bone, 0.0);
                        (Timeline::BoneAxis(tl, BoneAxis::TranslateY), d)
                    }
                    "scalex" => {
                        let (tl, d) = read_timeline1(keys, bone, 1.0);
                        (Timeline::BoneAxis(tl, BoneAxis::ScaleX), d)
                    }
                    "scaley" => {
                        let (tl, d) = read_timeline1(keys, bone, 1.0);
                        (Timeline::BoneAxis(tl, BoneAxis::ScaleY), d)
                    }
                    "shearx" => {
                        let (tl, d) = read_timeline1(keys, bone, 0.0);
                        (Timeline::BoneAxis(tl, BoneAxis::ShearX), d)
                    }
                    "sheary" => {
                        let (tl, d) = read_timeline1(keys, bone, 0.0);
                        (Timeline::BoneAxis(tl, BoneAxis::ShearY), d)
                    }
                    _ => continue, // unknown bone channels are skipped
                };
                duration = duration.max(dur);
                timelines.push(timeline);
            }
        }
    }
    if let Some(iks) = anim.get("ik").and_then(Value::as_object) {
        for (cname, keys) in iks {
            let Some(keys) = keys.as_array() else {
                continue;
            };
            if keys.is_empty() {
                continue;
            }
            let idx = data
                .ik_constraints
                .iter()
                .position(|c| c.name == *cname)
                .ok_or_else(|| LoadError::BadReference(cname.clone()))?;
            let (tl, dur) = read_ik_timeline(keys, idx);
            duration = duration.max(dur);
            timelines.push(Timeline::Ik(tl));
        }
    }
    if let Some(tcs) = anim.get("transform").and_then(Value::as_object) {
        for (cname, keys) in tcs {
            let Some(keys) = keys.as_array() else {
                continue;
            };
            if keys.is_empty() {
                continue;
            }
            let idx = data
                .transform_constraints
                .iter()
                .position(|c| c.name == *cname)
                .ok_or_else(|| LoadError::BadReference(cname.clone()))?;
            let (tl, dur) = read_curve_timeline(
                keys,
                idx,
                &[
                    ("mixRotate", 1.0),
                    ("mixX", 1.0),
                    ("mixY", 1.0),
                    ("mixScaleX", 1.0),
                    ("mixScaleY", 1.0),
                    ("mixShearY", 1.0),
                ],
            );
            duration = duration.max(dur);
            timelines.push(Timeline::TransformMix(tl));
        }
    }
    if let Some(pcs) = anim.get("path").and_then(Value::as_object) {
        for (cname, channels) in pcs {
            let idx = data
                .path_constraints
                .iter()
                .position(|c| c.name == *cname)
                .ok_or_else(|| LoadError::BadReference(cname.clone()))?;
            let Some(channels) = channels.as_object() else {
                continue;
            };
            for (channel, keys) in channels {
                let Some(keys) = keys.as_array() else {
                    continue;
                };
                if keys.is_empty() {
                    continue;
                }
                match channel.as_str() {
                    "position" => {
                        let (tl, dur) = read_curve_timeline(keys, idx, &[("position", 0.0)]);
                        duration = duration.max(dur);
                        timelines.push(Timeline::PathPosition(tl));
                    }
                    "spacing" => {
                        let (tl, dur) = read_curve_timeline(keys, idx, &[("spacing", 0.0)]);
                        duration = duration.max(dur);
                        timelines.push(Timeline::PathSpacing(tl));
                    }
                    "mix" => {
                        let (tl, dur) = read_curve_timeline(
                            keys,
                            idx,
                            &[("mixRotate", 1.0), ("mixX", 1.0), ("mixY", 1.0)],
                        );
                        duration = duration.max(dur);
                        timelines.push(Timeline::PathMix(tl));
                    }
                    _ => {}
                }
            }
        }
    }

    if let Some(pcs) = anim.get("physics").and_then(Value::as_object) {
        for (cname, channels) in pcs {
            // An empty constraint name marks a global timeline (it drives every
            // physics constraint whose matching global flag is set). The "reset"
            // channel is still skipped below.
            let idx = if cname.is_empty() {
                GLOBAL_PHYSICS
            } else {
                data.physics_constraints
                    .iter()
                    .position(|c| c.name == *cname)
                    .ok_or_else(|| LoadError::BadReference(cname.clone()))?
            };
            let Some(channels) = channels.as_object() else {
                continue;
            };
            for (channel, keys) in channels {
                let Some(keys) = keys.as_array() else {
                    continue;
                };
                if keys.is_empty() {
                    continue;
                }
                if channel == "reset" {
                    let times: Vec<f32> = keys.iter().map(|k| f(k, "time")).collect();
                    duration = duration.max(times.last().copied().unwrap_or(0.0));
                    timelines.push(Timeline::PhysicsReset(PhysicsResetTimeline::new(
                        idx, times,
                    )));
                    continue;
                }
                let (property, default) = match channel.as_str() {
                    "inertia" => (PhysicsProperty::Inertia, 0.5),
                    "strength" => (PhysicsProperty::Strength, 100.0),
                    "damping" => (PhysicsProperty::Damping, 0.85),
                    "mass" => (PhysicsProperty::Mass, 1.0),
                    "wind" => (PhysicsProperty::Wind, 0.0),
                    "gravity" => (PhysicsProperty::Gravity, 0.0),
                    "mix" => (PhysicsProperty::Mix, 1.0),
                    _ => continue, // Unknown channels are skipped.
                };
                let (tl, dur) = read_curve_timeline(keys, idx, &[("value", default)]);
                duration = duration.max(dur);
                timelines.push(Timeline::Physics(tl, property));
            }
        }
    }

    if let Some(slots) = anim.get("slots").and_then(Value::as_object) {
        for (slot_name, channels) in slots {
            let idx = data
                .find_slot(slot_name)
                .ok_or_else(|| LoadError::BadReference(slot_name.clone()))?;
            let Some(channels) = channels.as_object() else {
                continue;
            };
            for (channel, keys) in channels {
                let Some(keys) = keys.as_array() else {
                    continue;
                };
                if keys.is_empty() {
                    continue;
                }
                match channel.as_str() {
                    "rgba" => {
                        let (tl, dur) = read_slot_rgba_timeline(keys, idx);
                        duration = duration.max(dur);
                        timelines.push(Timeline::SlotColor(tl, true));
                    }
                    "rgb" => {
                        let (tl, dur) = read_slot_rgb_timeline(keys, idx);
                        duration = duration.max(dur);
                        timelines.push(Timeline::SlotColor(tl, false));
                    }
                    "attachment" => {
                        let mut times = Vec::with_capacity(keys.len());
                        let mut names = Vec::with_capacity(keys.len());
                        for k in keys {
                            times.push(f(k, "time"));
                            names.push(k.get("name").and_then(Value::as_str).map(String::from));
                        }
                        duration = duration.max(times.last().copied().unwrap_or(0.0));
                        timelines.push(Timeline::Attachment(AttachmentTimeline::new(
                            idx, times, names,
                        )));
                    }
                    _ => {} // rgba2 / rgb2 (two-color) arrive later.
                }
            }
        }
    }

    if let Some(dos) = anim
        .get("drawOrder")
        .or_else(|| anim.get("draworder"))
        .and_then(Value::as_array)
    {
        let slot_count = data.slots.len();
        let mut times = Vec::with_capacity(dos.len());
        let mut orders = Vec::with_capacity(dos.len());
        for k in dos {
            times.push(f(k, "time"));
            orders.push(read_draw_order(k, slot_count, data));
        }
        if let Some(&dur) = times.last() {
            duration = duration.max(dur);
            timelines.push(Timeline::DrawOrder(DrawOrderTimeline::new(times, orders)));
        }
    }

    if let Some(events) = anim.get("events").and_then(Value::as_array) {
        if !events.is_empty() {
            let (tl, dur) = read_event_timeline(events, data);
            duration = duration.max(dur);
            timelines.push(Timeline::Event(tl));
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

/// A timeline whose curve interpolation the loader can configure (stepped or
/// Bezier), implemented by bone and constraint timelines so [`read_curve`] works
/// for both.
trait CurveBuilder {
    fn stepped(&mut self, frame: usize);
    #[allow(clippy::too_many_arguments)]
    fn bezier(
        &mut self,
        bezier: usize,
        frame: usize,
        value: usize,
        time1: f32,
        value1: f32,
        cx1: f32,
        cy1: f32,
        cx2: f32,
        cy2: f32,
        time2: f32,
        value2: f32,
    );
}

impl CurveBuilder for BoneTimeline {
    fn stepped(&mut self, frame: usize) {
        self.set_stepped(frame);
    }
    #[allow(clippy::too_many_arguments)]
    fn bezier(
        &mut self,
        bezier: usize,
        frame: usize,
        value: usize,
        time1: f32,
        value1: f32,
        cx1: f32,
        cy1: f32,
        cx2: f32,
        cy2: f32,
        time2: f32,
        value2: f32,
    ) {
        self.set_bezier(
            bezier, frame, value, time1, value1, cx1, cy1, cx2, cy2, time2, value2,
        );
    }
}

impl CurveBuilder for ConstraintTimeline {
    fn stepped(&mut self, frame: usize) {
        self.set_stepped(frame);
    }
    #[allow(clippy::too_many_arguments)]
    fn bezier(
        &mut self,
        bezier: usize,
        frame: usize,
        value: usize,
        time1: f32,
        value1: f32,
        cx1: f32,
        cy1: f32,
        cx2: f32,
        cy2: f32,
        time2: f32,
        value2: f32,
    ) {
        self.set_bezier(
            bezier, frame, value, time1, value1, cx1, cy1, cx2, cy2, time2, value2,
        );
    }
}

/// Apply one keyframe's `curve` field (absent = linear, `"stepped"`, or a Bezier
/// array; the array holds 4 floats per value channel at offset `value_ord * 4`).
/// Mirrors Spine's `readCurve`.
#[allow(clippy::too_many_arguments)]
fn read_curve(
    curve: &Value,
    tl: &mut impl CurveBuilder,
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
            tl.stepped(frame);
        }
        return bezier;
    }
    let Some(arr) = curve.as_array() else {
        return bezier;
    };
    let base = value_ord * 4;
    let at = |i: usize| arr.get(base + i).and_then(Value::as_f64).unwrap_or(0.0) as f32;
    tl.bezier(
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

/// Read a generic N-channel constraint timeline (all channels Bezier-curved).
/// `channels` is `(json field, default)` per channel.
fn read_curve_timeline(
    keys: &[Value],
    constraint: usize,
    channels: &[(&str, f32)],
) -> (ConstraintTimeline, f32) {
    let n = keys.len();
    let nc = channels.len();
    let mut tl = ConstraintTimeline::new(constraint, n, n * nc, nc + 1);
    let mut bezier = 0;
    let mut duration = 0.0_f32;
    let mut frame = 0;
    while frame < n {
        let k = &keys[frame];
        let time = f(k, "time");
        let values: Vec<f32> = channels
            .iter()
            .map(|(field, def)| f_or(k, field, *def))
            .collect();
        tl.set_frame(frame, time, &values);
        duration = duration.max(time);
        if frame + 1 < n {
            if let Some(curve) = k.get("curve") {
                let next = &keys[frame + 1];
                let time2 = f(next, "time");
                for (ci, (field, def)) in channels.iter().enumerate() {
                    let v2 = f_or(next, field, *def);
                    bezier = read_curve(
                        curve, &mut tl, bezier, frame, ci, time, time2, values[ci], v2,
                    );
                }
            }
        }
        frame += 1;
    }
    (tl, duration)
}

/// Read an IK constraint timeline: mix and softness are Bezier-curved; bend
/// direction, compress, and stretch are stored stepped.
fn read_ik_timeline(keys: &[Value], constraint: usize) -> (ConstraintTimeline, f32) {
    let n = keys.len();
    let mut tl = ConstraintTimeline::new(constraint, n, n * 2, 6);
    let mut bezier = 0;
    let mut duration = 0.0_f32;
    let mut frame = 0;
    while frame < n {
        let k = &keys[frame];
        let time = f(k, "time");
        let mix = f_or(k, "mix", 1.0);
        let softness = f(k, "softness");
        let bend = if bool_or(k, "bendPositive", true) {
            1.0
        } else {
            -1.0
        };
        let compress = f32::from(bool_or(k, "compress", false));
        let stretch = f32::from(bool_or(k, "stretch", false));
        tl.set_frame(frame, time, &[mix, softness, bend, compress, stretch]);
        duration = duration.max(time);
        if frame + 1 < n {
            if let Some(curve) = k.get("curve") {
                let next = &keys[frame + 1];
                let time2 = f(next, "time");
                let mix2 = f_or(next, "mix", 1.0);
                let soft2 = f(next, "softness");
                bezier = read_curve(curve, &mut tl, bezier, frame, 0, time, time2, mix, mix2);
                bezier = read_curve(
                    curve, &mut tl, bezier, frame, 1, time, time2, softness, soft2,
                );
            }
        }
        frame += 1;
    }
    (tl, duration)
}

/// Read a slot RGBA color timeline (four channels from each keyframe's `color`
/// hex string).
fn read_slot_rgba_timeline(keys: &[Value], slot: usize) -> (ConstraintTimeline, f32) {
    let n = keys.len();
    let mut tl = ConstraintTimeline::new(slot, n, n * 4, 5);
    let mut bezier = 0;
    let mut duration = 0.0_f32;
    let mut frame = 0;
    while frame < n {
        let k = &keys[frame];
        let time = f(k, "time");
        let c = parse_color(k.get("color").and_then(Value::as_str), Color::WHITE);
        tl.set_frame(frame, time, &[c.r, c.g, c.b, c.a]);
        duration = duration.max(time);
        if frame + 1 < n {
            if let Some(curve) = k.get("curve") {
                let next = &keys[frame + 1];
                let time2 = f(next, "time");
                let c2 = parse_color(next.get("color").and_then(Value::as_str), Color::WHITE);
                bezier = read_curve(curve, &mut tl, bezier, frame, 0, time, time2, c.r, c2.r);
                bezier = read_curve(curve, &mut tl, bezier, frame, 1, time, time2, c.g, c2.g);
                bezier = read_curve(curve, &mut tl, bezier, frame, 2, time, time2, c.b, c2.b);
                bezier = read_curve(curve, &mut tl, bezier, frame, 3, time, time2, c.a, c2.a);
            }
        }
        frame += 1;
    }
    (tl, duration)
}

/// Read a slot RGB color timeline (three channels from each keyframe's `color`
/// hex string; the slot's alpha is left unchanged).
fn read_slot_rgb_timeline(keys: &[Value], slot: usize) -> (ConstraintTimeline, f32) {
    let n = keys.len();
    let mut tl = ConstraintTimeline::new(slot, n, n * 3, 4);
    let mut bezier = 0;
    let mut duration = 0.0_f32;
    let mut frame = 0;
    while frame < n {
        let k = &keys[frame];
        let time = f(k, "time");
        let c = parse_color(k.get("color").and_then(Value::as_str), Color::WHITE);
        tl.set_frame(frame, time, &[c.r, c.g, c.b]);
        duration = duration.max(time);
        if frame + 1 < n {
            if let Some(curve) = k.get("curve") {
                let next = &keys[frame + 1];
                let time2 = f(next, "time");
                let c2 = parse_color(next.get("color").and_then(Value::as_str), Color::WHITE);
                bezier = read_curve(curve, &mut tl, bezier, frame, 0, time, time2, c.r, c2.r);
                bezier = read_curve(curve, &mut tl, bezier, frame, 1, time, time2, c.g, c2.g);
                bezier = read_curve(curve, &mut tl, bezier, frame, 2, time, time2, c.b, c2.b);
            }
        }
        frame += 1;
    }
    (tl, duration)
}

/// Read one draw-order keyframe's `offsets` into a full slot ordering (the setup
/// order when there are no offsets).
fn read_draw_order(k: &Value, slot_count: usize, data: &SkeletonData) -> Vec<usize> {
    let mut offsets: Vec<(usize, i32)> = Vec::new();
    if let Some(offs) = k.get("offsets").and_then(Value::as_array) {
        for o in offs {
            let Some(slot_name) = o.get("slot").and_then(Value::as_str) else {
                continue;
            };
            let Some(slot_index) = data.find_slot(slot_name) else {
                continue;
            };
            let offset = o.get("offset").and_then(Value::as_i64).unwrap_or(0) as i32;
            offsets.push((slot_index, offset));
        }
    }
    if offsets.is_empty() {
        return (0..slot_count).collect();
    }
    compute_draw_order(slot_count, &mut offsets)
}

/// Compute a draw order from slot offsets, mirroring Spine: each listed slot
/// moves by its offset; the rest keep their relative order.
fn compute_draw_order(slot_count: usize, offsets: &mut [(usize, i32)]) -> Vec<usize> {
    offsets.sort_by_key(|(slot, _)| *slot);
    let mut draw_order = vec![usize::MAX; slot_count];
    let mut unchanged = vec![0usize; slot_count.saturating_sub(offsets.len())];
    let mut original_index = 0;
    let mut unchanged_index = 0;
    for &(slot_index, offset) in offsets.iter() {
        while original_index != slot_index && original_index < slot_count {
            unchanged[unchanged_index] = original_index;
            unchanged_index += 1;
            original_index += 1;
        }
        let pos = original_index as i32 + offset;
        if (0..slot_count as i32).contains(&pos) {
            draw_order[pos as usize] = original_index;
        }
        original_index += 1;
    }
    while original_index < slot_count {
        unchanged[unchanged_index] = original_index;
        unchanged_index += 1;
        original_index += 1;
    }
    for i in (0..slot_count).rev() {
        if draw_order[i] == usize::MAX && unchanged_index > 0 {
            unchanged_index -= 1;
            draw_order[i] = unchanged[unchanged_index];
        }
    }
    draw_order
}

/// Read an animation event timeline: each keyframe's event, resolving its value
/// overrides against the named event's setup defaults.
fn read_event_timeline(keys: &[Value], data: &SkeletonData) -> (EventTimeline, f32) {
    let mut times = Vec::with_capacity(keys.len());
    let mut events = Vec::with_capacity(keys.len());
    let mut duration = 0.0_f32;
    for k in keys {
        let time = f(k, "time");
        let name = k.get("name").and_then(Value::as_str).unwrap_or("");
        let setup = data.events.iter().find(|e| e.name == name);
        let (di, df, ds, dv, db) = match setup {
            Some(e) => (
                e.int_value,
                e.float_value,
                e.string_value.clone(),
                e.volume,
                e.balance,
            ),
            None => (0, 0.0, String::new(), 1.0, 0.0),
        };
        events.push(Event {
            name: name.to_string(),
            time,
            int_value: k
                .get("int")
                .and_then(Value::as_i64)
                .map_or(di, |v| v as i32),
            float_value: f_or(k, "float", df),
            string_value: k
                .get("string")
                .and_then(Value::as_str)
                .map_or(ds, String::from),
            volume: f_or(k, "volume", dv),
            balance: f_or(k, "balance", db),
        });
        times.push(time);
        duration = duration.max(time);
    }
    (EventTimeline::new(times, events), duration)
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

    #[test]
    fn parses_transform_constraint() {
        let json = r#"{
            "bones": [
                { "name": "root" }, { "name": "src", "parent": "root" },
                { "name": "dst", "parent": "root" }
            ],
            "constraints": [
                { "type": "transform", "name": "follow", "bones": ["dst"], "source": "src",
                  "mixRotate": 0.5, "properties": { "rotate": { "to": { "rotate": {} } } } }
            ]
        }"#;
        let data = from_json(json).unwrap();
        assert_eq!(data.transform_constraints.len(), 1);
        let t = &data.transform_constraints[0];
        assert_eq!(t.bones, vec![2]);
        assert_eq!(t.source, 1);
        assert_eq!(t.order, 0);
        assert!((t.mix_rotate - 0.5).abs() < 1e-6);
        assert_eq!(t.properties.len(), 1);
        assert_eq!(t.properties[0].to.len(), 1);
    }

    #[test]
    fn parses_physics_constraint() {
        let json = r#"{
            "skeleton": { "referenceScale": 120 },
            "bones": [ { "name": "root" }, { "name": "tail", "parent": "root", "length": 30 } ],
            "constraints": [
                { "type": "physics", "name": "jiggle", "bone": "tail",
                  "y": 1, "rotate": 1, "gravity": 2, "strength": 80, "damping": 0.9,
                  "mass": 2, "fps": 120, "mix": 1 }
            ]
        }"#;
        let data = from_json(json).unwrap();
        assert!((data.reference_scale - 120.0).abs() < 1e-6);
        assert_eq!(data.physics_constraints.len(), 1);
        let p = &data.physics_constraints[0];
        assert_eq!(p.bone, 1);
        assert_eq!(p.order, 0);
        assert!((p.gravity - 2.0).abs() < 1e-6);
        assert!((p.strength - 80.0).abs() < 1e-6);
        // fps 120 -> step 1/120.
        assert!((p.step - 1.0 / 120.0).abs() < 1e-6);
        // mass 2 -> mass_inverse 0.5.
        assert!((p.mass_inverse - 0.5).abs() < 1e-6);
        // omitted inertia defaults to 0.5.
        assert!((p.inertia - 0.5).abs() < 1e-6);
    }

    #[test]
    fn physics_strength_timeline_animates_the_constraint() {
        let json = r#"{
            "bones": [ { "name": "root" }, { "name": "tail", "parent": "root", "length": 20 } ],
            "constraints": [
                { "type": "physics", "name": "jiggle", "bone": "tail", "y": 1, "strength": 100 }
            ],
            "animations": {
                "soften": {
                    "physics": {
                        "jiggle": {
                            "strength": [ { "time": 0, "value": 100 }, { "time": 1, "value": 20 } ]
                        }
                    }
                }
            }
        }"#;
        let data = from_json(json).unwrap();
        let anim = data.find_animation("soften").unwrap().clone();
        let mut sk = crate::skel::Skeleton::new(std::sync::Arc::new(data));
        let mut state = crate::anim::AnimationState::new();
        state.set_animation(anim, false);
        // Halfway through, strength interpolates 100 -> 20, i.e. 60.
        state.update(0.5);
        state.apply(&mut sk);
        let s = sk.physics_constraint(0).unwrap().strength;
        assert!((s - 60.0).abs() < 1e-3, "strength={s}");
    }

    #[test]
    fn global_physics_timeline_drives_flagged_constraints() {
        let json = r#"{
            "bones": [
                { "name": "root" },
                { "name": "a", "parent": "root", "length": 10 },
                { "name": "b", "parent": "root", "length": 10 }
            ],
            "constraints": [
                { "type": "physics", "name": "pa", "bone": "a", "y": 1, "strength": 100, "strengthGlobal": true },
                { "type": "physics", "name": "pb", "bone": "b", "y": 1, "strength": 100 }
            ],
            "animations": {
                "soften": {
                    "physics": {
                        "": { "strength": [ { "time": 0, "value": 40 }, { "time": 1, "value": 40 } ] }
                    }
                }
            }
        }"#;
        let data = from_json(json).unwrap();
        assert_eq!(data.physics_constraints.len(), 2);
        assert!(data.physics_constraints[0].strength_global);
        assert!(!data.physics_constraints[1].strength_global);
        let anim = data.find_animation("soften").unwrap().clone();
        let mut sk = crate::skel::Skeleton::new(std::sync::Arc::new(data));
        let mut state = crate::anim::AnimationState::new();
        state.set_animation(anim, false);
        state.update(0.5);
        state.apply(&mut sk);
        // The global strength timeline (value 40) drives only "pa"
        // (strengthGlobal); "pb" keeps its setup strength (100).
        assert!((sk.physics_constraint(0).unwrap().strength - 40.0).abs() < 1e-3);
        assert!((sk.physics_constraint(1).unwrap().strength - 100.0).abs() < 1e-3);
    }

    #[test]
    fn physics_reset_timeline_zeroes_the_offset() {
        let json = r#"{
            "skeleton": { "referenceScale": 100 },
            "bones": [ { "name": "root" }, { "name": "tail", "parent": "root", "length": 20 } ],
            "constraints": [
                { "type": "physics", "name": "p", "bone": "tail",
                  "y": 1, "gravity": 1, "strength": 50, "damping": 0.9, "mass": 1, "fps": 60 }
            ],
            "animations": {
                "blink": { "physics": { "p": { "reset": [ { "time": 0.5 } ] } } }
            }
        }"#;
        let data = from_json(json).unwrap();
        let anim = data.find_animation("blink").unwrap().clone();
        let mut sk = crate::skel::Skeleton::new(std::sync::Arc::new(data));
        let mut state = crate::anim::AnimationState::new();
        state.set_animation(anim, false);
        let dt = 1.0 / 60.0;
        let step = |state: &mut crate::anim::AnimationState, sk: &mut crate::skel::Skeleton| {
            state.update(dt);
            sk.update(dt);
            sk.set_bones_to_setup_pose();
            state.apply(sk);
            sk.update_world_transform();
        };
        // Droop for ~0.48s, before the reset keyframe at t=0.5.
        for _ in 0..29 {
            step(&mut state, &mut sk);
        }
        let drooped = sk.bone(1).unwrap().world_y();
        assert!(drooped < -0.2, "expected droop before reset, got {drooped}");
        // Two more frames cross t=0.5 and fire the reset, undoing the droop.
        step(&mut state, &mut sk);
        step(&mut state, &mut sk);
        let after = sk.bone(1).unwrap().world_y();
        assert!(
            after.abs() < 0.1,
            "expected reset to undo the droop, got {after}"
        );
    }

    #[test]
    fn slot_color_timeline_animates_the_tint() {
        let json = r#"{
            "bones": [ { "name": "root" } ],
            "slots": [ { "name": "s", "bone": "root", "attachment": "a", "color": "ffffffff" } ],
            "animations": {
                "fade": {
                    "slots": {
                        "s": { "rgba": [ { "time": 0, "color": "ffffffff" }, { "time": 1, "color": "ff000000" } ] }
                    }
                }
            }
        }"#;
        let data = from_json(json).unwrap();
        let anim = data.find_animation("fade").unwrap().clone();
        let mut sk = crate::skel::Skeleton::new(std::sync::Arc::new(data));
        let mut state = crate::anim::AnimationState::new();
        state.set_animation(anim, false);
        // Halfway, white (1,1,1,1) -> opaque-red-faded (1,0,0,0): (1, 0.5, 0.5, 0.5).
        state.update(0.5);
        sk.set_slots_to_setup_pose();
        state.apply(&mut sk);
        let c = sk.slot(0).unwrap().color;
        assert!((c.r - 1.0).abs() < 1e-2, "r={}", c.r);
        assert!((c.g - 0.5).abs() < 1e-2, "g={}", c.g);
        assert!((c.a - 0.5).abs() < 1e-2, "a={}", c.a);
    }

    #[test]
    fn slot_rgb_timeline_preserves_alpha() {
        let json = r#"{
            "bones": [ { "name": "root" } ],
            "slots": [ { "name": "s", "bone": "root", "attachment": "a", "color": "ffffff80" } ],
            "animations": {
                "tint": {
                    "slots": { "s": { "rgb": [ { "time": 0, "color": "ffffff" }, { "time": 1, "color": "ff0000" } ] } }
                }
            }
        }"#;
        let data = from_json(json).unwrap();
        let anim = data.find_animation("tint").unwrap().clone();
        let mut sk = crate::skel::Skeleton::new(std::sync::Arc::new(data));
        let mut state = crate::anim::AnimationState::new();
        state.set_animation(anim, false);
        state.update(0.5);
        sk.set_slots_to_setup_pose();
        state.apply(&mut sk);
        let c = sk.slot(0).unwrap().color;
        // rgb interpolates white -> red (g 0.5); alpha stays the setup 0x80 (~0.502).
        assert!((c.g - 0.5).abs() < 1e-2, "g={}", c.g);
        assert!((c.a - 0.502).abs() < 1e-2, "a={}", c.a);
    }

    #[test]
    fn event_timeline_fires_events() {
        let json = r#"{
            "bones": [ { "name": "root" } ],
            "events": { "footstep": { "int": 5 } },
            "animations": {
                "walk": { "events": [ { "time": 0.5, "name": "footstep", "int": 7 } ] }
            }
        }"#;
        let data = from_json(json).unwrap();
        assert_eq!(data.events.len(), 1);
        let anim = data.find_animation("walk").unwrap().clone();
        let mut sk = crate::skel::Skeleton::new(std::sync::Arc::new(data));
        let mut state = crate::anim::AnimationState::new();
        state.set_animation(anim, false);
        // Before the keyframe at 0.5, nothing fires.
        state.update(0.3);
        state.apply(&mut sk);
        assert!(sk.events().is_empty());
        // Crossing 0.5 fires the footstep with the keyframe int override (7).
        state.update(0.3);
        state.apply(&mut sk);
        assert_eq!(sk.events().len(), 1);
        assert_eq!(sk.events()[0].name, "footstep");
        assert_eq!(sk.events()[0].int_value, 7);
        // The next apply clears it.
        state.update(0.3);
        state.apply(&mut sk);
        assert!(sk.events().is_empty());
    }

    #[test]
    fn single_axis_bone_timelines() {
        let json = r#"{
            "bones": [ { "name": "b" } ],
            "animations": {
                "anim": {
                    "bones": {
                        "b": {
                            "translatex": [ { "time": 0, "value": 0 }, { "time": 1, "value": 10 } ],
                            "scaley": [ { "time": 0, "value": 1 }, { "time": 1, "value": 3 } ],
                            "shearx": [ { "time": 0, "value": 0 }, { "time": 1, "value": 20 } ]
                        }
                    }
                }
            }
        }"#;
        let data = from_json(json).unwrap();
        let anim = data.find_animation("anim").unwrap().clone();
        let mut sk = crate::skel::Skeleton::new(std::sync::Arc::new(data));
        let mut state = crate::anim::AnimationState::new();
        state.set_animation(anim, false);
        state.update(0.5);
        sk.set_bones_to_setup_pose();
        state.apply(&mut sk);
        let b = sk.bone(0).unwrap();
        // translatex 0->10 at half = 5; scaley 1->3 = 2; shearx 0->20 = 10.
        assert!((b.x - 5.0).abs() < 1e-3, "x={}", b.x);
        assert!((b.scale_y - 2.0).abs() < 1e-3, "scale_y={}", b.scale_y);
        assert!((b.shear_x - 10.0).abs() < 1e-3, "shear_x={}", b.shear_x);
    }

    #[test]
    fn shear_bone_timeline() {
        let json = r#"{
            "bones": [ { "name": "b" } ],
            "animations": {
                "anim": {
                    "bones": {
                        "b": { "shear": [ { "time": 0, "x": 0, "y": 0 }, { "time": 1, "x": 30, "y": 40 } ] }
                    }
                }
            }
        }"#;
        let data = from_json(json).unwrap();
        let anim = data.find_animation("anim").unwrap().clone();
        let mut sk = crate::skel::Skeleton::new(std::sync::Arc::new(data));
        let mut state = crate::anim::AnimationState::new();
        state.set_animation(anim, false);
        state.update(0.5);
        sk.set_bones_to_setup_pose();
        state.apply(&mut sk);
        let b = sk.bone(0).unwrap();
        assert!((b.shear_x - 15.0).abs() < 1e-3, "shear_x={}", b.shear_x);
        assert!((b.shear_y - 20.0).abs() < 1e-3, "shear_y={}", b.shear_y);
    }

    #[test]
    fn attachment_timeline_swaps_the_attachment() {
        let json = r#"{
            "bones": [ { "name": "root" } ],
            "slots": [ { "name": "s", "bone": "root", "attachment": "a" } ],
            "animations": {
                "swap": {
                    "slots": {
                        "s": { "attachment": [ { "time": 0, "name": "a" }, { "time": 1, "name": "b" } ] }
                    }
                }
            }
        }"#;
        let data = from_json(json).unwrap();
        let anim = data.find_animation("swap").unwrap().clone();
        let mut sk = crate::skel::Skeleton::new(std::sync::Arc::new(data));
        let mut state = crate::anim::AnimationState::new();
        state.set_animation(anim, false);
        // t=0.5 is before the second key, so the attachment is still "a" (stepped).
        state.update(0.5);
        sk.set_slots_to_setup_pose();
        state.apply(&mut sk);
        assert_eq!(sk.slot(0).unwrap().attachment.as_deref(), Some("a"));
        // Past t=1.0 the attachment switches to "b".
        state.update(0.6);
        sk.set_slots_to_setup_pose();
        state.apply(&mut sk);
        assert_eq!(sk.slot(0).unwrap().attachment.as_deref(), Some("b"));
    }

    #[test]
    fn draw_order_timeline_reorders_slots() {
        let json = r#"{
            "bones": [ { "name": "root" } ],
            "slots": [
                { "name": "s0", "bone": "root" },
                { "name": "s1", "bone": "root" },
                { "name": "s2", "bone": "root" }
            ],
            "animations": {
                "reorder": {
                    "drawOrder": [
                        { "time": 0 },
                        { "time": 1, "offsets": [ { "slot": "s0", "offset": 2 } ] }
                    ]
                }
            }
        }"#;
        let data = from_json(json).unwrap();
        let anim = data.find_animation("reorder").unwrap().clone();
        let mut sk = crate::skel::Skeleton::new(std::sync::Arc::new(data));
        let mut state = crate::anim::AnimationState::new();
        state.set_animation(anim, false);
        // t=0: identity draw order.
        state.update(0.0);
        sk.set_slots_to_setup_pose();
        state.apply(&mut sk);
        assert_eq!(sk.draw_order(), &[0, 1, 2]);
        // t>=1: slot 0 moves two places back, others shift forward.
        state.update(1.0);
        sk.set_slots_to_setup_pose();
        state.apply(&mut sk);
        assert_eq!(sk.draw_order(), &[1, 2, 0]);
    }

    #[test]
    fn ik_mix_timeline_fades_the_constraint() {
        let json = r#"{
            "bones": [
                { "name": "root" }, { "name": "aim", "parent": "root", "length": 10 },
                { "name": "target", "parent": "root", "y": 10 }
            ],
            "constraints": [
                { "type": "ik", "name": "aim-ik", "bones": ["aim"], "target": "target", "mix": 1 }
            ],
            "animations": {
                "fade": { "ik": { "aim-ik": [ { "time": 0, "mix": 0 }, { "time": 1, "mix": 1 } ] } }
            }
        }"#;
        let data = from_json(json).unwrap();
        let aim = data.find_bone("aim").unwrap();
        let anim = data.find_animation("fade").unwrap().clone();
        let mut sk = crate::skel::Skeleton::new(std::sync::Arc::new(data));
        let mut state = crate::anim::AnimationState::new();
        state.set_animation(anim, false);

        // time 0: IK mix 0 -> the constraint does nothing, aim keeps +x (a~1).
        state.apply(&mut sk);
        sk.update_world_transform();
        assert!(
            (sk.bone(aim).unwrap().a() - 1.0).abs() < 1e-2,
            "a={}",
            sk.bone(aim).unwrap().a()
        );

        // time 1: IK mix 1 -> aim is driven to point at the target (+y): a~0, c~1.
        state.update(1.0);
        sk.set_bones_to_setup_pose();
        state.apply(&mut sk);
        sk.update_world_transform();
        let aim_bone = sk.bone(aim).unwrap();
        assert!(aim_bone.a().abs() < 1e-2, "a={}", aim_bone.a());
        assert!((aim_bone.c() - 1.0).abs() < 1e-2, "c={}", aim_bone.c());
    }
}
