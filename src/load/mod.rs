//! Loading skeleton data from Spine exports.
//!
//! Behind the `json` feature, [`from_json`] parses a Spine `.json` export into
//! a [`SkeletonData`] rig: bones, slots, skins, region / mesh / path
//! attachments, animations, and IK / transform / path / physics / slider
//! constraints. (The binary `.skel` loader lives in the `binary` module.)
//!
//! Region attachments are parsed with their transform but without UVs. Those
//! are filled in once an [`crate::atlas::Atlas`] is bound (the UV/offset layout
//! depends on the packed region).

use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::sync::Arc;

use glam::Vec2;
use serde_json::Value;

use crate::attach::Attachment;
use crate::constraint::ik::IkConstraintData;
use crate::constraint::path::{PathConstraintData, PositionMode, RotateMode, SpacingMode};
use crate::constraint::physics::PhysicsConstraintData;
use crate::constraint::slider::{SliderData, SliderProperty};
use crate::constraint::transform::{
    FromMapping, FromProp, ToMapping, ToProp, TransformConstraintData,
};
use crate::constraint::ScaleYMode;
use crate::data::{BlendMode, BoneData, Color, Inherit, SkeletonData, SlotData};
use crate::event::EventData;
use crate::skin::{Skin, SkinConstraint};

mod attachments;
mod timelines;

use attachments::{attachment_cost, linked_copy_cost, read_attachment};
use timelines::parse_animation;

/// Bytes of derived data any load may expand to, whatever the input size.
const BUDGET_BASE: usize = 64 << 20;
/// Further bytes of derived data allowed per byte of JSON text.
const BUDGET_PER_BYTE: usize = 256;
/// The largest physics `fps` a Spine export can hold.
const MAX_PHYSICS_FPS: f64 = 255.0;

/// A cap on the data the loader derives from compact records. Some records
/// are cheap to write but expand to a copy of other data: a deform key to a
/// full vertex array, a draw order key to a full slot list, an event key to
/// its setup string, a linked mesh to its source's geometry, and a sequence
/// to per-frame UVs once an atlas is bound. The cap grows with the input
/// size, so it bounds memory without limiting real exports.
struct Budget {
    remaining: usize,
}

impl Budget {
    fn new(input_len: usize) -> Self {
        Self {
            remaining: BUDGET_BASE.saturating_add(input_len.saturating_mul(BUDGET_PER_BYTE)),
        }
    }

    /// Spend `bytes` on `what`, or fail if the cap is reached.
    fn charge(&mut self, bytes: usize, what: &str) -> Result<(), LoadError> {
        self.remaining = self
            .remaining
            .checked_sub(bytes)
            .ok_or_else(|| LoadError::Schema(format!("{what} expands past the load limit")))?;
        Ok(())
    }
}

/// Name-to-index maps for the bones, slots, constraints, skins, and events
/// that records refer to by name. Almost every record resolves a name, so
/// hash maps keep loading linear in the input where a linear search per
/// record would make it quadratic. Each map keeps the first entry with a
/// name, as the linear `SkeletonData::find_*` searches do.
#[derive(Default)]
struct Names<'a> {
    bones: HashMap<&'a str, usize>,
    slots: HashMap<&'a str, usize>,
    ik: HashMap<&'a str, usize>,
    transform: HashMap<&'a str, usize>,
    path: HashMap<&'a str, usize>,
    physics: HashMap<&'a str, usize>,
    sliders: HashMap<&'a str, usize>,
    /// Indices into `SkeletonData::skins`, which excludes the default skin.
    skins: HashMap<&'a str, usize>,
    events: HashMap<&'a str, usize>,
}

impl Names<'_> {
    /// The index of the bone `name`, or a [`LoadError::BadReference`].
    fn bone(&self, name: &str) -> Result<usize, LoadError> {
        lookup(&self.bones, name)
    }

    /// The index of the slot `name`, or a [`LoadError::BadReference`].
    fn slot(&self, name: &str) -> Result<usize, LoadError> {
        lookup(&self.slots, name)
    }

    /// The named (non-default) skin `name` in `data`.
    fn skin<'d>(&self, data: &'d SkeletonData, name: &str) -> Option<&'d Skin> {
        self.skins.get(name).and_then(|&i| data.skins.get(i))
    }
}

fn lookup(map: &HashMap<&str, usize>, name: &str) -> Result<usize, LoadError> {
    map.get(name)
        .copied()
        .ok_or_else(|| LoadError::BadReference(name.to_string()))
}

/// Record `name` at `index` unless an earlier entry has the same name.
fn add_name<'a>(map: &mut HashMap<&'a str, usize>, name: &'a str, index: usize) {
    map.entry(name).or_insert(index);
}

/// Record the `name` field of `v` at `index`, if it has one.
fn add_named<'a>(map: &mut HashMap<&'a str, usize>, v: &'a Value, index: usize) {
    if let Some(name) = v.get("name").and_then(Value::as_str) {
        add_name(map, name, index);
    }
}

/// An error parsing a Spine export.
#[derive(Debug)]
pub enum LoadError {
    /// The JSON text was malformed.
    Json(serde_json::Error),
    /// A referenced bone, slot, or constraint name (or a weighted vertex's
    /// bone index) was not found.
    BadReference(String),
    /// A required field was missing or had the wrong type, the data was
    /// inconsistent, or it expanded past the load limit.
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
/// a bone/slot reference can't be resolved, attachment geometry or draw order
/// data is inconsistent, or the export expands to far more data than its size
/// accounts for.
pub fn from_json(text: &str) -> Result<SkeletonData, LoadError> {
    let root: Value = serde_json::from_str(text).map_err(LoadError::Json)?;
    let mut data = SkeletonData::default();
    let mut names = Names::default();
    let mut budget = Budget::new(text.len());

    if let Some(skel) = root.get("skeleton") {
        data.spine_version = skel.get("spine").and_then(Value::as_str).map(String::from);
        data.position = Vec2::new(f(skel, "x"), f(skel, "y"));
        data.size = Vec2::new(f(skel, "width"), f(skel, "height"));
        data.reference_scale = f_or(skel, "referenceScale", 100.0);
    }

    if let Some(bones) = root.get("bones").and_then(Value::as_array) {
        for (index, b) in bones.iter().enumerate() {
            let name = str_ref(b, "name")?;
            // Only bones already read are named, so a parent always precedes
            // its children.
            let parent = b
                .get("parent")
                .and_then(Value::as_str)
                .map(|p| names.bone(p))
                .transpose()?;
            add_name(&mut names.bones, name, index);
            data.bones.push(BoneData {
                index,
                name: name.to_string(),
                parent,
                length: f(b, "length"),
                position: Vec2::new(f(b, "x"), f(b, "y")),
                rotation: f(b, "rotation"),
                scale: Vec2::new(f_or(b, "scaleX", 1.0), f_or(b, "scaleY", 1.0)),
                shear: Vec2::new(f(b, "shearX"), f(b, "shearY")),
                inherit: parse_inherit(b.get("inherit").or(b.get("transform"))),
                skin_required: bool_or(b, "skin", false),
            });
        }
    }

    if let Some(slots) = root.get("slots").and_then(Value::as_array) {
        for (index, s) in slots.iter().enumerate() {
            let name = str_ref(s, "name")?;
            let bone = names.bone(str_ref(s, "bone")?)?;
            add_name(&mut names.slots, name, index);
            data.slots.push(SlotData {
                index,
                name: name.to_string(),
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

    if let Some(constraints) = root.get("constraints").and_then(Value::as_array) {
        for (order, cm) in constraints.iter().enumerate() {
            match cm.get("type").and_then(Value::as_str) {
                Some("ik") => {
                    let c = parse_ik(cm, order, &names)?;
                    add_named(&mut names.ik, cm, data.ik_constraints.len());
                    data.ik_constraints.push(c);
                }
                Some("transform") => {
                    let c = parse_transform(cm, order, &names)?;
                    add_named(&mut names.transform, cm, data.transform_constraints.len());
                    data.transform_constraints.push(c);
                }
                Some("path") => {
                    let c = parse_path(cm, order, &names)?;
                    add_named(&mut names.path, cm, data.path_constraints.len());
                    data.path_constraints.push(c);
                }
                Some("physics") => {
                    let c = parse_physics(cm, order, &names)?;
                    add_named(&mut names.physics, cm, data.physics_constraints.len());
                    data.physics_constraints.push(c);
                }
                Some("slider") => {
                    let c = parse_slider(cm, order, &names)?;
                    add_named(&mut names.sliders, cm, data.sliders.len());
                    data.sliders.push(c);
                }
                _ => {}
            }
        }
    }

    if let Some(skins) = root.get("skins").and_then(Value::as_array) {
        for sk in skins {
            let skin_name = sk.get("name").and_then(Value::as_str).unwrap_or("default");
            let mut skin = Skin::new(skin_name);
            read_skin_requirements(sk, &names, &mut skin)?;
            if let Some(slot_map) = sk.get("attachments").and_then(Value::as_object) {
                for (slot_name, atts) in slot_map {
                    let slot = names.slot(slot_name)?;
                    if let Some(atts) = atts.as_object() {
                        for (att_name, att) in atts {
                            let Some(mut attachment) =
                                read_attachment(att_name, att, data.bones.len())?
                            else {
                                continue;
                            };
                            budget
                                .charge(attachment_cost(&attachment), "an attachment sequence")?;
                            // A linked mesh names its source's slot, if it is
                            // not the link's own.
                            if let Attachment::LinkedMesh(link) = &mut attachment {
                                if let Some(source) = att.get("slot").and_then(Value::as_str) {
                                    link.source_slot = Some(names.slot(source)?);
                                }
                            }
                            skin.set(slot, att_name.clone(), attachment);
                        }
                    }
                }
            }
            if skin_name == "default" {
                data.default_skin = skin;
            } else {
                add_name(&mut names.skins, skin_name, data.skins.len());
                data.skins.push(skin);
            }
        }
    }

    if let Some(events) = root.get("events").and_then(Value::as_object) {
        for (name, e) in events {
            add_name(&mut names.events, name, data.events.len());
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
    // resolved (source-shared) geometry rather than unresolved links.
    charge_linked_meshes(&data, &mut budget)?;
    crate::link::resolve_linked_meshes(&mut data);

    if let Some(anims) = root.get("animations").and_then(Value::as_object) {
        for (name, anim) in anims {
            let animation = parse_animation(name, anim, &data, &names, &mut budget)?;
            data.animations.push(Arc::new(animation));
        }
    }
    resolve_slider_animations(&root, &mut data)?;
    Ok(data)
}

/// Read the skin-required bones (`bones`) and constraints (`ik`, `transform`,
/// `path`, `physics`, and `slider`) a skin lists by name.
///
/// # Errors
/// Returns [`LoadError::BadReference`] for a name that names no bone or no
/// constraint of its type.
fn read_skin_requirements(sk: &Value, names: &Names, skin: &mut Skin) -> Result<(), LoadError> {
    let entries = |key: &str| {
        sk.get(key)
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
    };
    for bone in entries("bones") {
        skin.bones.push(names.bone(bone)?);
    }
    let mut add =
        |key: &str, map: &HashMap<&str, usize>, constraint: fn(usize) -> SkinConstraint| {
            for name in entries(key) {
                skin.constraints.push(constraint(lookup(map, name)?));
            }
            Ok::<(), LoadError>(())
        };
    add("ik", &names.ik, SkinConstraint::Ik)?;
    add("transform", &names.transform, SkinConstraint::Transform)?;
    add("path", &names.path, SkinConstraint::Path)?;
    add("physics", &names.physics, SkinConstraint::Physics)?;
    add("slider", &names.sliders, SkinConstraint::Slider)
}

/// Resolve the animation each slider scrubs, named in its `animation` field,
/// once every animation is loaded. Sliders are in the order of the
/// `constraints` entries.
fn resolve_slider_animations(root: &Value, data: &mut SkeletonData) -> Result<(), LoadError> {
    let Some(constraints) = root.get("constraints").and_then(Value::as_array) else {
        return Ok(());
    };
    let sliders = constraints
        .iter()
        .filter(|cm| cm.get("type").and_then(Value::as_str) == Some("slider"));
    for (slider, cm) in data.sliders.iter_mut().zip(sliders) {
        if let Some(name) = cm.get("animation").and_then(Value::as_str) {
            let index = data.animations.iter().position(|a| a.name() == name);
            slider.animation_index =
                Some(index.ok_or_else(|| LoadError::BadReference(name.to_string()))?);
        }
    }
    Ok(())
}

/// Charge the copies linked-mesh resolution makes: each linked mesh becomes a
/// full copy of its source mesh, with its own sequence.
fn charge_linked_meshes(data: &SkeletonData, budget: &mut Budget) -> Result<(), LoadError> {
    let mut charged = Ok(());
    crate::link::for_each_link_source(data, |link, source| {
        if charged.is_ok() {
            charged = budget.charge(linked_copy_cost(link, source), "a linked mesh");
        }
    });
    charged
}

/// Resolve a constraint's named constrained bones to indices.
fn constraint_bones(cm: &Value, kind: &str, names: &Names) -> Result<Vec<usize>, LoadError> {
    let mut bones = Vec::new();
    if let Some(bs) = cm.get("bones").and_then(Value::as_array) {
        for bn in bs {
            let bn = bn
                .as_str()
                .ok_or_else(|| LoadError::Schema(format!("{kind} bone name")))?;
            bones.push(names.bone(bn)?);
        }
    }
    Ok(bones)
}

/// Parse an `ik` constraint entry.
fn parse_ik(cm: &Value, order: usize, names: &Names) -> Result<IkConstraintData, LoadError> {
    let name = str_field(cm, "name")?;
    let bones = constraint_bones(cm, "ik", names)?;
    let target = names.bone(str_ref(cm, "target")?)?;
    let scale_y_mode = match cm.get("scaleY").and_then(Value::as_str) {
        Some("uniform") => ScaleYMode::Uniform,
        Some("volume") => ScaleYMode::Volume,
        _ => ScaleYMode::None,
    };
    Ok(IkConstraintData {
        name,
        order,
        skin_required: bool_or(cm, "skin", false),
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
    names: &Names,
) -> Result<TransformConstraintData, LoadError> {
    let name = str_field(cm, "name")?;
    let bones = constraint_bones(cm, "transform", names)?;
    let source = names.bone(str_ref(cm, "source")?)?;
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
        skin_required: bool_or(cm, "skin", false),
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
fn parse_path(cm: &Value, order: usize, names: &Names) -> Result<PathConstraintData, LoadError> {
    let name = str_field(cm, "name")?;
    let bones = constraint_bones(cm, "path", names)?;
    let slot = names.slot(str_ref(cm, "slot")?)?;
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
        skin_required: bool_or(cm, "skin", false),
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

/// Parse a `slider` constraint entry. A slider driven by a bone maps the
/// bone's `property` to a scrub time, and one without a bone keeps a setup
/// `time`. Its animation is resolved once the animations are loaded.
fn parse_slider(cm: &Value, order: usize, names: &Names) -> Result<SliderData, LoadError> {
    let mut slider = SliderData {
        name: str_field(cm, "name")?,
        order,
        skin_required: bool_or(cm, "skin", false),
        looping: bool_or(cm, "loop", false),
        additive: bool_or(cm, "additive", false),
        mix: f_or(cm, "mix", 1.0),
        ..SliderData::default()
    };
    match cm.get("bone").and_then(Value::as_str) {
        Some(bone) => {
            slider.bone = Some(names.bone(bone)?);
            let property = str_ref(cm, "property")?;
            slider.property = Some(slider_property(property).ok_or_else(|| {
                LoadError::Schema(format!("unknown slider property '{property}'"))
            })?);
            slider.property_offset = f(cm, "from");
            slider.offset = f(cm, "to");
            slider.scale = f_or(cm, "scale", 1.0);
            slider.max = f(cm, "max");
            slider.local = bool_or(cm, "local", false);
        }
        None => slider.time = f(cm, "time"),
    }
    Ok(slider)
}

fn slider_property(name: &str) -> Option<SliderProperty> {
    Some(match name {
        "rotate" => SliderProperty::Rotate,
        "x" => SliderProperty::X,
        "y" => SliderProperty::Y,
        "scaleX" => SliderProperty::ScaleX,
        "scaleY" => SliderProperty::ScaleY,
        "shearY" => SliderProperty::ShearY,
        _ => return None,
    })
}

/// Parse a `physics` constraint entry. `fps` becomes a fixed `step` of `1/fps`
/// and `mass` is stored inverted.
fn parse_physics(
    cm: &Value,
    order: usize,
    names: &Names,
) -> Result<PhysicsConstraintData, LoadError> {
    let name = str_field(cm, "name")?;
    let bone = names.bone(str_ref(cm, "bone")?)?;
    let scale_y_mode = match cm.get("scaleY").and_then(Value::as_str) {
        Some("uniform") => ScaleYMode::Uniform,
        Some("volume") => ScaleYMode::Volume,
        _ => ScaleYMode::None,
    };
    // Spine's binary format stores `fps` in one byte, so exports stay within
    // 1 to 255. The cap also bounds the physics substeps per second, which a
    // huge `fps` (a step near zero) would make endless.
    let fps = cm
        .get("fps")
        .and_then(Value::as_f64)
        .unwrap_or(60.0)
        .clamp(1.0, MAX_PHYSICS_FPS);
    let mass = f_or(cm, "mass", 1.0);
    Ok(PhysicsConstraintData {
        name,
        order,
        skin_required: bool_or(cm, "skin", false),
        bone,
        x: f(cm, "x"),
        y: f(cm, "y"),
        rotate: f(cm, "rotate"),
        scale_x: f(cm, "scaleX"),
        shear_x: f(cm, "shearX"),
        limit: f_or(cm, "limit", 5000.0),
        step: 1.0 / (fps as f32),
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

fn str_ref<'v>(v: &'v Value, key: &str) -> Result<&'v str, LoadError> {
    v.get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| LoadError::Schema(format!("missing string field '{key}'")))
}

fn str_field(v: &Value, key: &str) -> Result<String, LoadError> {
    str_ref(v, key).map(String::from)
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

/// Parse one attachment as [`from_json`] does, for tests that check a single
/// attachment. No bones exist, and an error reads as `None`.
#[cfg(test)]
fn parse_attachment(name: &str, v: &Value) -> Option<Attachment> {
    read_attachment(name, v, 0).ok().flatten()
}

#[cfg(test)]
mod layout;
#[cfg(test)]
mod robustness;
#[cfg(test)]
mod tests;
