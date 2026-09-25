//! Loading skeleton data from Spine's binary `.skel` export.
//!
//! Behind the `binary` feature, [`from_binary`] parses a Spine `.skel` export
//! into a [`SkeletonData`] rig. The binary format is a packed big-endian stream:
//! variable-length integers, 4-byte floats, and length-prefixed UTF-8 strings,
//! read section by section (header, bones, slots, constraints, skins, events,
//! animations) in a fixed order.
//!
//! The reader primitives ([`BinaryReader`]) are stable across Spine 4.x, while
//! the section layouts follow the Spine 4.3 binary format. The loader
//! reads every section: header, bones, slots, constraints, skins, events,
//! and animations (with the per-slider animation index trailer).

use std::sync::Arc;

use glam::Vec2;

use crate::anim::{
    compute_draw_order, Animation, AttachmentTimeline, BoneAxis, BoneTimeline, ConstraintTimeline,
    DeformTimeline, DrawOrderTimeline, EventTimeline, PhysicsProperty, PhysicsResetTimeline,
    SequenceTimeline, Timeline, PATH_MIX, PATH_POSITION, PATH_SPACING, TRANSFORM_MIX,
};
use crate::attach::{
    Attachment, BoundingBoxAttachment, ClippingAttachment, LinkedMeshAttachment, MeshAttachment,
    MeshVertices, PathAttachment, PointAttachment, RegionAttachment, Sequence,
};
use crate::constraint::ik::IkConstraintData;
use crate::constraint::path::{PathConstraintData, PositionMode, RotateMode, SpacingMode};
use crate::constraint::physics::PhysicsConstraintData;
use crate::constraint::slider::{SliderData, SliderProperty};
use crate::constraint::transform::{
    FromMapping, FromProp, ToMapping, ToProp, TransformConstraintData,
};
use crate::constraint::ScaleYMode;
use crate::data::{BlendMode, BoneData, Color, Inherit, SkeletonData, SlotData};
use crate::event::{Event, EventData};
use crate::skin::Skin;

/// The Spine `major.minor` release whose binary layout this loader reads.
///
/// The header's version string selects the section layouts: other releases
/// add, drop, or re-pack fields, so their exports cannot be read as 4.3 data.
const SUPPORTED_VERSION: &str = "4.3";

/// An error from binary skeleton loading.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BinaryError {
    /// The data ended before the skeleton could be read.
    Truncated,
    /// The header declared an export from a Spine release whose binary layout
    /// this loader does not read (it reads Spine 4.3 exports).
    UnsupportedVersion {
        /// The version string the header declared.
        found: String,
        /// The `major.minor` release this loader reads.
        expected: &'static str,
    },
    /// A list declared more items than the remaining bytes could hold. Since
    /// every item occupies at least one byte, the declared length is corrupt.
    CorruptLength,
    /// A constraint carried an unrecognized type tag (valid tags are 0 to 4).
    UnknownConstraintType(u8),
    /// An animation timeline carried an unrecognized type tag.
    UnknownTimelineType(u8),
}

impl std::fmt::Display for BinaryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BinaryError::Truncated => write!(f, "binary skeleton data is truncated"),
            BinaryError::UnsupportedVersion { found, expected } => {
                write!(
                    f,
                    "unsupported Spine version {found} (this loader reads {expected} exports)"
                )
            }
            BinaryError::CorruptLength => {
                write!(f, "a list length exceeds the remaining data")
            }
            BinaryError::UnknownConstraintType(tag) => {
                write!(f, "unknown constraint type tag {tag}")
            }
            BinaryError::UnknownTimelineType(tag) => {
                write!(f, "unknown timeline type tag {tag}")
            }
        }
    }
}

impl std::error::Error for BinaryError {}

/// Whether a header version string declares the `major.minor` release this
/// loader reads. The editor writes `major.minor.patch`, and only `major.minor`
/// selects the binary layout, so every 4.3 patch release is accepted.
fn version_supported(version: &str) -> bool {
    match version.strip_prefix(SUPPORTED_VERSION) {
        Some(rest) => rest.is_empty() || rest.starts_with('.'),
        None => false,
    }
}

/// Record a corrupt index or length as [`BinaryError::CorruptLength`].
///
/// Once the data has run out, every read returns zero, so a bad value read
/// after that point is a side effect of the truncation. It is not recorded,
/// and the load reports [`BinaryError::Truncated`] instead.
fn corrupt(r: &mut BinaryReader) {
    if !r.overran() {
        r.fail(BinaryError::CorruptLength);
    }
}

/// The load's outcome so far: the first structural error, then truncation.
fn status(r: &BinaryReader) -> Result<(), BinaryError> {
    if let Some(error) = r.error() {
        return Err(error.clone());
    }
    if r.overran() {
        return Err(BinaryError::Truncated);
    }
    Ok(())
}

/// Whether every index in `indices` is below `len`.
fn all_below(indices: &[usize], len: usize) -> bool {
    indices.iter().all(|&i| i < len)
}

/// The number of constraints of every type. Spine 4.3 keeps one constraint
/// list, and constraint timelines and skins index into it.
fn constraint_count(data: &SkeletonData) -> usize {
    data.ik_constraints.len()
        + data.path_constraints.len()
        + data.transform_constraints.len()
        + data.physics_constraints.len()
        + data.sliders.len()
}

/// Map a Spine inherit-mode ordinal to [`Inherit`].
fn inherit_from(ordinal: usize) -> Inherit {
    match ordinal {
        1 => Inherit::OnlyTranslation,
        2 => Inherit::NoRotationOrReflection,
        3 => Inherit::NoScale,
        4 => Inherit::NoScaleOrReflection,
        _ => Inherit::Normal,
    }
}

/// Parse a Spine `.skel` binary export into a [`SkeletonData`].
///
/// The full section sequence is read: header, bones, slots, the IK / transform
/// / path / physics / slider constraints, skins (including sequence
/// attachments), events, and animations. The timeline groups decoded are bone
/// (rotate / translate / scale / shear), slot (attachment, color, two-color,
/// alpha), deform, draw order, event, the IK / transform / path / physics
/// constraint timelines, and the slider and sequence timelines. Bone inherit
/// timelines are not decoded and report
/// [`BinaryError::UnknownTimelineType`], and draw order folder timelines are
/// read past without being kept.
///
/// Every index the file stores (bone parents, slot bones, constraint bones and
/// targets, timeline targets, skin slots, weighted vertex bones, mesh
/// triangles, string table references, draw order moves, event and animation
/// references) is checked against the table it points into, so the loaded
/// data never indexes out of range.
///
/// # Errors
/// Returns [`BinaryError::UnsupportedVersion`] if the header declares an
/// export from a Spine release other than 4.3,
/// [`BinaryError::Truncated`] if the data ends mid-skeleton,
/// [`BinaryError::CorruptLength`] if a length exceeds the remaining data or
/// an index is out of range, and [`BinaryError::UnknownConstraintType`] or
/// [`BinaryError::UnknownTimelineType`] for an unrecognized type tag.
pub fn from_binary(bytes: &[u8]) -> Result<SkeletonData, BinaryError> {
    let mut r = BinaryReader::new(bytes);
    let mut data = SkeletonData::default();

    // Header: a 64-bit hash, the editor version, setup bounds, reference scale,
    // and a non-essential flag (with editor-only fields when set).
    let _hash = (r.u32(), r.u32());
    data.spine_version = r.string();
    // Validate the version before trusting the section layouts below: an
    // export from another release (4.4+, 4.2, 3.x) would otherwise misparse
    // silently. A missing or empty version (hand-built data) is accepted.
    if let Some(found) = data.spine_version.as_deref() {
        if !found.is_empty() && !version_supported(found) {
            return Err(BinaryError::UnsupportedVersion {
                found: found.to_string(),
                expected: SUPPORTED_VERSION,
            });
        }
    }
    let x = r.float();
    let y = r.float();
    let _width = r.float();
    let _height = r.float();
    data.position = Vec2::new(x, y);
    data.reference_scale = r.float();
    let nonessential = r.bool();
    if nonessential {
        let _fps = r.float();
        let _images_path = r.string();
        let _audio_path = r.string();
    }

    // String table: names (attachments, events, ...) the later sections refer to
    // by index. Consumed here and used once those sections are read.
    let string_count = r.count();
    let strings: Vec<String> = (0..string_count)
        .map(|_| r.string().unwrap_or_default())
        .collect();

    // Bones: hierarchy order, the root first (and parentless).
    let bone_count = r.count();
    for index in 0..bone_count {
        let name = r.string().unwrap_or_default();
        let parent = if index == 0 {
            None
        } else {
            Some(r.var_usize())
        };
        let rotation = r.float();
        let bx = r.float();
        let by = r.float();
        let scale_x = r.float();
        let scale_y = r.float();
        let shear_x = r.float();
        let shear_y = r.float();
        let length = r.float();
        let inherit = inherit_from(r.byte() as usize);
        let _skin_required = r.bool();
        if nonessential {
            let _color = r.u32();
            let _icon = r.string();
            // Spine 4.3 added two editor-only bone floats here. Consume them.
            let _f0 = r.float();
            let _f1 = r.float();
            let _visible = r.bool();
        }
        data.bones.push(BoneData {
            index,
            name,
            parent,
            length,
            position: Vec2::new(bx, by),
            rotation,
            scale: Vec2::new(scale_x, scale_y),
            shear: Vec2::new(shear_x, shear_y),
            inherit,
        });
    }
    // Spine writes every parent before its children, so a parent at or after
    // its child is corrupt. This also rules out parent cycles.
    if !data
        .bones
        .iter()
        .enumerate()
        .all(|(i, b)| !matches!(b.parent, Some(p) if p >= i))
    {
        corrupt(&mut r);
    }
    status(&r)?;

    // Slots: draw order, each with a setup color, an optional dark (two-color)
    // tint, and the setup attachment (referenced into the string table).
    let slot_count = r.count();
    for index in 0..slot_count {
        let name = r.string().unwrap_or_default();
        let bone = r.var_usize();
        let color = color_rgba(r.u32());
        let dark = r.u32();
        let dark_color = (dark != 0xFFFF_FFFF).then(|| color_rgb(dark));
        let attachment = string_ref(&mut r, &strings);
        let blend = blend_from(r.var_usize());
        if nonessential {
            let _visible = r.bool();
        }
        data.slots.push(SlotData {
            index,
            name,
            bone,
            color,
            dark_color,
            attachment,
            blend,
        });
    }
    let bone_count = data.bones.len();
    if !data.slots.iter().all(|s| s.bone < bone_count) {
        corrupt(&mut r);
    }
    status(&r)?;

    // Constraints are one typed list. Each begins with a name and a one-byte
    // type (Spine 4.3: IK=0, path=1, transform=2, physics=3, slider=4). There
    // is no explicit order (it is the read order across all types). IK packs
    // its flags into one byte and stores only non-default mix/softness.
    let constraint_count = r.count();
    for order in 0..constraint_count {
        let name = r.string().unwrap_or_default();
        let kind = r.byte();
        match kind {
            0 => data.ik_constraints.push(parse_ik(&mut r, name, order)),
            1 => data.path_constraints.push(parse_path(&mut r, name, order)),
            2 => data
                .transform_constraints
                .push(parse_transform(&mut r, name, order)),
            3 => data
                .physics_constraints
                .push(parse_physics(&mut r, name, order)),
            4 => data
                .sliders
                .push(parse_slider(&mut r, name, order, nonessential)),
            _ => {
                r.fail(BinaryError::UnknownConstraintType(kind));
                break;
            }
        }
    }
    if !constraint_indices_valid(&data) {
        corrupt(&mut r);
    }
    status(&r)?;

    // Skins: the default skin, then named skins. Attachments resolve their
    // names and paths through the string table.
    data.default_skin = read_skin(&mut r, &strings, &data, true, nonessential);
    let skin_count = r.count();
    for _ in 0..skin_count {
        let skin = read_skin(&mut r, &strings, &data, false, nonessential);
        data.skins.push(skin);
    }
    status(&r)?;

    // Resolve linked meshes before animations so deform timelines bind to the
    // resolved (parent-shared) geometry rather than unresolved links.
    crate::link::resolve_linked_meshes(&mut data);

    // Events: setup-pose values for named animation events.
    let event_count = r.count();
    for _ in 0..event_count {
        let name = r.string().unwrap_or_default();
        let int_value = r.var_int();
        let float_value = r.float();
        let string_value = r.string().unwrap_or_default();
        let audio_path = r.string();
        let (volume, balance) = if audio_path.is_some() {
            (r.float(), r.float())
        } else {
            (1.0, 0.0)
        };
        data.events.push(EventData {
            name,
            int_value,
            float_value,
            string_value,
            audio_path,
            volume,
            balance,
        });
    }

    // Animations.
    let animation_count = r.count();
    for _ in 0..animation_count {
        let aname = r.string().unwrap_or_default();
        let anim = read_animation(&mut r, aname, &data, &strings, nonessential);
        data.animations.push(Arc::new(anim));
        status(&r)?;
    }

    // After the animations, each slider constraint records the index of the
    // animation it scrubs (read in constraint order, which is slider order).
    let animation_count = data.animations.len();
    for slider in &mut data.sliders {
        let index = r.var_usize();
        if index >= animation_count {
            corrupt(&mut r);
        }
        slider.animation_index = Some(index);
    }

    status(&r)?;
    Ok(data)
}

/// Whether every constraint's bone, target, and slot index is in range.
fn constraint_indices_valid(data: &SkeletonData) -> bool {
    let bones = data.bones.len();
    let slots = data.slots.len();
    data.ik_constraints
        .iter()
        .all(|c| c.target < bones && all_below(&c.bones, bones))
        && data
            .transform_constraints
            .iter()
            .all(|c| c.source < bones && all_below(&c.bones, bones))
        && data
            .path_constraints
            .iter()
            .all(|c| c.slot < slots && all_below(&c.bones, bones))
        && data.physics_constraints.iter().all(|c| c.bone < bones)
        && data
            .sliders
            .iter()
            .all(|c| !matches!(c.bone, Some(b) if b >= bones))
}

/// Map a scale-Y-mode ordinal to [`ScaleYMode`].
fn scale_y_from(ordinal: usize) -> ScaleYMode {
    match ordinal {
        1 => ScaleYMode::Uniform,
        2 => ScaleYMode::Volume,
        _ => ScaleYMode::None,
    }
}

/// Parse one IK constraint (Spine 4.3 bit-packed layout).
fn parse_ik(r: &mut BinaryReader, name: String, order: usize) -> IkConstraintData {
    let bone_count = r.count();
    let bones = (0..bone_count).map(|_| r.var_usize()).collect();
    let target = r.var_usize();
    let flags = r.byte();
    let scale_y_mode = if flags & 2 != 0 {
        scale_y_from(r.byte() as usize)
    } else {
        ScaleYMode::None
    };
    let mix = if flags & 32 != 0 {
        if flags & 64 != 0 {
            r.float()
        } else {
            1.0
        }
    } else {
        1.0
    };
    let softness = if flags & 128 != 0 { r.float() } else { 0.0 };
    IkConstraintData {
        name,
        order,
        bones,
        target,
        scale_y_mode,
        mix,
        softness,
        bend_direction: if flags & 4 != 0 { -1 } else { 1 },
        compress: flags & 8 != 0,
        stretch: flags & 16 != 0,
    }
}

/// Parse one slider constraint (Spine 4.3 bit-packed layout). chine loads at
/// scale 1, so the property scale factor (applied to X / Y) is the identity.
fn parse_slider(
    r: &mut BinaryReader,
    name: String,
    order: usize,
    nonessential: bool,
) -> SliderData {
    let mut data = SliderData {
        name,
        order,
        ..Default::default()
    };
    let flags = r.byte();
    data.skin_required = flags & 1 != 0;
    data.looping = flags & 2 != 0;
    data.additive = flags & 4 != 0;
    if flags & 8 != 0 {
        let value = r.float();
        // When nonessential, this float is the editor maximum. Otherwise it is
        // the setup-pose slider time.
        if nonessential && flags & 64 != 0 {
            data.max = value;
        } else {
            data.time = value;
        }
    }
    if flags & 16 != 0 {
        data.mix = if flags & 32 != 0 { r.float() } else { 1.0 };
    }
    if flags & 64 != 0 {
        data.local = flags & 128 != 0;
        data.bone = Some(r.var_usize());
        let property_offset = r.float();
        match r.byte() {
            0 => data.property = Some(SliderProperty::Rotate),
            1 => data.property = Some(SliderProperty::X),
            2 => data.property = Some(SliderProperty::Y),
            3 => data.property = Some(SliderProperty::ScaleX),
            4 => data.property = Some(SliderProperty::ScaleY),
            5 => data.property = Some(SliderProperty::ShearY),
            // An unknown property ordinal is corrupt (Spine's reader cannot
            // load it), and the fields after it are left unread.
            _ => {
                corrupt(r);
                return data;
            }
        }
        data.property_offset = property_offset;
        data.offset = r.float();
        data.scale = r.float();
    }
    data
}

/// Parse one path constraint (Spine 4.3 bit-packed layout). chine loads at
/// scale 1, so the fixed-position / fixed-spacing scale factors are the
/// identity.
fn parse_path(r: &mut BinaryReader, name: String, order: usize) -> PathConstraintData {
    let bone_count = r.count();
    let bones = (0..bone_count).map(|_| r.var_usize()).collect();
    let slot = r.var_usize();
    let flags = r.byte();
    let position_mode = if (flags >> 1) & 1 == 0 {
        PositionMode::Fixed
    } else {
        PositionMode::Percent
    };
    let spacing_mode = match (flags >> 2) & 0b11 {
        0 => SpacingMode::Length,
        1 => SpacingMode::Fixed,
        3 => SpacingMode::Proportional,
        _ => SpacingMode::Percent,
    };
    let rotate_mode = match (flags >> 4) & 0b11 {
        0 => RotateMode::Tangent,
        2 => RotateMode::ChainScale,
        _ => RotateMode::Chain,
    };
    let offset_rotation = if flags & 128 != 0 { r.float() } else { 0.0 };
    PathConstraintData {
        name,
        order,
        bones,
        slot,
        position_mode,
        spacing_mode,
        rotate_mode,
        offset_rotation,
        position: r.float(),
        spacing: r.float(),
        mix_rotate: r.float(),
        mix_x: r.float(),
        mix_y: r.float(),
    }
}

/// Parse one physics constraint (Spine 4.3 bit-packed layout). A negative packed
/// scale-X encodes the Y-scale mode. chine loads at scale 1.
fn parse_physics(r: &mut BinaryReader, name: String, order: usize) -> PhysicsConstraintData {
    let bone = r.var_usize();
    let flags = r.byte();
    let x = if flags & 2 != 0 { r.float() } else { 0.0 };
    let y = if flags & 4 != 0 { r.float() } else { 0.0 };
    let rotate = if flags & 8 != 0 { r.float() } else { 0.0 };
    let (scale_x, scale_y_mode) = if flags & 16 != 0 {
        let raw = r.float();
        if raw < -2.0 {
            (-2.0 - raw, ScaleYMode::Volume)
        } else if raw < 0.0 {
            (-1.0 - raw, ScaleYMode::Uniform)
        } else {
            (raw, ScaleYMode::None)
        }
    } else {
        (0.0, ScaleYMode::None)
    };
    let shear_x = if flags & 32 != 0 { r.float() } else { 0.0 };
    let limit = if flags & 64 != 0 { r.float() } else { 5000.0 };
    let step = 1.0 / f32::from(r.byte());
    let inertia = r.float();
    let strength = r.float();
    let damping = r.float();
    let mass_inverse = if flags & 128 != 0 { r.float() } else { 1.0 };
    let wind = r.float();
    let gravity = r.float();
    let global = r.byte();
    let mix = if global & 128 != 0 { r.float() } else { 1.0 };
    PhysicsConstraintData {
        name,
        order,
        bone,
        x,
        y,
        rotate,
        scale_x,
        shear_x,
        limit,
        step,
        scale_y_mode,
        inertia,
        strength,
        damping,
        mass_inverse,
        wind,
        gravity,
        mix,
        inertia_global: global & 1 != 0,
        strength_global: global & 2 != 0,
        damping_global: global & 4 != 0,
        mass_global: global & 8 != 0,
        wind_global: global & 16 != 0,
        gravity_global: global & 32 != 0,
        mix_global: global & 64 != 0,
    }
}

/// Map a transform source-property ordinal to a [`FromProp`].
fn from_prop_byte(b: u8) -> Option<FromProp> {
    Some(match b {
        0 => FromProp::Rotate,
        1 => FromProp::X,
        2 => FromProp::Y,
        3 => FromProp::ScaleX,
        4 => FromProp::ScaleY,
        5 => FromProp::ShearY,
        _ => return None,
    })
}

/// Map a transform target-property ordinal to a [`ToProp`].
fn to_prop_byte(b: u8) -> Option<ToProp> {
    Some(match b {
        0 => ToProp::Rotate,
        1 => ToProp::X,
        2 => ToProp::Y,
        3 => ToProp::ScaleX,
        4 => ToProp::ScaleY,
        5 => ToProp::ShearY,
        _ => return None,
    })
}

/// Parse one transform constraint (Spine 4.3 property-mapping layout). The flags
/// byte's high bits hold the source-property count. Offsets default to 0 and
/// mixes to 1 when their flag is clear. chine loads at scale 1.
fn parse_transform(r: &mut BinaryReader, name: String, order: usize) -> TransformConstraintData {
    let bone_count = r.count();
    let bones = (0..bone_count).map(|_| r.var_usize()).collect();
    let source = r.var_usize();
    let flags = r.byte();
    let local_source = flags & 2 != 0;
    let local_target = flags & 4 != 0;
    let additive = flags & 8 != 0;
    let clamp = flags & 16 != 0;
    let property_count = (flags >> 5) as usize;
    let mut properties = Vec::with_capacity(property_count);
    for _ in 0..property_count {
        // An unknown property ordinal is corrupt (Spine's reader cannot load
        // it), and the mapping is skipped.
        let Some(property) = from_prop_byte(r.byte()) else {
            corrupt(r);
            continue;
        };
        let offset = r.float();
        let to_count = usize::from(r.byte());
        let mut to = Vec::with_capacity(to_count);
        for _ in 0..to_count {
            let Some(to_property) = to_prop_byte(r.byte()) else {
                corrupt(r);
                continue;
            };
            to.push(ToMapping {
                property: to_property,
                offset: r.float(),
                max: r.float(),
                scale: r.float(),
            });
        }
        properties.push(FromMapping {
            property,
            offset,
            to,
        });
    }
    let off_flags = r.byte();
    let mut offsets = [0.0_f32; 6];
    for (i, offset) in offsets.iter_mut().enumerate() {
        if off_flags & (1 << i) != 0 {
            *offset = r.float();
        }
    }
    let mix_flags = r.byte();
    TransformConstraintData {
        name,
        order,
        bones,
        source,
        offsets,
        local_source,
        local_target,
        additive,
        clamp,
        properties,
        mix_rotate: if mix_flags & 1 != 0 { r.float() } else { 1.0 },
        mix_x: if mix_flags & 2 != 0 { r.float() } else { 1.0 },
        mix_y: if mix_flags & 4 != 0 { r.float() } else { 1.0 },
        mix_scale_x: if mix_flags & 8 != 0 { r.float() } else { 1.0 },
        mix_scale_y: if mix_flags & 16 != 0 { r.float() } else { 1.0 },
        mix_shear_y: if mix_flags & 32 != 0 { r.float() } else { 1.0 },
    }
}

/// RGBA8888 (`0xRRGGBBAA`) to a [`Color`].
fn color_rgba(v: u32) -> Color {
    Color::new(
        ((v >> 24) & 0xff) as f32 / 255.0,
        ((v >> 16) & 0xff) as f32 / 255.0,
        ((v >> 8) & 0xff) as f32 / 255.0,
        (v & 0xff) as f32 / 255.0,
    )
}

/// RGB888 (`0x00RRGGBB`, opaque) to a [`Color`].
fn color_rgb(v: u32) -> Color {
    Color::new(
        ((v >> 16) & 0xff) as f32 / 255.0,
        ((v >> 8) & 0xff) as f32 / 255.0,
        (v & 0xff) as f32 / 255.0,
        1.0,
    )
}

/// Map a Spine blend-mode ordinal to [`BlendMode`].
fn blend_from(ordinal: usize) -> BlendMode {
    match ordinal {
        1 => BlendMode::Additive,
        2 => BlendMode::Multiply,
        3 => BlendMode::Screen,
        _ => BlendMode::Normal,
    }
}

/// Read a string-table reference: a `var_uint` index where `0` is `None` and `i`
/// is `strings[i - 1]`. An index past the table is corrupt.
fn string_ref(r: &mut BinaryReader, strings: &[String]) -> Option<String> {
    let index = r.var_usize().checked_sub(1)?;
    let s = strings.get(index).cloned();
    if s.is_none() {
        corrupt(r);
    }
    s
}

mod reader;
pub use reader::BinaryReader;

mod skins;
use skins::read_skin;

mod timelines;
use timelines::read_animation;

#[cfg(test)]
mod robustness;
#[cfg(test)]
mod tests;
