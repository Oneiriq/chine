//! Loading skeleton data from Spine's binary `.skel` export.
//!
//! Behind the `binary` feature, [`from_binary`] parses a Spine `.skel` export
//! into a [`SkeletonData`] rig. The binary format is a packed big-endian stream:
//! variable-length integers, 4-byte floats, and length-prefixed UTF-8 strings,
//! read section by section (header, bones, slots, constraints, skins, events,
//! animations) in a fixed order.
//!
//! The reader primitives ([`BinaryReader`]) are stable across Spine 4.x, while
//! the section layouts are transcribed from the Spine 4.3 format. The loader
//! covers the full format: header, bones, slots, constraints, skins, events,
//! and animations (with the per-slider physics index trailer).

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

/// An error from binary skeleton loading.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BinaryError {
    /// The data ended before the skeleton could be read.
    Truncated,
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

/// A cursor over a Spine `.skel` byte stream with the format's primitive reads.
///
/// All multi-byte values are big-endian. Reading past the end sets an overrun
/// flag and yields zero/empty, so a truncated file fails cleanly rather than
/// panicking.
pub struct BinaryReader<'a> {
    data: &'a [u8],
    pos: usize,
    overran: bool,
    error: Option<BinaryError>,
}

impl<'a> BinaryReader<'a> {
    /// A reader over `data`, positioned at the start.
    #[must_use]
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            pos: 0,
            overran: false,
            error: None,
        }
    }

    /// `true` if any read went past the end of the data.
    #[must_use]
    pub fn overran(&self) -> bool {
        self.overran
    }

    /// Bytes not yet consumed.
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    /// A var_uint list length, validated against the bytes left. Because every
    /// list item occupies at least one byte, a length greater than the
    /// remaining data is corrupt: this records [`BinaryError::CorruptLength`]
    /// and returns `0`, so the caller loops a bounded number of times instead
    /// of trusting an attacker-chosen length.
    pub fn count(&mut self) -> usize {
        let n = self.var_usize();
        if n > self.remaining() {
            self.fail(BinaryError::CorruptLength);
            return 0;
        }
        n
    }

    /// Record the first structural error seen. Truncation is tracked separately
    /// by [`Self::overran`]; later errors do not overwrite the first.
    pub fn fail(&mut self, error: BinaryError) {
        if self.error.is_none() {
            self.error = Some(error);
        }
    }

    /// The first structural error recorded, if any.
    #[must_use]
    pub fn error(&self) -> Option<&BinaryError> {
        self.error.as_ref()
    }

    /// One byte (zero past the end).
    pub fn byte(&mut self) -> u8 {
        match self.data.get(self.pos) {
            Some(&b) => {
                self.pos += 1;
                b
            }
            None => {
                self.overran = true;
                0
            }
        }
    }

    /// One boolean (any non-zero byte is `true`).
    pub fn bool(&mut self) -> bool {
        self.byte() != 0
    }

    /// A big-endian `u32` (4 bytes).
    pub fn u32(&mut self) -> u32 {
        let b0 = u32::from(self.byte());
        let b1 = u32::from(self.byte());
        let b2 = u32::from(self.byte());
        let b3 = u32::from(self.byte());
        (b0 << 24) | (b1 << 16) | (b2 << 8) | b3
    }

    /// A big-endian `f32` (4 bytes).
    pub fn float(&mut self) -> f32 {
        f32::from_bits(self.u32())
    }

    /// A variable-length unsigned integer: 7 bits per byte, high bit continues
    /// (Spine's `readInt(true)`). Up to 5 bytes.
    pub fn var_uint(&mut self) -> u32 {
        let mut b = u32::from(self.byte());
        let mut value = b & 0x7f;
        let mut shift = 7;
        while b & 0x80 != 0 && shift < 35 {
            b = u32::from(self.byte());
            value |= (b & 0x7f) << shift;
            shift += 7;
        }
        value
    }

    /// A variable-length count/index as `usize`.
    pub fn var_usize(&mut self) -> usize {
        self.var_uint() as usize
    }

    /// A zig-zag-encoded signed variable-length integer (Spine's `readInt(false)`).
    pub fn var_int(&mut self) -> i32 {
        let v = self.var_uint();
        ((v >> 1) ^ (v & 1).wrapping_neg()) as i32
    }

    /// A length-prefixed UTF-8 string: `var_uint` length where `0` is `None`,
    /// `1` is the empty string, and `n` is `n - 1` following bytes.
    pub fn string(&mut self) -> Option<String> {
        match self.var_usize() {
            0 => None,
            1 => Some(String::new()),
            n => {
                let len = n - 1;
                let end = self.pos.saturating_add(len);
                if end > self.data.len() {
                    self.overran = true;
                    self.pos = self.data.len();
                    return Some(String::new());
                }
                let s = String::from_utf8_lossy(&self.data[self.pos..end]).into_owned();
                self.pos = end;
                Some(s)
            }
        }
    }
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
/// attachments), events, and animations. Every timeline group is decoded: bone
/// (rotate / translate / scale / shear), slot (attachment, color, two-color,
/// alpha), deform, draw order, event, the IK / transform / path / physics
/// constraint timelines, and the slider and sequence timelines.
///
/// # Errors
/// Returns [`BinaryError::Truncated`] if the data ends mid-skeleton.
pub fn from_binary(bytes: &[u8]) -> Result<SkeletonData, BinaryError> {
    let mut r = BinaryReader::new(bytes);
    let mut data = SkeletonData::default();

    // Header: a 64-bit hash, the editor version, setup bounds, reference scale,
    // and a non-essential flag (with editor-only fields when set).
    let _hash = (r.u32(), r.u32());
    data.spine_version = r.string();
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
    // by index. Consumed here; used once those sections are read.
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
            // Spine 4.3 added two editor-only bone floats here; consume them.
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

    // Constraints are one typed list. Each begins with a name and a one-byte
    // type (Spine 4.3: IK=0, path=1, transform=2, physics=3, slider=4); there
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

    // Skins: the default skin, then named skins. Attachments resolve their
    // names and paths through the string table.
    let slot_names: Vec<String> = data.slots.iter().map(|s| s.name.clone()).collect();
    data.default_skin = read_skin(&mut r, &strings, &slot_names, true, nonessential);
    let skin_count = r.count();
    for _ in 0..skin_count {
        let skin = read_skin(&mut r, &strings, &slot_names, false, nonessential);
        data.skins.push(skin);
    }

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
    }

    // After the animations, each slider constraint records the index of the
    // animation it scrubs (read in constraint order, which is slider order).
    for slider in &mut data.sliders {
        slider.animation_index = Some(r.var_usize());
    }

    if let Some(error) = r.error() {
        return Err(error.clone());
    }
    if r.overran() {
        return Err(BinaryError::Truncated);
    }
    Ok(data)
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
fn parse_slider(r: &mut BinaryReader, name: String, order: usize, nonessential: bool) -> SliderData {
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
        // When nonessential, this float is the editor maximum; otherwise it is
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
            // An unknown property ordinal stops this constraint's property
            // block (matching the reference's `continue`).
            _ => return data,
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
/// scale-X encodes the Y-scale mode; chine loads at scale 1.
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
/// byte's high bits hold the source-property count; offsets default to 0 and
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
        let Some(property) = from_prop_byte(r.byte()) else {
            continue;
        };
        let offset = r.float();
        let to_count = usize::from(r.byte());
        let mut to = Vec::with_capacity(to_count);
        for _ in 0..to_count {
            let Some(to_property) = to_prop_byte(r.byte()) else {
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
/// is `strings[i - 1]`.
fn string_ref(r: &mut BinaryReader, strings: &[String]) -> Option<String> {
    match r.var_usize() {
        0 => None,
        i => strings.get(i - 1).cloned(),
    }
}

/// Read a skin: the default skin (`is_default`, a slot count then attachments)
/// or a named skin (name, bone/constraint index lists, then attachments).
fn read_skin(
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


mod timelines;
use timelines::read_animation;

#[cfg(test)]
mod tests;
