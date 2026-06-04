//! Loading skeleton data from Spine's binary `.skel` export.
//!
//! Behind the `binary` feature, [`from_binary`] parses a Spine `.skel` export
//! into a [`SkeletonData`] rig. The binary format is a packed big-endian stream:
//! variable-length integers, 4-byte floats, and length-prefixed UTF-8 strings,
//! read section by section (header, bones, slots, constraints, skins, events,
//! animations) in a fixed order.
//!
//! This loader is built up section by section; the reader primitives
//! ([`BinaryReader`]) are stable across Spine 4.x, while the section layouts are
//! transcribed from the Spine 4.3 format. Sections beyond those already read are
//! left for later and simply end the parse early.

use std::sync::Arc;

use glam::Vec2;

use crate::anim::{
    compute_draw_order, Animation, AttachmentTimeline, BoneAxis, BoneTimeline, ConstraintTimeline,
    DeformTimeline, DrawOrderTimeline, EventTimeline, PhysicsProperty, PhysicsResetTimeline,
    SequenceTimeline, Timeline,
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
}

impl std::fmt::Display for BinaryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BinaryError::Truncated => write!(f, "binary skeleton data is truncated"),
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
}

impl<'a> BinaryReader<'a> {
    /// A reader over `data`, positioned at the start.
    #[must_use]
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            pos: 0,
            overran: false,
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
/// The full section sequence is read: header, bones, slots, constraints, skins,
/// events, and animations (bone timelines; other timeline groups are skipped
/// while empty). A truncated stream returns [`BinaryError::Truncated`].
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
    let string_count = r.var_usize();
    let strings: Vec<String> = (0..string_count)
        .map(|_| r.string().unwrap_or_default())
        .collect();

    // Bones: hierarchy order, the root first (and parentless).
    let bone_count = r.var_usize();
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
    let slot_count = r.var_usize();
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
    let constraint_count = r.var_usize();
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
            _ => break,
        }
    }

    // Skins: the default skin, then named skins. Attachments resolve their
    // names and paths through the string table.
    let slot_names: Vec<String> = data.slots.iter().map(|s| s.name.clone()).collect();
    data.default_skin = read_skin(&mut r, &strings, &slot_names, true, nonessential);
    let skin_count = r.var_usize();
    for _ in 0..skin_count {
        let skin = read_skin(&mut r, &strings, &slot_names, false, nonessential);
        data.skins.push(skin);
    }

    // Events: setup-pose values for named animation events.
    let event_count = r.var_usize();
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
    let animation_count = r.var_usize();
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
    let bone_count = r.var_usize();
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
    let bone_count = r.var_usize();
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
    let bone_count = r.var_usize();
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
        slot_count = r.var_usize();
        skin = Skin::new("default");
        if slot_count == 0 {
            return skin;
        }
    } else {
        skin = Skin::new(r.string().unwrap_or_default());
        if nonessential {
            let _ = r.u32();
        }
        let bone_count = r.var_usize();
        for _ in 0..bone_count {
            r.var_usize();
        }
        let constraint_count = r.var_usize();
        for _ in 0..constraint_count {
            r.var_usize();
        }
        slot_count = r.var_usize();
    }
    for _ in 0..slot_count {
        let slot = r.var_usize();
        let att_count = r.var_usize();
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
            let timeline_slots = r.var_usize();
            for _ in 0..timeline_slots {
                r.var_usize();
            }
            if nonessential {
                let edges = r.var_usize();
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
    let count = r.var_usize();
    let start = r.var_usize();
    let digits = r.var_usize();
    let setup_index = r.var_usize();
    Some(Sequence::new(count, start, digits, setup_index))
}

/// Read mesh/polygon vertices: unweighted (`2 * vertexCount` floats) or weighted
/// (per-vertex bone influences). Returns the vertices and the vertex count.
fn read_vertices(r: &mut BinaryReader, weighted: bool) -> (MeshVertices, usize) {
    let count = r.var_usize();
    if !weighted {
        return (
            MeshVertices::Unweighted(read_float_array(r, count * 2)),
            count,
        );
    }
    let total = r.var_usize();
    let mut bones = Vec::new();
    let mut vertices = Vec::new();
    while bones.len() < total {
        let influences = r.var_usize();
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

/// Wrap a parsed bone timeline in its [`Timeline`] variant.
fn wrap((tl, d): (BoneTimeline, f32), make: fn(BoneTimeline) -> Timeline) -> (Timeline, f32) {
    (make(tl), d)
}

/// Wrap a parsed single-axis bone timeline.
fn axis((tl, d): (BoneTimeline, f32), a: BoneAxis) -> (Timeline, f32) {
    (Timeline::BoneAxis(tl, a), d)
}

/// Parse one animation. Built group by group; on an unhandled timeline type the
/// parse stops early and returns what it has (later sections are left unread).
fn read_animation(
    r: &mut BinaryReader,
    name: String,
    data: &SkeletonData,
    strings: &[String],
    nonessential: bool,
) -> Animation {
    let _timeline_count = r.var_usize();
    let mut timelines = Vec::new();
    let mut duration = 0.0_f32;

    // Slot timelines: per animated slot, one or more typed timelines (color,
    // two-color, attachment; alpha is consumed but not yet applied).
    let slot_groups = r.var_usize();
    for _ in 0..slot_groups {
        let slot = r.var_usize();
        let count = r.var_usize();
        for _ in 0..count {
            let kind = r.byte();
            let frames = r.var_usize();
            let entry: Option<(Timeline, f32)> = match kind {
                0 => {
                    let (tl, d) = read_slot_attachment_timeline(r, slot, frames, strings);
                    Some((Timeline::Attachment(tl), d))
                }
                1 => {
                    let (tl, d) = read_slot_color_timeline(r, slot, frames, 4);
                    Some((Timeline::SlotColor(tl, true), d))
                }
                2 => {
                    let (tl, d) = read_slot_color_timeline(r, slot, frames, 3);
                    Some((Timeline::SlotColor(tl, false), d))
                }
                3 => {
                    let (tl, d) = read_slot_color_timeline(r, slot, frames, 7);
                    Some((Timeline::SlotTwoColor(tl, true), d))
                }
                4 => {
                    let (tl, d) = read_slot_color_timeline(r, slot, frames, 6);
                    Some((Timeline::SlotTwoColor(tl, false), d))
                }
                5 => {
                    let (tl, d) = read_slot_color_timeline(r, slot, frames, 1);
                    Some((Timeline::SlotAlpha(tl), d))
                }
                _ => return Animation::new(name, duration, timelines),
            };
            if let Some((timeline, d)) = entry {
                duration = duration.max(d);
                timelines.push(timeline);
            }
        }
    }

    // Bone timelines: one group per animated bone, each with one or more typed
    // timelines (rotate / translate / scale / shear and their single axes).
    let bone_groups = r.var_usize();
    for _ in 0..bone_groups {
        let bone = r.var_usize();
        let count = r.var_usize();
        for _ in 0..count {
            let kind = r.byte();
            let frames = r.var_usize();
            let (timeline, d) = match kind {
                0 => wrap(read_bone_timeline1(r, bone, frames), Timeline::Rotate),
                1 => wrap(read_bone_timeline2(r, bone, frames), Timeline::Translate),
                2 => axis(read_bone_timeline1(r, bone, frames), BoneAxis::TranslateX),
                3 => axis(read_bone_timeline1(r, bone, frames), BoneAxis::TranslateY),
                4 => wrap(read_bone_timeline2(r, bone, frames), Timeline::Scale),
                5 => axis(read_bone_timeline1(r, bone, frames), BoneAxis::ScaleX),
                6 => axis(read_bone_timeline1(r, bone, frames), BoneAxis::ScaleY),
                7 => wrap(read_bone_timeline2(r, bone, frames), Timeline::Shear),
                8 => axis(read_bone_timeline1(r, bone, frames), BoneAxis::ShearX),
                9 => axis(read_bone_timeline1(r, bone, frames), BoneAxis::ShearY),
                _ => return Animation::new(name, duration, timelines),
            };
            duration = duration.max(d);
            timelines.push(timeline);
        }
    }

    // Remaining timeline groups (IK, transform, path, physics, slider,
    // attachment/deform, draw order, draw-order folders, events): skip while
    // empty, stop at the first non-empty (not yet handled).
    // IK constraint timelines (one per animated IK constraint).
    let ik_groups = r.var_usize();
    for _ in 0..ik_groups {
        let index = r.var_usize();
        let frames = r.var_usize();
        let (tl, d) = read_ik_constraint_timeline(r, index, frames);
        duration = duration.max(d);
        timelines.push(Timeline::Ik(tl));
    }

    // Transform constraint timelines (six mix channels).
    let transform_groups = r.var_usize();
    for _ in 0..transform_groups {
        let index = r.var_usize();
        let frames = r.var_usize();
        let (tl, d) = read_curve_timeline_n(r, index, frames, 6);
        duration = duration.max(d);
        timelines.push(Timeline::TransformMix(tl));
    }

    // Path constraint timelines: position, spacing, or mix per inner entry.
    let path_groups = r.var_usize();
    for _ in 0..path_groups {
        let index = r.var_usize();
        let count = r.var_usize();
        for _ in 0..count {
            let kind = r.byte();
            let frames = r.var_usize();
            let entry = match kind {
                0 => {
                    let (tl, d) = read_curve_timeline_n(r, index, frames, 1);
                    Some((Timeline::PathPosition(tl), d))
                }
                1 => {
                    let (tl, d) = read_curve_timeline_n(r, index, frames, 1);
                    Some((Timeline::PathSpacing(tl), d))
                }
                2 => {
                    let (tl, d) = read_curve_timeline_n(r, index, frames, 3);
                    Some((Timeline::PathMix(tl), d))
                }
                _ => return Animation::new(name, duration, timelines),
            };
            if let Some((timeline, d)) = entry {
                duration = duration.max(d);
                timelines.push(timeline);
            }
        }
    }

    // Physics constraint timelines: one tunable (or a reset) per inner entry.
    // The stored index is one less than the stream's (a 0 marks a global
    // timeline, wrapping to the global sentinel).
    let physics_groups = r.var_usize();
    for _ in 0..physics_groups {
        let index = r.var_usize().wrapping_sub(1);
        let count = r.var_usize();
        for _ in 0..count {
            let kind = r.byte();
            let frames = r.var_usize();
            if kind == 8 {
                let times: Vec<f32> = (0..frames).map(|_| r.float()).collect();
                duration = duration.max(times.last().copied().unwrap_or(0.0));
                timelines.push(Timeline::PhysicsReset(PhysicsResetTimeline::new(index, times)));
                continue;
            }
            let property = match kind {
                0 => PhysicsProperty::Inertia,
                1 => PhysicsProperty::Strength,
                2 => PhysicsProperty::Damping,
                4 => PhysicsProperty::Mass,
                5 => PhysicsProperty::Wind,
                6 => PhysicsProperty::Gravity,
                7 => PhysicsProperty::Mix,
                _ => return Animation::new(name, duration, timelines),
            };
            let (tl, d) = read_curve_timeline_n(r, index, frames, 1);
            duration = duration.max(d);
            timelines.push(Timeline::Physics(tl, property));
        }
    }

    // Slider timelines (Spine 4.3): SLIDER_TIME (0) and SLIDER_MIX (1), each a
    // one-value curve setting the slider's pose.
    let slider_groups = r.var_usize();
    for _ in 0..slider_groups {
        let index = r.var_usize();
        let count = r.var_usize();
        for _ in 0..count {
            let kind = r.byte();
            let frames = r.var_usize();
            let (tl, d) = read_curve_timeline_n(r, index, frames, 1);
            duration = duration.max(d);
            match kind {
                0 => timelines.push(Timeline::SliderTime(tl)),
                1 => timelines.push(Timeline::SliderMix(tl)),
                _ => return Animation::new(name, duration, timelines),
            }
        }
    }

    // Attachment timelines, nested skins -> slots -> attachments. Mesh deforms
    // are built and applied; sequence (animated attachment) timelines are
    // consumed (chine does not run sequences yet).
    let deform_skins = r.var_usize();
    for _ in 0..deform_skins {
        let skin_index = r.var_usize();
        let slots = r.var_usize();
        for _ in 0..slots {
            let slot = r.var_usize();
            let atts = r.var_usize();
            for _ in 0..atts {
                let att_name = string_ref(r, strings).unwrap_or_default();
                let kind = r.byte();
                let frames = r.var_usize();
                match kind {
                    0 => match deform_mesh_info(data, skin_index, slot, &att_name) {
                        Some((frame_len, setup, tl_skin)) => {
                            let (tl, d) = read_deform_timeline(
                                r, slot, att_name, tl_skin, setup, frame_len, frames,
                            );
                            duration = duration.max(d);
                            timelines.push(Timeline::Deform(tl));
                        }
                        // No matching mesh: consume the bytes to stay aligned.
                        None => skip_deform_timeline(r, frames),
                    },
                    1 => {
                        let count =
                            sequence_count(data, skin_index, slot, &att_name).unwrap_or(0);
                        let (tl, d) = read_sequence_timeline(r, slot, att_name, count, frames);
                        duration = duration.max(d);
                        timelines.push(Timeline::Sequence(tl));
                    }
                    _ => return Animation::new(name, duration, timelines),
                }
            }
        }
    }

    // Draw order timeline: per keyframe, a slot permutation.
    let draw_order_count = r.var_usize();
    if draw_order_count != 0 {
        let (tl, d) = read_draw_order_timeline(r, draw_order_count, data.slots.len());
        duration = duration.max(d);
        timelines.push(Timeline::DrawOrder(tl));
    }

    // Draw-order folder timelines (new in Spine 4.3): folder-scoped slot
    // reorders. chine has no folder timeline, so these are consumed to keep the
    // stream aligned.
    let folder_count = r.var_usize();
    for _ in 0..folder_count {
        let folder_slot_count = r.var_usize();
        for _ in 0..folder_slot_count {
            r.var_usize();
        }
        let key_count = r.var_usize();
        for _ in 0..key_count {
            r.float();
            let change_count = r.var_usize();
            for _ in 0..change_count {
                r.var_usize();
                r.var_usize();
            }
        }
    }

    // Event timeline: keyframes that fire named events with per-key overrides.
    let event_count = r.var_usize();
    if event_count != 0 {
        let (tl, d) = read_event_timeline(r, data, event_count);
        duration = duration.max(d);
        timelines.push(Timeline::Event(tl));
    }

    // A nonessential export appends the animation's editor color (RGBA8888).
    if nonessential {
        let _color = r.u32();
    }
    Animation::new(name, duration, timelines)
}

/// Read a one-value bone timeline (rotate / single axis). The stream gives the
/// Bezier-segment count first (curve storage), then per-frame time + value with
/// a stepped / linear / Bezier curve between frames.
fn read_bone_timeline1(r: &mut BinaryReader, bone: usize, frames: usize) -> (BoneTimeline, f32) {
    let bezier_count = r.var_usize();
    let mut tl = BoneTimeline::one_value(bone, frames, bezier_count);
    let mut bezier = 0;
    let mut duration = 0.0_f32;
    let mut time = r.float();
    let mut value = r.float();
    let last = frames.saturating_sub(1);
    for frame in 0..frames {
        tl.set_frame1(frame, time, value);
        duration = duration.max(time);
        if frame == last {
            break;
        }
        let time2 = r.float();
        let value2 = r.float();
        match r.byte() {
            1 => tl.set_stepped(frame),
            2 => {
                let cx1 = r.float();
                let cy1 = r.float();
                let cx2 = r.float();
                let cy2 = r.float();
                tl.set_bezier(
                    bezier, frame, 0, time, value, cx1, cy1, cx2, cy2, time2, value2,
                );
                bezier += 1;
            }
            _ => {}
        }
        time = time2;
        value = value2;
    }
    (tl, duration)
}

/// Read a two-value bone timeline (translate / scale / shear). The stream gives
/// the Bezier-segment count first, then per frame a time and two values, with
/// two Bezier segments per curved frame.
fn read_bone_timeline2(r: &mut BinaryReader, bone: usize, frames: usize) -> (BoneTimeline, f32) {
    let bezier_count = r.var_usize();
    let mut tl = BoneTimeline::two_value(bone, frames, bezier_count);
    let mut bezier = 0;
    let mut duration = 0.0_f32;
    let mut time = r.float();
    let mut v1 = r.float();
    let mut v2 = r.float();
    let last = frames.saturating_sub(1);
    for frame in 0..frames {
        tl.set_frame2(frame, time, v1, v2);
        duration = duration.max(time);
        if frame == last {
            break;
        }
        let time2 = r.float();
        let nv1 = r.float();
        let nv2 = r.float();
        match r.byte() {
            1 => tl.set_stepped(frame),
            2 => {
                let cx1 = r.float();
                let cy1 = r.float();
                let cx2 = r.float();
                let cy2 = r.float();
                tl.set_bezier(bezier, frame, 0, time, v1, cx1, cy1, cx2, cy2, time2, nv1);
                bezier += 1;
                let dx1 = r.float();
                let dy1 = r.float();
                let dx2 = r.float();
                let dy2 = r.float();
                tl.set_bezier(bezier, frame, 1, time, v2, dx1, dy1, dx2, dy2, time2, nv2);
                bezier += 1;
            }
            _ => {}
        }
        time = time2;
        v1 = nv1;
        v2 = nv2;
    }
    (tl, duration)
}

/// Read `n` color channels, each a `0..=255` byte scaled to `0..=1`.
fn read_color_channels(r: &mut BinaryReader, n: usize) -> Vec<f32> {
    (0..n).map(|_| f32::from(r.byte()) / 255.0).collect()
}

/// Read a slot color timeline (RGBA / RGB / two-color / alpha): per frame a time
/// and `channels` color components (each a byte), with a stepped / linear /
/// Bezier curve per channel between frames. The stream gives the Bezier-segment
/// count first.
fn read_slot_color_timeline(
    r: &mut BinaryReader,
    slot: usize,
    frames: usize,
    channels: usize,
) -> (ConstraintTimeline, f32) {
    let bezier_count = r.var_usize();
    let mut tl = ConstraintTimeline::new(slot, frames, bezier_count, channels + 1);
    let mut bezier = 0;
    let mut duration = 0.0_f32;
    let mut time = r.float();
    let mut vals = read_color_channels(r, channels);
    let last = frames.saturating_sub(1);
    for frame in 0..frames {
        tl.set_frame(frame, time, &vals);
        duration = duration.max(time);
        if frame == last {
            break;
        }
        let time2 = r.float();
        let vals2 = read_color_channels(r, channels);
        match r.byte() {
            1 => tl.set_stepped(frame),
            2 => {
                for ch in 0..channels {
                    let cx1 = r.float();
                    let cy1 = r.float();
                    let cx2 = r.float();
                    let cy2 = r.float();
                    tl.set_bezier(
                        bezier, frame, ch, time, vals[ch], cx1, cy1, cx2, cy2, time2, vals2[ch],
                    );
                    bezier += 1;
                }
            }
            _ => {}
        }
        time = time2;
        vals = vals2;
    }
    (tl, duration)
}

/// Consume one mesh-deform timeline frame: a run of changed vertices encoded as
/// a count, a start offset, then that many float offsets (count `0` means the
/// frame uses the setup vertices).
fn skip_deform_frame(r: &mut BinaryReader) {
    let end = r.var_usize();
    if end != 0 {
        let _start = r.var_usize();
        for _ in 0..end {
            r.float();
        }
    }
}

/// Consume a mesh-deform timeline (Bezier count, then per-frame time and vertex
/// offsets, with stepped / linear / Bezier curves) without building anything.
fn skip_deform_timeline(r: &mut BinaryReader, frames: usize) {
    let _bezier_count = r.var_usize();
    let last = frames.saturating_sub(1);
    r.float(); // first frame time
    skip_deform_frame(r);
    for frame in 0..frames {
        if frame == last {
            break;
        }
        r.float(); // time
        if r.byte() == 2 {
            // Bezier: four control floats.
            for _ in 0..4 {
                r.float();
            }
        }
        skip_deform_frame(r);
    }
}

/// The number of regions in a sequenced region/mesh attachment, for a sequence
/// timeline's index wrapping.
fn sequence_count(data: &SkeletonData, skin_index: usize, slot: usize, name: &str) -> Option<usize> {
    let skin = if skin_index == 0 {
        None
    } else {
        data.skins.get(skin_index - 1)
    };
    match data.attachment(slot, name, skin) {
        Some(Attachment::Region(r)) => r.sequence.as_ref().map(|s| s.count),
        Some(Attachment::Mesh(m)) => m.sequence.as_ref().map(|s| s.count),
        _ => None,
    }
}

/// Read a sequence (flipbook) timeline: per frame a time, a packed mode-and-index
/// int, and a delay. Stepped, so there is no curve data.
fn read_sequence_timeline(
    r: &mut BinaryReader,
    slot: usize,
    attachment: String,
    count: usize,
    frames: usize,
) -> (SequenceTimeline, f32) {
    let mut times = Vec::with_capacity(frames);
    let mut mode_and_index = Vec::with_capacity(frames);
    let mut delays = Vec::with_capacity(frames);
    let mut duration = 0.0_f32;
    for _ in 0..frames {
        let time = r.float();
        mode_and_index.push(r.u32());
        delays.push(r.float());
        times.push(time);
        duration = duration.max(time);
    }
    (
        SequenceTimeline::new(slot, attachment, count, times, mode_and_index, delays),
        duration,
    )
}

/// Resolve a deform timeline's mesh: the setup-pose deform length, the setup
/// vertices to add at apply time (zeros for a weighted mesh), and the
/// timeline's skin name (`None` for the default skin). Returns `None` when no
/// matching mesh is found, so the caller consumes the bytes instead.
fn deform_mesh_info(
    data: &SkeletonData,
    skin_index: usize,
    slot: usize,
    name: &str,
) -> Option<(usize, Vec<f32>, Option<String>)> {
    let skin = if skin_index == 0 {
        None
    } else {
        data.skins.get(skin_index - 1)
    };
    let Some(Attachment::Mesh(mesh)) = data.attachment(slot, name, skin) else {
        return None;
    };
    let frame_len = mesh.deform_len();
    let setup = mesh
        .setup_vertices()
        .map_or_else(|| vec![0.0; frame_len], <[f32]>::to_vec);
    Some((frame_len, setup, skin.map(|s| s.name.clone())))
}

/// Read a mesh-deform timeline into chine's relative-offset model: per frame a
/// time and a sparse run of vertex offsets (zeros elsewhere; the setup vertices
/// are added at apply time), with stepped / linear / Bezier curves. The offsets
/// are read raw, never adding the setup, which matches the JSON loader.
fn read_deform_timeline(
    r: &mut BinaryReader,
    slot: usize,
    attachment: String,
    skin: Option<String>,
    setup: Vec<f32>,
    frame_len: usize,
    frames: usize,
) -> (DeformTimeline, f32) {
    let bezier_count = r.var_usize();
    let last = frames.saturating_sub(1);
    let mut times = Vec::with_capacity(frames);
    let mut offsets = Vec::with_capacity(frames);
    let mut segments: Vec<(u8, [f32; 4])> = Vec::new();
    let mut time = r.float();
    for frame in 0..frames {
        let mut deform = vec![0.0_f32; frame_len];
        let end = r.var_usize();
        if end != 0 {
            let start = r.var_usize();
            for i in 0..end {
                let value = r.float();
                if let Some(v) = deform.get_mut(start + i) {
                    *v = value;
                }
            }
        }
        times.push(time);
        offsets.push(deform);
        if frame == last {
            break;
        }
        let time2 = r.float();
        let segment = match r.byte() {
            1 => (1, [0.0; 4]),
            2 => (2, [r.float(), r.float(), r.float(), r.float()]),
            _ => (0, [0.0; 4]),
        };
        segments.push(segment);
        time = time2;
    }
    let duration = times.last().copied().unwrap_or(0.0);
    let mut tl = DeformTimeline::new(
        slot,
        attachment,
        skin,
        setup,
        times.clone(),
        offsets,
        bezier_count,
    );
    let mut bezier = 0;
    for (frame, &(kind, c)) in segments.iter().enumerate() {
        match kind {
            1 => tl.set_stepped(frame),
            2 => {
                tl.set_bezier(
                    bezier,
                    frame,
                    0,
                    times[frame],
                    0.0,
                    c[0],
                    c[1],
                    c[2],
                    c[3],
                    times[frame + 1],
                    1.0,
                );
                bezier += 1;
            }
            _ => {}
        }
    }
    (tl, duration)
}

/// Read a curve timeline whose `channels` values are plain floats (transform,
/// path, and physics constraint timelines), with a stepped / linear / Bezier
/// curve per channel between frames.
fn read_curve_timeline_n(
    r: &mut BinaryReader,
    constraint: usize,
    frames: usize,
    channels: usize,
) -> (ConstraintTimeline, f32) {
    let bezier_count = r.var_usize();
    let mut tl = ConstraintTimeline::new(constraint, frames, bezier_count, channels + 1);
    let mut bezier = 0;
    let mut duration = 0.0_f32;
    let mut time = r.float();
    let mut vals: Vec<f32> = (0..channels).map(|_| r.float()).collect();
    let last = frames.saturating_sub(1);
    for frame in 0..frames {
        tl.set_frame(frame, time, &vals);
        duration = duration.max(time);
        if frame == last {
            break;
        }
        let time2 = r.float();
        let vals2: Vec<f32> = (0..channels).map(|_| r.float()).collect();
        match r.byte() {
            1 => tl.set_stepped(frame),
            2 => {
                for ch in 0..channels {
                    let cx1 = r.float();
                    let cy1 = r.float();
                    let cx2 = r.float();
                    let cy2 = r.float();
                    tl.set_bezier(
                        bezier, frame, ch, time, vals[ch], cx1, cy1, cx2, cy2, time2, vals2[ch],
                    );
                    bezier += 1;
                }
            }
            _ => {}
        }
        time = time2;
        vals = vals2;
    }
    (tl, duration)
}

/// Read an IK constraint timeline (Spine 4.3 flags-packed). Each frame packs mix
/// presence, softness presence, bend direction, compress, and stretch into one
/// flags byte (also carrying the following segment's curve type); mix and
/// softness are the two curved channels.
fn read_ik_constraint_timeline(
    r: &mut BinaryReader,
    constraint: usize,
    frames: usize,
) -> (ConstraintTimeline, f32) {
    let bezier_count = r.var_usize();
    let mut tl = ConstraintTimeline::new(constraint, frames, bezier_count, 6);
    let mut bezier = 0;
    let mut duration = 0.0_f32;
    let last = frames.saturating_sub(1);
    let mut flags = r.byte();
    let mut time = r.float();
    let mut mix = ik_flag_value(r, flags, 1, 2);
    let mut softness = if flags & 4 != 0 { r.float() } else { 0.0 };
    for frame in 0..frames {
        let bend = if flags & 8 != 0 { 1.0 } else { -1.0 };
        let compress = f32::from(flags & 16 != 0);
        let stretch = f32::from(flags & 32 != 0);
        tl.set_frame(frame, time, &[mix, softness, bend, compress, stretch]);
        duration = duration.max(time);
        if frame == last {
            break;
        }
        flags = r.byte();
        let time2 = r.float();
        let mix2 = ik_flag_value(r, flags, 1, 2);
        let softness2 = if flags & 4 != 0 { r.float() } else { 0.0 };
        if flags & 64 != 0 {
            tl.set_stepped(frame);
        } else if flags & 128 != 0 {
            let (cx1, cy1, cx2, cy2) = (r.float(), r.float(), r.float(), r.float());
            tl.set_bezier(bezier, frame, 0, time, mix, cx1, cy1, cx2, cy2, time2, mix2);
            bezier += 1;
            let (dx1, dy1, dx2, dy2) = (r.float(), r.float(), r.float(), r.float());
            tl.set_bezier(
                bezier, frame, 1, time, softness, dx1, dy1, dx2, dy2, time2, softness2,
            );
            bezier += 1;
        }
        time = time2;
        mix = mix2;
        softness = softness2;
    }
    (tl, duration)
}

/// Decode an IK timeline mix from its flags: absent (`0`), a default of `1`, or
/// an explicit float.
fn ik_flag_value(r: &mut BinaryReader, flags: u8, present: u8, explicit: u8) -> f32 {
    if flags & present != 0 {
        if flags & explicit != 0 {
            r.float()
        } else {
            1.0
        }
    } else {
        0.0
    }
}

/// Read a slot attachment-swap timeline: per frame a time and the attachment
/// name shown (`None` hides the slot). Stepped, so there is no curve data.
fn read_slot_attachment_timeline(
    r: &mut BinaryReader,
    slot: usize,
    frames: usize,
    strings: &[String],
) -> (AttachmentTimeline, f32) {
    let mut times = Vec::with_capacity(frames);
    let mut names = Vec::with_capacity(frames);
    let mut duration = 0.0_f32;
    for _ in 0..frames {
        let time = r.float();
        names.push(string_ref(r, strings));
        times.push(time);
        duration = duration.max(time);
    }
    (AttachmentTimeline::new(slot, times, names), duration)
}

/// Read a draw-order timeline: per frame a time and a set of slot moves (each a
/// slot index and an offset), resolved into a full slot permutation. A frame
/// with no moves keeps the setup order.
fn read_draw_order_timeline(
    r: &mut BinaryReader,
    frames: usize,
    slot_count: usize,
) -> (DrawOrderTimeline, f32) {
    let mut times = Vec::with_capacity(frames);
    let mut orders = Vec::with_capacity(frames);
    let mut duration = 0.0_f32;
    for _ in 0..frames {
        let time = r.float();
        let change_count = r.var_usize();
        let mut offsets: Vec<(usize, i32)> = Vec::with_capacity(change_count);
        for _ in 0..change_count {
            let slot = r.var_usize();
            let offset = r.var_usize() as i32;
            offsets.push((slot, offset));
        }
        let order = if offsets.is_empty() {
            (0..slot_count).collect()
        } else {
            compute_draw_order(slot_count, &mut offsets)
        };
        times.push(time);
        orders.push(order);
        duration = duration.max(time);
    }
    (DrawOrderTimeline::new(times, orders), duration)
}

/// Read an event timeline: per frame a time and the fired event. Each event's
/// base values come from the skeleton's [`EventData`] (by index) and are
/// overridden by the keyframe; volume / balance are only stored when the event
/// has an audio path.
fn read_event_timeline(
    r: &mut BinaryReader,
    data: &SkeletonData,
    frames: usize,
) -> (EventTimeline, f32) {
    let mut times = Vec::with_capacity(frames);
    let mut events = Vec::with_capacity(frames);
    let mut duration = 0.0_f32;
    for _ in 0..frames {
        let time = r.float();
        let index = r.var_usize();
        let (name, def_string, has_audio, def_volume, def_balance) = match data.events.get(index) {
            Some(e) => (
                e.name.clone(),
                e.string_value.clone(),
                e.audio_path.is_some(),
                e.volume,
                e.balance,
            ),
            None => (String::new(), String::new(), false, 1.0, 0.0),
        };
        let int_value = r.var_int();
        let float_value = r.float();
        let string_value = r.string().unwrap_or(def_string);
        let (volume, balance) = if has_audio {
            (r.float(), r.float())
        } else {
            (def_volume, def_balance)
        };
        events.push(Event {
            name,
            time,
            int_value,
            float_value,
            string_value,
            volume,
            balance,
        });
        times.push(time);
        duration = duration.max(time);
    }
    (EventTimeline::new(times, events), duration)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_var_uint_multi_byte() {
        // 300 = 0b1_0010_1100 -> low 7 bits 0x2C with continuation, then 0x02.
        let mut r = BinaryReader::new(&[0xAC, 0x02]);
        assert_eq!(r.var_uint(), 300);
        assert!(!r.overran());
    }

    #[test]
    fn reads_float_big_endian() {
        let bytes = 1.0_f32.to_be_bytes();
        let mut r = BinaryReader::new(&bytes);
        assert!((r.float() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn reads_length_prefixed_strings() {
        // 0 -> None, 1 -> "", else n-1 bytes.
        let mut r = BinaryReader::new(&[0x00, 0x01, 0x03, b'h', b'i']);
        assert_eq!(r.string(), None);
        assert_eq!(r.string(), Some(String::new()));
        assert_eq!(r.string(), Some("hi".to_string()));
    }

    #[test]
    fn flags_truncation() {
        let mut r = BinaryReader::new(&[0x05, b'h', b'i']); // claims 4 bytes, has 2
        let _ = r.string();
        assert!(r.overran());
    }

    fn enc_str(out: &mut Vec<u8>, s: &str) {
        out.push((s.len() + 1) as u8); // short names only
        out.extend_from_slice(s.as_bytes());
    }

    #[test]
    fn parses_header_and_bones() {
        let mut b = Vec::new();
        b.extend_from_slice(&[0; 8]); // hash
        enc_str(&mut b, "4.3.00"); // version
        for v in [0.0_f32, 0.0, 200.0, 300.0, 1.0] {
            b.extend_from_slice(&v.to_be_bytes()); // x, y, width, height, referenceScale
        }
        b.push(0); // nonessential = false
        b.push(0); // string table count = 0
        b.push(1); // bone count = 1
        enc_str(&mut b, "root");
        // rotation, x, y, scaleX, scaleY, shearX, shearY, length
        for v in [0.0_f32, 10.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0] {
            b.extend_from_slice(&v.to_be_bytes());
        }
        b.push(3); // inherit = NoScale
        b.push(0); // skinRequired = false
        b.push(0); // slot count = 0
        b.push(0); // constraint count = 0
        b.push(0); // default skin slot count = 0
        b.push(0); // named skin count = 0
        b.push(0); // event count = 0
        b.push(0); // animation count = 0

        let data = from_binary(&b).unwrap();
        assert_eq!(data.spine_version.as_deref(), Some("4.3.00"));
        assert!((data.reference_scale - 1.0).abs() < 1e-6);
        assert_eq!(data.bones.len(), 1);
        let root = &data.bones[0];
        assert_eq!(root.name, "root");
        assert_eq!(root.parent, None);
        assert!((root.position.x - 10.0).abs() < 1e-6);
        assert!((root.scale.x - 1.0).abs() < 1e-6);
        assert_eq!(root.inherit, Inherit::NoScale);
    }

    // Validates the parser against a real Spine 4.3 `.skel` when the local
    // fixture is present (it is not committed); skips cleanly otherwise.
    #[test]
    fn parses_real_skel_header_and_bones() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/data/Spine.skel");
        let Ok(bytes) = std::fs::read(path) else {
            return;
        };
        let data = from_binary(&bytes).unwrap();
        assert_eq!(data.spine_version.as_deref(), Some("4.3.13"));
        assert_eq!(data.bones.len(), 40);
        assert_eq!(data.bones[0].name, "root");
        assert_eq!(data.bones[0].parent, None);
        assert_eq!(data.bones[1].name, "skeleton-control");
        assert_eq!(data.bones[1].parent, Some(0));
        // Every bone has a non-empty name and a valid parent index.
        for (i, b) in data.bones.iter().enumerate() {
            assert!(!b.name.is_empty(), "bone {i} has empty name");
            if let Some(p) = b.parent {
                assert!(p < data.bones.len(), "bone {i} bad parent {p}");
            }
        }

        // Slots: draw order back-to-front, each on a valid bone.
        assert_eq!(data.slots.len(), 32);
        assert_eq!(data.slots[0].name, "foot-back");
        for s in &data.slots {
            assert!(!s.name.is_empty());
            assert!(
                s.bone < data.bones.len(),
                "slot {} bad bone {}",
                s.name,
                s.bone
            );
        }
        // Setup attachments resolve through the string table.
        assert!(data.slots.iter().any(|s| s.attachment.is_some()));

        // IK constraints (the rig's three foot/leg IK chains).
        assert_eq!(data.ik_constraints.len(), 3);
        let ik = &data.ik_constraints[0];
        assert_eq!(ik.name, "leg-front-IK");
        assert_eq!(ik.bones, vec![3, 4]);
        assert_eq!(ik.target, 37);
        for c in &data.ik_constraints {
            assert!((c.mix - 1.0).abs() < 1e-6, "{} mix={}", c.name, c.mix);
            assert!(c.target < data.bones.len());
            assert!(c.bones.iter().all(|&b| b < data.bones.len()));
        }

        // Default skin: one attachment per slot, meshes with valid triangles.
        assert_eq!(data.default_skin.iter().count(), 32);
        assert!(data.skins.is_empty());
        let mesh = data
            .default_skin
            .iter()
            .find_map(|(_, _, a)| match a {
                Attachment::Mesh(m) => Some(m),
                _ => None,
            })
            .expect("a mesh attachment");
        assert!(mesh.vertex_count() > 0);
        assert!(mesh
            .triangles
            .iter()
            .all(|&t| (t as usize) < mesh.vertex_count()));

        // One animation (empty in this WIP rig) named "animation".
        assert_eq!(data.animations.len(), 1);
        assert_eq!(data.animations[0].name(), "animation");
    }

    // Validates the binary loader against a complete Spine 4.3 project (the
    // diamond rig) when the local fixture is present; skips otherwise. Its
    // non-bone animation timelines are still being added, so this asserts the
    // structural pieces that are wired up: bones, slots, the new 4.3 slider
    // constraint, and the first (bone-only) animation parsing in full.
    #[test]
    fn parses_diamond_rig() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/data/diamond-pro.skel");
        let Ok(bytes) = std::fs::read(path) else {
            return;
        };
        let data = from_binary(&bytes).expect("diamond parses without overrun");
        assert_eq!(data.bones.len(), 8);
        assert_eq!(data.slots.len(), 30);

        // The new 4.3 slider constraint ("rotation") drives a bone property.
        assert_eq!(data.sliders.len(), 1);
        let slider = &data.sliders[0];
        assert_eq!(slider.name, "rotation");
        assert!(slider.bone.is_some());
        assert!(slider.property.is_some());
        // The slider scrubs the like-named "rotation" animation.
        let scrubbed = slider.animation_index.expect("slider animation index");
        assert_eq!(data.animations[scrubbed].name(), "rotation");

        // All eight animations parse end to end; a misaligned parse would
        // surface as a garbage or truncated name. "appear" is bone-only;
        // "disappear" exercises slot color, attachment, slider, deform, and
        // sequence timelines.
        let names: Vec<&str> = data.animations.iter().map(|a| a.name()).collect();
        assert_eq!(
            names,
            [
                "appear",
                "disappear",
                "idle-rotating",
                "idle-rotating-alt-shape",
                "idle-still",
                "rotation",
                "size-changing-rotation",
                "size-changing-rotation-perspective",
            ]
        );
        assert!(data.animations[0].duration() > 0.0);
        assert!(data.animations[1].duration() > 0.0);
    }

    // The "disappear" animation deforms the diamond mesh; loading it from binary
    // must build a real deform timeline (not silently fall back to consuming the
    // bytes). Playing it populates a slot's deform buffer.
    #[test]
    fn diamond_deform_timeline_applies() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/data/diamond-pro.skel");
        let Ok(bytes) = std::fs::read(path) else {
            return;
        };
        let data = from_binary(&bytes).unwrap();
        let slot_count = data.slots.len();
        let anims: Vec<_> = data.animations.clone();
        let arc = std::sync::Arc::new(data);
        // A deform applies only while its slot shows the keyed attachment and
        // time is past the first keyframe. Sample every animation across time
        // and confirm at least one populates a deform buffer.
        let deformed = anims.iter().any(|anim| {
            let dur = anim.duration();
            (0..=10).any(|i| {
                let t = dur * (i as f32 / 10.0);
                let mut sk = crate::skel::Skeleton::new(arc.clone());
                let mut state = crate::anim::AnimationState::new();
                state.set_animation(anim.clone(), false);
                state.update(t);
                sk.set_slots_to_setup_pose();
                state.apply(&mut sk);
                (0..slot_count).any(|s| sk.slot(s).is_some_and(|sl| !sl.deform.is_empty()))
            })
        });
        assert!(deformed, "expected a diamond animation to deform a mesh");
    }

    // Exercises the slider constraint on the real rig: playing an animation and
    // updating world transforms runs the slider, which scrubs the "rotation"
    // animation from its bone. This must pose the rig without panicking.
    #[test]
    fn diamond_slider_runs() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/data/diamond-pro.skel");
        let Ok(bytes) = std::fs::read(path) else {
            return;
        };
        let data = from_binary(&bytes).unwrap();
        let anim = data.find_animation("idle-rotating").unwrap().clone();
        let mut sk = crate::skel::Skeleton::new(std::sync::Arc::new(data));
        let mut state = crate::anim::AnimationState::new();
        state.set_animation(anim, false);
        state.update(0.5);
        sk.set_slots_to_setup_pose();
        state.apply(&mut sk);
        sk.update_world_transform();
        assert!(sk.bone(0).is_some());
    }

    #[test]
    fn reads_a_one_value_bone_timeline() {
        // Bezier count 0, then two rotate frames (0,0) and (1,90) with a linear
        // curve between them.
        let mut b = Vec::new();
        b.push(0); // bezier-segment count = 0
        for v in [0.0_f32, 0.0, 1.0, 90.0] {
            b.extend_from_slice(&v.to_be_bytes());
        }
        b.push(0); // curve = linear
        let mut r = BinaryReader::new(&b);
        let (_tl, duration) = read_bone_timeline1(&mut r, 0, 2);
        assert!((duration - 1.0).abs() < 1e-6);
        assert!(!r.overran());
    }

    #[test]
    fn reads_an_event_timeline() {
        // A skeleton with a root bone and one event named "footstep".
        let data = SkeletonData {
            bones: vec![BoneData {
                index: 0,
                name: "root".into(),
                ..Default::default()
            }],
            events: vec![EventData {
                name: "footstep".into(),
                int_value: 5,
                float_value: 0.0,
                string_value: String::new(),
                audio_path: None,
                volume: 1.0,
                balance: 0.0,
            }],
            ..Default::default()
        };
        // One keyframe at time 0.5: event index 0, int override 7, float 1.5,
        // no string override, no audio (so no volume / balance bytes).
        let mut b = Vec::new();
        b.extend_from_slice(&0.5_f32.to_be_bytes());
        b.push(0); // event index
        b.push(14); // int value 7 as a zig-zag var-int
        b.extend_from_slice(&1.5_f32.to_be_bytes());
        b.push(0); // string value = None
        let mut r = BinaryReader::new(&b);
        let (tl, dur) = read_event_timeline(&mut r, &data, 1);
        assert!((dur - 0.5).abs() < 1e-6);
        assert!(!r.overran());

        // Playing past the keyframe fires "footstep" with the keyframe int (7).
        let anim = Animation::new("walk", dur, vec![Timeline::Event(tl)]);
        let mut sk = crate::skel::Skeleton::new(std::sync::Arc::new(data));
        let mut state = crate::anim::AnimationState::new();
        state.set_animation(std::sync::Arc::new(anim), false);
        state.update(1.0);
        state.apply(&mut sk);
        assert_eq!(sk.events().len(), 1);
        assert_eq!(sk.events()[0].name, "footstep");
        assert_eq!(sk.events()[0].int_value, 7);
    }

    #[test]
    fn reads_a_draw_order_timeline() {
        fn slot(index: usize, name: &str) -> SlotData {
            SlotData {
                index,
                name: name.into(),
                bone: 0,
                color: Color::WHITE,
                dark_color: None,
                attachment: None,
                blend: BlendMode::Normal,
            }
        }
        // Three slots; one keyframe at 0.5 moves slot 0 back by 2 (to the end).
        let data = SkeletonData {
            bones: vec![BoneData {
                index: 0,
                name: "root".into(),
                ..Default::default()
            }],
            slots: vec![slot(0, "a"), slot(1, "b"), slot(2, "c")],
            ..Default::default()
        };
        let mut b = Vec::new();
        b.extend_from_slice(&0.5_f32.to_be_bytes()); // time
        b.push(1); // change count
        b.push(0); // slot index
        b.push(2); // offset
        let mut r = BinaryReader::new(&b);
        let (tl, dur) = read_draw_order_timeline(&mut r, 1, 3);
        assert!((dur - 0.5).abs() < 1e-6);
        assert!(!r.overran());

        let anim = Animation::new("reorder", dur, vec![Timeline::DrawOrder(tl)]);
        let mut sk = crate::skel::Skeleton::new(std::sync::Arc::new(data));
        let mut state = crate::anim::AnimationState::new();
        state.set_animation(std::sync::Arc::new(anim), false);
        state.update(1.0);
        state.apply(&mut sk);
        // Slot 0 moved to the end: the order becomes b, c, a.
        assert_eq!(sk.draw_order(), &[1, 2, 0]);
    }

    #[test]
    fn reads_a_slot_alpha_timeline() {
        let data = SkeletonData {
            bones: vec![BoneData {
                index: 0,
                name: "root".into(),
                ..Default::default()
            }],
            slots: vec![SlotData {
                index: 0,
                name: "s".into(),
                bone: 0,
                color: Color::WHITE,
                dark_color: None,
                attachment: None,
                blend: BlendMode::Normal,
            }],
            ..Default::default()
        };
        // Alpha 1.0 -> 0.0 over time 0 -> 1, linear (one byte per alpha channel).
        let mut b = Vec::new();
        b.push(0); // bezier-segment count
        b.extend_from_slice(&0.0_f32.to_be_bytes());
        b.push(255); // alpha 1.0
        b.extend_from_slice(&1.0_f32.to_be_bytes());
        b.push(0); // alpha 0.0
        b.push(0); // linear curve
        let mut r = BinaryReader::new(&b);
        let (tl, dur) = read_slot_color_timeline(&mut r, 0, 2, 1);
        assert!((dur - 1.0).abs() < 1e-6);
        assert!(!r.overran());

        let anim = Animation::new("fade", dur, vec![Timeline::SlotAlpha(tl)]);
        let mut sk = crate::skel::Skeleton::new(std::sync::Arc::new(data));
        let mut state = crate::anim::AnimationState::new();
        state.set_animation(std::sync::Arc::new(anim), false);
        state.update(0.5);
        sk.set_slots_to_setup_pose();
        state.apply(&mut sk);
        // Half-way the alpha is ~0.5; red stays at the setup white.
        let c = sk.slot(0).unwrap().color;
        assert!((c.a - 0.5).abs() < 0.05, "alpha={}", c.a);
        assert!((c.r - 1.0).abs() < 1e-3, "r={}", c.r);
    }

    #[test]
    fn reads_an_ik_constraint_timeline() {
        use crate::constraint::ik::IkConstraintData;
        use crate::constraint::ScaleYMode;

        // Mix 0 at time 0, mix 1 at time 1 (flags-packed, linear).
        let mut b = Vec::new();
        b.push(0); // bezier-segment count
        b.push(0); // flags: mix absent (0)
        b.extend_from_slice(&0.0_f32.to_be_bytes()); // time 0
        b.push(1); // flags: mix present, default 1, linear
        b.extend_from_slice(&1.0_f32.to_be_bytes()); // time 1
        let mut r = BinaryReader::new(&b);
        let (tl, dur) = read_ik_constraint_timeline(&mut r, 0, 2);
        assert!((dur - 1.0).abs() < 1e-6);
        assert!(!r.overran());

        // A one-bone IK aiming at a target above it; the timeline drives its mix.
        let data = SkeletonData {
            bones: vec![
                BoneData {
                    index: 0,
                    name: "root".into(),
                    ..Default::default()
                },
                BoneData {
                    index: 1,
                    name: "aim".into(),
                    parent: Some(0),
                    length: 10.0,
                    ..Default::default()
                },
                BoneData {
                    index: 2,
                    name: "target".into(),
                    parent: Some(0),
                    position: Vec2::new(0.0, 10.0),
                    ..Default::default()
                },
            ],
            ik_constraints: vec![IkConstraintData {
                name: "aim-ik".into(),
                order: 0,
                bones: vec![1],
                target: 2,
                scale_y_mode: ScaleYMode::None,
                mix: 1.0,
                softness: 0.0,
                bend_direction: 1,
                compress: false,
                stretch: false,
            }],
            ..Default::default()
        };
        let anim = Animation::new("ik", dur, vec![Timeline::Ik(tl)]);
        let mut sk = crate::skel::Skeleton::new(std::sync::Arc::new(data));
        let mut state = crate::anim::AnimationState::new();
        state.set_animation(std::sync::Arc::new(anim), false);
        // At time 0 the keyed mix is 0, so the aim keeps its setup +x heading
        // rather than rotating up toward the target (which mix 1 would do).
        state.update(0.0);
        state.apply(&mut sk);
        sk.update_world_transform();
        let aim = sk.bone(1).unwrap();
        assert!((aim.a() - 1.0).abs() < 0.1, "a={}", aim.a());
        assert!(aim.c().abs() < 0.1, "c={}", aim.c());
    }

    #[test]
    fn reads_a_transform_mix_timeline() {
        // Two frames, six float mix channels, linear.
        let mut b = Vec::new();
        b.push(0); // bezier-segment count
        b.extend_from_slice(&0.0_f32.to_be_bytes());
        for _ in 0..6 {
            b.extend_from_slice(&1.0_f32.to_be_bytes());
        }
        b.extend_from_slice(&1.0_f32.to_be_bytes());
        for _ in 0..6 {
            b.extend_from_slice(&0.5_f32.to_be_bytes());
        }
        b.push(0); // linear curve
        let mut r = BinaryReader::new(&b);
        let (_tl, dur) = read_curve_timeline_n(&mut r, 0, 2, 6);
        assert!((dur - 1.0).abs() < 1e-6);
        assert!(!r.overran());
    }

    #[test]
    fn parses_a_path_constraint() {
        // bone count 1, bone 0, slot 0, flags (position/spacing Percent, rotate
        // Chain).
        let mut b = vec![1_u8, 0, 0, 26];
        for v in [0.5_f32, 10.0, 1.0, 1.0, 1.0] {
            b.extend_from_slice(&v.to_be_bytes());
        }
        let mut r = BinaryReader::new(&b);
        let pc = parse_path(&mut r, "path".into(), 2);
        assert!(!r.overran());
        assert_eq!(pc.slot, 0);
        assert_eq!(pc.position_mode, PositionMode::Percent);
        assert_eq!(pc.spacing_mode, SpacingMode::Percent);
        assert_eq!(pc.rotate_mode, RotateMode::Chain);
        assert!((pc.position - 0.5).abs() < 1e-6);
        assert!((pc.spacing - 10.0).abs() < 1e-6);
    }

    #[test]
    fn parses_a_physics_constraint() {
        // bone 0, flags (none), step divisor 30 -> 1/30.
        let mut b = vec![0_u8, 0, 30];
        for v in [0.5_f32, 1.0, 0.9, 0.0, -10.0] {
            b.extend_from_slice(&v.to_be_bytes()); // inertia, strength, damping, wind, gravity
        }
        b.push(0); // global flags (mix defaults to 1)
        let mut r = BinaryReader::new(&b);
        let pc = parse_physics(&mut r, "phys".into(), 3);
        assert!(!r.overran());
        assert_eq!(pc.bone, 0);
        assert!((pc.step - 1.0 / 30.0).abs() < 1e-6);
        assert!((pc.inertia - 0.5).abs() < 1e-6);
        assert!((pc.gravity + 10.0).abs() < 1e-6);
        assert!((pc.mix - 1.0).abs() < 1e-6);
        assert!((pc.limit - 5000.0).abs() < 1e-3);
        assert!((pc.mass_inverse - 1.0).abs() < 1e-6);
    }

    #[test]
    fn parses_a_transform_constraint() {
        // 1 bone, source 0, no property mappings, no offsets, default mixes.
        let b = vec![1_u8, 0, 0, 0, 0, 0];
        let mut r = BinaryReader::new(&b);
        let tc = parse_transform(&mut r, "tf".into(), 2);
        assert!(!r.overran());
        assert_eq!(tc.bones, vec![0]);
        assert_eq!(tc.source, 0);
        assert!(tc.properties.is_empty());
        assert!((tc.mix_rotate - 1.0).abs() < 1e-6);
        assert!(tc.offsets[0].abs() < 1e-6);
    }

    #[test]
    fn parses_a_transform_constraint_with_a_property() {
        // flags 32 -> one source property; rotation -> rotation, scale 2.
        let mut b = vec![1_u8, 0, 0, 32];
        b.push(0); // from property: Rotate
        b.extend_from_slice(&0.5_f32.to_be_bytes()); // from offset
        b.push(1); // to count
        b.push(0); // to property: Rotate
        b.extend_from_slice(&0.0_f32.to_be_bytes()); // to offset
        b.extend_from_slice(&1.0_f32.to_be_bytes()); // to max
        b.extend_from_slice(&2.0_f32.to_be_bytes()); // to scale
        b.push(0); // offset flags
        b.push(0); // mix flags
        let mut r = BinaryReader::new(&b);
        let tc = parse_transform(&mut r, "tf".into(), 2);
        assert!(!r.overran());
        assert_eq!(tc.properties.len(), 1);
        assert_eq!(tc.properties[0].property, FromProp::Rotate);
        assert!((tc.properties[0].offset - 0.5).abs() < 1e-6);
        assert_eq!(tc.properties[0].to.len(), 1);
        assert_eq!(tc.properties[0].to[0].property, ToProp::Rotate);
        assert!((tc.properties[0].to[0].scale - 2.0).abs() < 1e-6);
    }

    // Binds the diamond rig against its atlas and confirms a sequenced
    // attachment resolved its frames to distinct UVs (the per-frame regions
    // were found, i.e. the frame path names match the atlas). Skips if the
    // local fixtures are absent.
    #[test]
    fn diamond_sequence_binds_distinct_frames() {
        let skel = concat!(env!("CARGO_MANIFEST_DIR"), "/data/diamond-pro.skel");
        let atlas = concat!(env!("CARGO_MANIFEST_DIR"), "/data/diamond-pro.atlas");
        let (Ok(bytes), Ok(atlas_text)) = (std::fs::read(skel), std::fs::read_to_string(atlas))
        else {
            return;
        };
        let mut data = from_binary(&bytes).unwrap();
        let atlas = crate::atlas::Atlas::parse(&atlas_text);
        crate::render::bind_atlas(&mut data, &atlas);

        let distinct = data.default_skin.iter().any(|(_, _, att)| {
            let seq = match att {
                Attachment::Region(r) => r.sequence.as_ref(),
                Attachment::Mesh(m) => m.sequence.as_ref(),
                _ => None,
            };
            seq.and_then(|s| Some((s.frame(0)?.0.to_vec(), s.frame(1)?.0.to_vec())))
                .is_some_and(|(a, b)| a != b)
        });
        assert!(distinct, "a sequenced attachment should bind distinct frame UVs");
    }

    #[test]
    fn applies_a_sequence_timeline() {
        use crate::anim::SequenceTimeline;

        // A slot showing a four-region sequence "seq". One looping keyframe at
        // time 0 (mode loop = 2, index 0) with a 0.1s delay: at 0.25s the index
        // advances by floor(0.25 / 0.1) = 2.
        let data = SkeletonData {
            bones: vec![BoneData {
                index: 0,
                name: "root".into(),
                ..Default::default()
            }],
            slots: vec![SlotData {
                index: 0,
                name: "s".into(),
                bone: 0,
                color: Color::WHITE,
                dark_color: None,
                attachment: Some("seq".into()),
                blend: BlendMode::Normal,
            }],
            ..Default::default()
        };
        let tl = SequenceTimeline::new(0, "seq".into(), 4, vec![0.0], vec![2_u32], vec![0.1]);
        let anim = Animation::new("flip", 1.0, vec![Timeline::Sequence(tl)]);
        let mut sk = crate::skel::Skeleton::new(std::sync::Arc::new(data));
        let mut state = crate::anim::AnimationState::new();
        state.set_animation(std::sync::Arc::new(anim), false);
        state.update(0.25);
        state.apply(&mut sk);
        assert_eq!(sk.slot(0).unwrap().sequence_index, 2);
    }
}
