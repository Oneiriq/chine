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

use glam::Vec2;

use crate::attach::{
    Attachment, BoundingBoxAttachment, ClippingAttachment, LinkedMeshAttachment, MeshAttachment,
    MeshVertices, PathAttachment, PointAttachment, RegionAttachment,
};
use crate::constraint::ik::IkConstraintData;
use crate::constraint::ScaleYMode;
use crate::data::{BlendMode, BoneData, Color, Inherit, SkeletonData, SlotData};
use crate::event::EventData;
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
/// The header, bone, slot, constraint, skin (attachment), and event sections are
/// read; animations are added as the loader grows. A truncated stream returns
/// [`BinaryError::Truncated`].
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

    // IK constraints. Spine 4.3 packs the flags into one byte and stores only
    // non-default mix/softness; there is no explicit order (it is the read
    // order across all constraint types).
    // Constraints are one list; each begins with a name and a type byte
    // (0 = IK). This WIP rig has only IK constraints.
    let constraint_count = r.var_usize();
    for order in 0..constraint_count {
        let name = r.string().unwrap_or_default();
        let kind = r.byte();
        match kind {
            0 => data.ik_constraints.push(parse_ik(&mut r, name, order)),
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
            read_sequence(r, flags & 64 != 0);
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
            read_sequence(r, flags & 64 != 0);
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
            Some(Attachment::Mesh(m))
        }
        3 => {
            let path = if flags & 16 != 0 {
                string_ref(r, strings).unwrap_or_else(|| name.clone())
            } else {
                name.clone()
            };
            let color = read_att_color(r, flags & 32 != 0);
            read_sequence(r, flags & 64 != 0);
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
fn read_sequence(r: &mut BinaryReader, present: bool) {
    if present {
        for _ in 0..4 {
            r.var_usize();
        }
    }
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
    }
}
