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

use crate::data::{BoneData, Inherit, SkeletonData};

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
/// The header and bone sections are read; further sections are added as the
/// loader grows. A truncated stream returns [`BinaryError::Truncated`].
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
        let inherit = inherit_from(r.var_usize());
        let _skin_required = r.bool();
        if nonessential {
            let _color = r.u32();
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

    if r.overran() {
        return Err(BinaryError::Truncated);
    }
    Ok(data)
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
        b.push(1); // bone count = 1
        enc_str(&mut b, "root");
        // rotation, x, y, scaleX, scaleY, shearX, shearY, length
        for v in [0.0_f32, 10.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0] {
            b.extend_from_slice(&v.to_be_bytes());
        }
        b.push(3); // inherit = NoScale
        b.push(0); // skinRequired = false

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
}
