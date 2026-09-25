//! The byte-stream cursor for Spine's binary `.skel` format.
//!
//! [`BinaryReader`] provides the format's primitive reads (big-endian ints and
//! floats, variable-length integers, length-prefixed strings) over a byte
//! slice. The primitives are stable across Spine 4.x. The section layouts that
//! consume them live in the parent module.

use super::BinaryError;

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
    /// by [`Self::overran`]. Later errors do not overwrite the first.
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
                let end = self.pos.saturating_add(n - 1);
                let Some(bytes) = self.data.get(self.pos..end) else {
                    self.overran = true;
                    self.pos = self.data.len();
                    return Some(String::new());
                };
                let s = String::from_utf8_lossy(bytes).into_owned();
                self.pos = end;
                Some(s)
            }
        }
    }
}
