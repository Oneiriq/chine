//! WebAssembly + WebGL2 web-component runtime for the [`chine`] Spine 4.3
//! runtime.
//!
//! This crate compiles to WebAssembly and renders a chine-posed skeleton to an
//! HTML `<canvas>` through WebGL2, driven by a drop-in custom element. chine
//! itself stays renderer-agnostic; all the web/GPU code lives here.
#![warn(missing_docs)]
#![warn(clippy::all)]

use wasm_bindgen::prelude::*;

/// Parse a binary `.skel` and report its bone count.
///
/// A bindings sanity check that exercises the chine-to-WASM boundary; the real
/// load / pose / render API follows.
#[wasm_bindgen]
#[must_use]
pub fn skel_bone_count(bytes: &[u8]) -> usize {
    chine::binary::from_binary(bytes).map_or(0, |data| data.bones.len())
}
