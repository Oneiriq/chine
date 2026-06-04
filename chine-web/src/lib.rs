//! WebAssembly + WebGL2 web-component runtime for the [`chine`] Spine 4.3
//! runtime.
//!
//! This crate compiles to WebAssembly and renders a chine-posed skeleton to an
//! HTML `<canvas>` through WebGL2, driven by a drop-in custom element (see
//! `js/chine-spine.js`). chine itself stays renderer-agnostic; all the web and
//! GPU code lives here.
//!
//! The [`WebSpine`] type is the JS-facing API: construct it from a skeleton
//! (`.skel` or `.json`) and atlas, register each atlas-page image, then call
//! [`WebSpine::frame`] every animation frame.
#![warn(missing_docs)]
#![warn(clippy::all)]

mod renderer;

use std::sync::Arc;

use wasm_bindgen::prelude::*;
use web_sys::{HtmlCanvasElement, HtmlImageElement};

use chine::anim::AnimationState;
use chine::atlas::Atlas;
use chine::render::{bind_atlas, render_into, RenderCommand};
use chine::skel::Skeleton;

use renderer::GlRenderer;

/// A loaded, animatable Spine skeleton bound to a WebGL2 canvas.
///
/// Build with [`WebSpine::from_binary`] or [`WebSpine::from_json`], register the
/// atlas-page images with [`WebSpine::add_page`] (in the order given by
/// [`WebSpine::page_names`]), then drive it each frame with [`WebSpine::frame`].
#[wasm_bindgen]
pub struct WebSpine {
    skeleton: Skeleton,
    state: AnimationState,
    renderer: GlRenderer,
    commands: Vec<RenderCommand>,
    page_names: Vec<String>,
    page_pma: Vec<bool>,
    /// Setup-pose fit: world-space center and half-extents, used to auto-fit
    /// the skeleton to the canvas.
    fit: (f32, f32, f32, f32),
}

#[wasm_bindgen]
impl WebSpine {
    /// Load from a binary `.skel` export plus its atlas text, rendering to
    /// `canvas`.
    ///
    /// # Errors
    /// Returns a JS error if the skeleton is malformed or WebGL2 is unavailable.
    pub fn from_binary(
        canvas: &HtmlCanvasElement,
        skeleton: &[u8],
        atlas_text: &str,
    ) -> Result<WebSpine, JsValue> {
        let mut data =
            chine::binary::from_binary(skeleton).map_err(|e| JsValue::from_str(&e.to_string()))?;
        Self::assemble(canvas, &mut data, atlas_text)
    }

    /// Load from a JSON skeleton export plus its atlas text, rendering to
    /// `canvas`.
    ///
    /// # Errors
    /// Returns a JS error if the JSON is malformed or WebGL2 is unavailable.
    pub fn from_json(
        canvas: &HtmlCanvasElement,
        json: &str,
        atlas_text: &str,
    ) -> Result<WebSpine, JsValue> {
        let mut data =
            chine::load::from_json(json).map_err(|e| JsValue::from_str(&e.to_string()))?;
        Self::assemble(canvas, &mut data, atlas_text)
    }

    /// The atlas page image names, in order. The host fetches each image and
    /// passes it back through [`WebSpine::add_page`].
    #[must_use]
    pub fn page_names(&self) -> Vec<String> {
        self.page_names.clone()
    }

    /// Register the next atlas-page image as a texture. Call once per page, in
    /// the order returned by [`WebSpine::page_names`].
    ///
    /// # Errors
    /// Returns a JS error if the texture cannot be uploaded.
    pub fn add_page(&mut self, image: &HtmlImageElement) -> Result<(), JsValue> {
        let pma = self
            .page_pma
            .get(self.renderer.page_count())
            .copied()
            .unwrap_or(false);
        self.renderer
            .add_page(image, pma)
            .map_err(|e| JsValue::from_str(&e))
    }

    /// Play the named animation, looping or not. Unknown names are ignored.
    pub fn set_animation(&mut self, name: &str, looping: bool) {
        if let Some(anim) = self.skeleton.data().find_animation(name) {
            self.state.set_animation(anim.clone(), looping);
        }
    }

    /// The animation names available on this skeleton.
    #[must_use]
    pub fn animation_names(&self) -> Vec<String> {
        self.skeleton
            .data()
            .animations
            .iter()
            .map(|a| a.name().to_string())
            .collect()
    }

    /// Advance the animation by `delta` seconds and render to the canvas, which
    /// is `width` by `height` device pixels. The skeleton is auto-fit and
    /// centered.
    pub fn frame(&mut self, delta: f32, width: i32, height: i32) {
        self.state.update(delta);
        self.skeleton.update(delta);
        self.skeleton.set_bones_to_setup_pose();
        self.skeleton.set_slots_to_setup_pose();
        self.state.apply(&mut self.skeleton);
        self.skeleton.update_world_transform();
        render_into(&self.skeleton, &mut self.commands);

        let view = self.fit_view(width, height);
        self.renderer.begin_frame(width, height);
        self.renderer.draw(&view, &self.commands);
    }
}

impl WebSpine {
    /// Bind the atlas, build the renderer and skeleton, and measure the
    /// setup-pose fit. Shared by both loaders.
    fn assemble(
        canvas: &HtmlCanvasElement,
        data: &mut chine::data::SkeletonData,
        atlas_text: &str,
    ) -> Result<WebSpine, JsValue> {
        let atlas = Atlas::parse(atlas_text);
        bind_atlas(data, &atlas);
        let page_names = atlas.pages.iter().map(|p| p.name.clone()).collect();
        let page_pma = atlas.pages.iter().map(|p| p.pma).collect();

        let renderer = GlRenderer::new(canvas).map_err(|e| JsValue::from_str(&e))?;
        let mut skeleton = Skeleton::new(Arc::new(std::mem::take(data)));
        skeleton.set_bones_to_setup_pose();
        skeleton.set_slots_to_setup_pose();
        skeleton.update_world_transform();

        let mut commands = Vec::new();
        render_into(&skeleton, &mut commands);
        let fit = setup_fit(&commands);

        Ok(WebSpine {
            skeleton,
            state: AnimationState::new(),
            renderer,
            commands,
            page_names,
            page_pma,
            fit,
        })
    }

    /// A column-major 4x4 orthographic view that centers the setup fit in the
    /// canvas, preserving aspect, with a little padding. Spine world space and
    /// WebGL clip space are both y-up, so no flip is needed.
    fn fit_view(&self, width: i32, height: i32) -> [f32; 16] {
        let (cx, cy, hx, hy) = self.fit;
        let aspect = (width.max(1) as f32) / (height.max(1) as f32);
        let pad = 1.1;
        let vw = (hx * pad).max(hy * pad * aspect).max(1.0);
        let vh = vw / aspect;
        [
            1.0 / vw,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0 / vh,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            -cx / vw,
            -cy / vh,
            0.0,
            1.0,
        ]
    }
}

/// Axis-aligned center and half-extents of the setup-pose geometry, used to
/// auto-fit the skeleton. Falls back to a unit box for an empty skeleton.
fn setup_fit(commands: &[RenderCommand]) -> (f32, f32, f32, f32) {
    let mut min = (f32::MAX, f32::MAX);
    let mut max = (f32::MIN, f32::MIN);
    for cmd in commands {
        for p in &cmd.positions {
            min.0 = min.0.min(p.x);
            min.1 = min.1.min(p.y);
            max.0 = max.0.max(p.x);
            max.1 = max.1.max(p.y);
        }
    }
    if min.0 > max.0 {
        return (0.0, 0.0, 1.0, 1.0);
    }
    let cx = (min.0 + max.0) * 0.5;
    let cy = (min.1 + max.1) * 0.5;
    let hx = ((max.0 - min.0) * 0.5).max(1.0);
    let hy = ((max.1 - min.1) * 0.5).max(1.0);
    (cx, cy, hx, hy)
}
