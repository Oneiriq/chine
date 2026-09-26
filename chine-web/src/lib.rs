//! WebAssembly + WebGL2 web-component runtime for the [`chine`] Spine 4.3
//! runtime.
//!
//! This crate compiles to WebAssembly and renders a chine-posed skeleton to an
//! HTML `<canvas>` through WebGL2, driven by a drop-in custom element (see
//! `js/chine-spine.js`). chine itself stays renderer-agnostic. All the web and
//! GPU code lives here.
//!
//! The [`WebSpine`] type is the JS-facing API: construct it from a skeleton
//! (`.skel` or `.json`) and atlas, register each atlas-page image, then call
//! [`WebSpine::frame`] every animation frame.
#![warn(missing_docs)]
#![warn(clippy::all)]

mod player;
mod renderer;

use wasm_bindgen::prelude::*;
use web_sys::{HtmlCanvasElement, HtmlImageElement};

use chine::atlas::Atlas;
use chine::render::{bind_atlas, RenderCommand};

use player::Player;
use renderer::GlRenderer;

/// A loaded, animatable Spine skeleton bound to a WebGL2 canvas.
///
/// Build with [`WebSpine::from_binary`] or [`WebSpine::from_json`], register the
/// atlas-page images with [`WebSpine::add_page`] (in the order given by
/// [`WebSpine::page_names`]), then drive it each frame with [`WebSpine::frame`].
/// [`WebSpine::set_skin`] shows one of the skins [`WebSpine::skin_names`]
/// lists.
#[wasm_bindgen]
pub struct WebSpine {
    player: Player,
    renderer: GlRenderer,
    page_names: Vec<String>,
    page_pma: Vec<bool>,
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
    #[cfg(feature = "json")]
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
        self.player.set_animation(name, looping);
    }

    /// The animation names available on this skeleton.
    #[must_use]
    pub fn animation_names(&self) -> Vec<String> {
        self.player.animation_names()
    }

    /// The skin names available on this skeleton, starting with `"default"`.
    #[must_use]
    pub fn skin_names(&self) -> Vec<String> {
        self.player.skin_names()
    }

    /// Show the named skin: its attachments, the skin-required bones and
    /// constraints it lists, and the deform and sequence keys that name it.
    /// `"default"` or an unknown name shows the default skin only. Returns
    /// `false` for an unknown name. The skeleton is fit to the canvas again,
    /// since the fit depends on the attachments shown.
    pub fn set_skin(&mut self, name: &str) -> bool {
        self.player.set_skin(name)
    }

    /// Advance the animation by `delta` seconds and render to the canvas, which
    /// is `width` by `height` device pixels. The skeleton is auto-fit and
    /// centered.
    pub fn frame(&mut self, delta: f32, width: i32, height: i32) {
        let view = fit_view(self.player.fit(), width, height);
        let commands = self.player.advance(delta);
        self.renderer.begin_frame(width, height);
        self.renderer.draw(&view, commands);
    }
}

impl WebSpine {
    /// Bind the atlas, build the renderer, and start the player on the
    /// default skin. Shared by both loaders.
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
        Ok(WebSpine {
            player: Player::new(std::mem::take(data)),
            renderer,
            page_names,
            page_pma,
        })
    }
}

/// A column-major 4x4 orthographic view that centers `fit` (the setup-pose
/// center and half-extents) in the canvas, preserving aspect, with a little
/// padding. Spine world space and WebGL clip space are both y-up, so no flip is
/// needed.
fn fit_view(fit: (f32, f32, f32, f32), width: i32, height: i32) -> [f32; 16] {
    let (cx, cy, hx, hy) = fit;
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

#[cfg(test)]
mod tests {
    use super::{fit_view, setup_fit};
    use chine::data::{BlendMode, Color};
    use chine::render::RenderCommand;
    use glam::Vec2;

    fn quad(min: Vec2, max: Vec2) -> RenderCommand {
        RenderCommand {
            positions: vec![min, Vec2::new(max.x, min.y), max, Vec2::new(min.x, max.y)],
            uvs: vec![0.0; 8],
            triangles: vec![0, 1, 2, 2, 3, 0],
            color: Color::WHITE,
            dark_color: None,
            page: 0,
            blend: BlendMode::Normal,
        }
    }

    #[test]
    fn setup_fit_measures_the_aabb() {
        let cmds = [quad(Vec2::new(-10.0, -4.0), Vec2::new(10.0, 4.0))];
        let (cx, cy, hx, hy) = setup_fit(&cmds);
        assert!(cx.abs() < 1e-6 && cy.abs() < 1e-6, "centered");
        assert!(
            (hx - 10.0).abs() < 1e-6 && (hy - 4.0).abs() < 1e-6,
            "half-extents"
        );
    }

    #[test]
    fn setup_fit_empty_falls_back_to_a_unit_box() {
        assert_eq!(setup_fit(&[]), (0.0, 0.0, 1.0, 1.0));
    }

    #[test]
    fn fit_view_centers_and_preserves_aspect() {
        // A 20x20 rig (half 10) in a square canvas: with pad 1.1 the view half
        // width is 11, so the rig edge at x = 11 maps to clip x = 1.
        let m = fit_view((0.0, 0.0, 10.0, 10.0), 100, 100);
        assert!((m[0] - 1.0 / 11.0).abs() < 1e-6, "x scale");
        assert!((m[5] - 1.0 / 11.0).abs() < 1e-6, "y scale");
        assert!(
            m[12].abs() < 1e-6 && m[13].abs() < 1e-6,
            "centered at origin"
        );
        // A wider canvas widens the view, shrinking the x scale.
        let wide = fit_view((0.0, 0.0, 10.0, 10.0), 200, 100);
        assert!(wide[0] < m[0], "wider canvas gives a smaller x scale");
    }
}
