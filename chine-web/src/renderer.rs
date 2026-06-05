//! A minimal WebGL2 renderer for chine's [`RenderCommand`] stream.
//!
//! Each command is a textured, tinted triangle list referencing one atlas page;
//! the renderer uploads its geometry, binds the page texture, selects the blend
//! mode, and draws. chine stays renderer-agnostic; this is the web GPU backend.

use chine::data::BlendMode;
use chine::render::RenderCommand;
use wasm_bindgen::{JsCast, JsValue};
use web_sys::{
    HtmlCanvasElement, HtmlImageElement, WebGl2RenderingContext as Gl, WebGlBuffer, WebGlProgram,
    WebGlTexture, WebGlUniformLocation, WebGlVertexArrayObject,
};

const VERTEX_SRC: &str = r"#version 300 es
layout(location=0) in vec2 a_pos;
layout(location=1) in vec2 a_uv;
layout(location=2) in vec4 a_light;
layout(location=3) in vec4 a_dark;
uniform mat4 u_view;
out vec2 v_uv;
out vec4 v_light;
out vec4 v_dark;
void main() {
    v_uv = a_uv;
    v_light = a_light;
    v_dark = a_dark;
    gl_Position = u_view * vec4(a_pos, 0.0, 1.0);
}
";

// Spine's two-color (tint-black) formula. With a zero dark color it reduces to
// the plain light tint `texture * v_light`, so single-color rigs are unchanged.
const FRAGMENT_SRC: &str = r"#version 300 es
precision mediump float;
in vec2 v_uv;
in vec4 v_light;
in vec4 v_dark;
uniform sampler2D u_tex;
out vec4 frag;
void main() {
    vec4 tex = texture(u_tex, v_uv);
    frag.a = tex.a * v_light.a;
    frag.rgb = ((tex.a - 1.0) * v_dark.a + 1.0 - tex.rgb) * v_dark.rgb + tex.rgb * v_light.rgb;
}
";

/// Twelve floats per vertex: position (2), uv (2), light color (4), dark (4).
const FLOATS_PER_VERTEX: usize = 12;

/// A WebGL2 backend that draws chine render commands to a canvas.
pub struct GlRenderer {
    gl: Gl,
    program: WebGlProgram,
    u_view: WebGlUniformLocation,
    vao: WebGlVertexArrayObject,
    vbo: WebGlBuffer,
    ibo: WebGlBuffer,
    /// Atlas-page textures, indexed by chine page, with each page's premultiply
    /// flag (so the tint and blend match the texture).
    pages: Vec<(WebGlTexture, bool)>,
    verts: Vec<f32>,
    indices: Vec<u16>,
}

impl GlRenderer {
    /// Create a renderer over `canvas`, compiling the shader program.
    ///
    /// # Errors
    /// Returns a message if the WebGL2 context, shaders, or program cannot be
    /// created.
    pub fn new(canvas: &HtmlCanvasElement) -> Result<Self, String> {
        // Keep the drawing buffer so the rendered frame can be read back or
        // captured (toDataURL / screenshots) after compositing.
        let options = js_sys::Object::new();
        let _ = js_sys::Reflect::set(
            &options,
            &JsValue::from_str("preserveDrawingBuffer"),
            &JsValue::TRUE,
        );
        let gl = canvas
            .get_context_with_context_options("webgl2", &options)
            .map_err(|_| "getContext failed".to_string())?
            .ok_or("WebGL2 is not available")?
            .dyn_into::<Gl>()
            .map_err(|_| "not a WebGL2 context".to_string())?;

        let program = link_program(&gl, VERTEX_SRC, FRAGMENT_SRC)?;
        let u_view = gl
            .get_uniform_location(&program, "u_view")
            .ok_or("missing u_view uniform")?;

        let vao = gl.create_vertex_array().ok_or("create_vertex_array")?;
        let vbo = gl.create_buffer().ok_or("create_buffer (vbo)")?;
        let ibo = gl.create_buffer().ok_or("create_buffer (ibo)")?;

        gl.bind_vertex_array(Some(&vao));
        gl.bind_buffer(Gl::ARRAY_BUFFER, Some(&vbo));
        gl.bind_buffer(Gl::ELEMENT_ARRAY_BUFFER, Some(&ibo));
        let stride = (FLOATS_PER_VERTEX * 4) as i32;
        // a_pos (2), a_uv (2), a_light (4), a_dark (4).
        gl.vertex_attrib_pointer_with_i32(0, 2, Gl::FLOAT, false, stride, 0);
        gl.enable_vertex_attrib_array(0);
        gl.vertex_attrib_pointer_with_i32(1, 2, Gl::FLOAT, false, stride, 2 * 4);
        gl.enable_vertex_attrib_array(1);
        gl.vertex_attrib_pointer_with_i32(2, 4, Gl::FLOAT, false, stride, 4 * 4);
        gl.enable_vertex_attrib_array(2);
        gl.vertex_attrib_pointer_with_i32(3, 4, Gl::FLOAT, false, stride, 8 * 4);
        gl.enable_vertex_attrib_array(3);
        gl.bind_vertex_array(None);

        gl.enable(Gl::BLEND);

        Ok(Self {
            gl,
            program,
            u_view,
            vao,
            vbo,
            ibo,
            pages: Vec::new(),
            verts: Vec::new(),
            indices: Vec::new(),
        })
    }

    /// Upload one atlas-page image as a texture. Pages must be added in chine
    /// page order; `pma` is the page's premultiplied-alpha flag.
    ///
    /// # Errors
    /// Returns a message if the texture cannot be created or uploaded.
    pub fn add_page(&mut self, image: &HtmlImageElement, pma: bool) -> Result<(), String> {
        let gl = &self.gl;
        let tex = gl.create_texture().ok_or("create_texture")?;
        gl.bind_texture(Gl::TEXTURE_2D, Some(&tex));
        gl.tex_image_2d_with_u32_and_u32_and_html_image_element(
            Gl::TEXTURE_2D,
            0,
            Gl::RGBA as i32,
            Gl::RGBA,
            Gl::UNSIGNED_BYTE,
            image,
        )
        .map_err(|_| "texImage2D failed".to_string())?;
        gl.tex_parameteri(Gl::TEXTURE_2D, Gl::TEXTURE_MIN_FILTER, Gl::LINEAR as i32);
        gl.tex_parameteri(Gl::TEXTURE_2D, Gl::TEXTURE_MAG_FILTER, Gl::LINEAR as i32);
        gl.tex_parameteri(Gl::TEXTURE_2D, Gl::TEXTURE_WRAP_S, Gl::CLAMP_TO_EDGE as i32);
        gl.tex_parameteri(Gl::TEXTURE_2D, Gl::TEXTURE_WRAP_T, Gl::CLAMP_TO_EDGE as i32);
        self.pages.push((tex, pma));
        Ok(())
    }

    /// The number of atlas pages uploaded so far.
    #[must_use]
    pub fn page_count(&self) -> usize {
        self.pages.len()
    }

    /// Clear the canvas to transparent and set the viewport.
    pub fn begin_frame(&self, width: i32, height: i32) {
        let gl = &self.gl;
        gl.viewport(0, 0, width, height);
        gl.clear_color(0.0, 0.0, 0.0, 0.0);
        gl.clear(Gl::COLOR_BUFFER_BIT);
    }

    /// Draw every command with the given column-major 4x4 `view` matrix.
    pub fn draw(&mut self, view: &[f32], commands: &[RenderCommand]) {
        let gl = &self.gl;
        gl.use_program(Some(&self.program));
        gl.uniform_matrix4fv_with_f32_array(Some(&self.u_view), false, view);
        gl.uniform1i(gl.get_uniform_location(&self.program, "u_tex").as_ref(), 0);
        gl.active_texture(Gl::TEXTURE0);
        gl.bind_vertex_array(Some(&self.vao));
        // ARRAY_BUFFER is not VAO state, so bind the streaming buffers
        // explicitly before uploading each command's geometry.
        gl.bind_buffer(Gl::ARRAY_BUFFER, Some(&self.vbo));
        gl.bind_buffer(Gl::ELEMENT_ARRAY_BUFFER, Some(&self.ibo));

        for cmd in commands {
            let Some((tex, pma)) = self.pages.get(cmd.page) else {
                continue;
            };
            let pma = *pma;
            // The dark tint for two-color (tint-black); zero means single-color.
            let (dr, dg, db, da) = match cmd.dark_color {
                Some(d) => (d.r, d.g, d.b, d.a),
                None => (0.0, 0.0, 0.0, 0.0),
            };
            self.verts.clear();
            for (i, pos) in cmd.positions.iter().enumerate() {
                let (u, v) = (cmd.uvs[i * 2], cmd.uvs[i * 2 + 1]);
                let c = cmd.color;
                // Premultiply the light tint for premultiplied-alpha pages so it
                // matches the texture and the blend equation.
                let (r, g, b) = if pma {
                    (c.r * c.a, c.g * c.a, c.b * c.a)
                } else {
                    (c.r, c.g, c.b)
                };
                self.verts
                    .extend_from_slice(&[pos.x, pos.y, u, v, r, g, b, c.a, dr, dg, db, da]);
            }
            self.indices.clear();
            self.indices.extend_from_slice(&cmd.triangles);

            gl.bind_texture(Gl::TEXTURE_2D, Some(tex));
            set_blend(gl, cmd.blend, pma);

            // SAFETY: the typed-array views borrow the scratch buffers and are
            // dropped before any reallocation; buffer_data copies immediately.
            unsafe {
                let vview = js_sys::Float32Array::view(&self.verts);
                gl.buffer_data_with_array_buffer_view(Gl::ARRAY_BUFFER, &vview, Gl::DYNAMIC_DRAW);
                let iview = js_sys::Uint16Array::view(&self.indices);
                gl.buffer_data_with_array_buffer_view(
                    Gl::ELEMENT_ARRAY_BUFFER,
                    &iview,
                    Gl::DYNAMIC_DRAW,
                );
            }
            gl.draw_elements_with_i32(
                Gl::TRIANGLES,
                self.indices.len() as i32,
                Gl::UNSIGNED_SHORT,
                0,
            );
        }
        gl.bind_vertex_array(None);
    }
}

/// Set the blend function for a chine blend mode, accounting for premultiplied
/// alpha (Spine's blend table).
fn set_blend(gl: &Gl, blend: BlendMode, pma: bool) {
    let (src, dst) = match blend {
        BlendMode::Normal => {
            if pma {
                (Gl::ONE, Gl::ONE_MINUS_SRC_ALPHA)
            } else {
                (Gl::SRC_ALPHA, Gl::ONE_MINUS_SRC_ALPHA)
            }
        }
        BlendMode::Additive => {
            if pma {
                (Gl::ONE, Gl::ONE)
            } else {
                (Gl::SRC_ALPHA, Gl::ONE)
            }
        }
        BlendMode::Multiply => (Gl::DST_COLOR, Gl::ONE_MINUS_SRC_ALPHA),
        BlendMode::Screen => (Gl::ONE, Gl::ONE_MINUS_SRC_COLOR),
    };
    gl.blend_func(src, dst);
}

fn link_program(gl: &Gl, vertex: &str, fragment: &str) -> Result<WebGlProgram, String> {
    let vs = compile_shader(gl, Gl::VERTEX_SHADER, vertex)?;
    let fs = compile_shader(gl, Gl::FRAGMENT_SHADER, fragment)?;
    let program = gl.create_program().ok_or("create_program")?;
    gl.attach_shader(&program, &vs);
    gl.attach_shader(&program, &fs);
    gl.link_program(&program);
    if gl
        .get_program_parameter(&program, Gl::LINK_STATUS)
        .as_bool()
        .unwrap_or(false)
    {
        Ok(program)
    } else {
        Err(gl
            .get_program_info_log(&program)
            .unwrap_or_else(|| "program link failed".to_string()))
    }
}

fn compile_shader(gl: &Gl, kind: u32, source: &str) -> Result<web_sys::WebGlShader, String> {
    let shader = gl.create_shader(kind).ok_or("create_shader")?;
    gl.shader_source(&shader, source);
    gl.compile_shader(&shader);
    if gl
        .get_shader_parameter(&shader, Gl::COMPILE_STATUS)
        .as_bool()
        .unwrap_or(false)
    {
        Ok(shader)
    } else {
        Err(gl
            .get_shader_info_log(&shader)
            .unwrap_or_else(|| "shader compile failed".to_string()))
    }
}
