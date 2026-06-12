//! Attachments (the geometry a slot can display) and world-vertex computation.
//!
//! [`RegionAttachment`] (a textured quad on one bone) and [`MeshAttachment`] (a
//! textured mesh whose vertices may be weighted across several bones) are the
//! renderable types; [`PathAttachment`] holds the Bezier control points that
//! path constraints follow. Bounding-box and point attachments are exposed as
//! geometry/transform data; linked meshes resolve to their parent's geometry at
//! load time; clipping attachments mask the slots they cover with a polygon.

use glam::Vec2;

use crate::atlas::AtlasRegion;
use crate::data::Color;
use crate::skel::{Bone, Skeleton};

/// Degrees-to-radians factor.
const DEG_RAD: f32 = core::f32::consts::PI / 180.0;

// Corner indices into the region offset/uv arrays (Spine's order).
const BLX: usize = 0;
const BLY: usize = 1;
const ULX: usize = 2;
const ULY: usize = 3;
const URX: usize = 4;
const URY: usize = 5;
const BRX: usize = 6;
const BRY: usize = 7;

/// What a slot displays.
#[derive(Debug, Clone)]
pub enum Attachment {
    /// A textured quad on a single bone.
    Region(RegionAttachment),
    /// A textured mesh, optionally weighted across bones.
    Mesh(MeshAttachment),
    /// A composite Bezier path that path constraints follow.
    Path(PathAttachment),
    /// A collision/hit polygon (not rendered).
    BoundingBox(BoundingBoxAttachment),
    /// A point with a position and rotation on a bone (not rendered).
    Point(PointAttachment),
    /// A mesh that borrows a parent mesh's geometry (resolved to `Mesh` at load).
    LinkedMesh(LinkedMeshAttachment),
    /// A polygon that masks the slots it covers (not rendered itself).
    Clipping(ClippingAttachment),
}

/// An animated (flipbook) attachment: a run of atlas regions shown over time.
/// Each region's UVs and atlas page are resolved at bind time and selected per
/// frame by the slot's sequence index.
#[derive(Debug, Clone)]
pub struct Sequence {
    /// Number of regions in the sequence.
    pub count: usize,
    /// Starting number for the region path suffix.
    pub start: usize,
    /// Minimum digits in the suffix (zero-padded).
    pub digits: usize,
    /// Region index shown in the setup pose.
    pub setup_index: usize,
    frames: Vec<SequenceFrame>,
}

/// One resolved sequence frame: its bound UVs (8 for a region, `2 * vertices`
/// for a mesh) and atlas page.
#[derive(Debug, Clone)]
struct SequenceFrame {
    uvs: Vec<f32>,
    page: usize,
}

impl Sequence {
    /// A sequence with the given counts and setup frame; its per-frame UVs are
    /// resolved at bind time.
    #[must_use]
    pub fn new(count: usize, start: usize, digits: usize, setup_index: usize) -> Self {
        Self {
            count,
            start,
            digits,
            setup_index,
            frames: Vec::new(),
        }
    }

    /// The atlas region name for frame `index`: the base path plus the
    /// zero-padded `start + index`.
    #[must_use]
    pub fn region_name(&self, base: &str, index: usize) -> String {
        let frame = (self.start + index).to_string();
        let pad = self.digits.saturating_sub(frame.len());
        format!("{base}{}{frame}", "0".repeat(pad))
    }

    /// Record a frame's bound UVs and atlas page (called at bind time).
    pub(crate) fn push_frame(&mut self, uvs: Vec<f32>, page: usize) {
        self.frames.push(SequenceFrame { uvs, page });
    }

    /// The bound UVs and page for frame `index` (clamped to the resolved range).
    #[must_use]
    pub(crate) fn frame(&self, index: usize) -> Option<(&[f32], usize)> {
        if self.frames.is_empty() {
            return None;
        }
        let i = index.min(self.frames.len() - 1);
        self.frames.get(i).map(|f| (f.uvs.as_slice(), f.page))
    }
}

/// A textured quad attached to a slot's bone.
#[derive(Debug, Clone)]
pub struct RegionAttachment {
    /// Attachment name (the key within a skin).
    pub name: String,
    /// Atlas region name this draws (`AtlasRegion::name`).
    pub path: String,
    /// Local x offset from the bone.
    pub x: f32,
    /// Local y offset from the bone.
    pub y: f32,
    /// Local x scale.
    pub scale_x: f32,
    /// Local y scale.
    pub scale_y: f32,
    /// Local rotation in degrees.
    pub rotation: f32,
    /// Quad width in world units.
    pub width: f32,
    /// Quad height in world units.
    pub height: f32,
    /// Tint color.
    pub color: Color,
    /// Eight local corner offsets `[BLx, BLy, ULx, ULy, URx, URy, BRx, BRy]`,
    /// computed by [`Self::update`].
    offset: [f32; 8],
    /// Eight UVs in the same corner order, computed by [`Self::update`].
    pub uvs: [f32; 8],
    /// Atlas page index (texture), set when the atlas is bound.
    pub page: usize,
    /// The animated sequence this attachment cycles through, if any.
    pub sequence: Option<Sequence>,
}

impl RegionAttachment {
    /// A region attachment with identity transform and the given name/path.
    #[must_use]
    pub fn new(name: impl Into<String>, path: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            path: path.into(),
            x: 0.0,
            y: 0.0,
            scale_x: 1.0,
            scale_y: 1.0,
            rotation: 0.0,
            width: 0.0,
            height: 0.0,
            color: Color::WHITE,
            offset: [0.0; 8],
            uvs: [0.0; 8],
            page: 0,
            sequence: None,
        }
    }

    /// Recompute the local corner offsets and UVs against an atlas region.
    /// Call after changing the transform or binding a region. `page_w`/`page_h`
    /// are the region's page size (to normalize UVs).
    pub fn update(&mut self, region: &AtlasRegion, page_w: u32, page_h: u32) {
        let mut local_x2 = self.width / 2.0;
        let mut local_y2 = self.height / 2.0;
        let mut local_x = -local_x2;
        let mut local_y = -local_y2;

        // Account for whitespace the packer stripped: shift/scale the quad so
        // the trimmed region still sits where the full image would.
        let ow = region.original_width.max(1) as f32;
        let oh = region.original_height.max(1) as f32;
        local_x += region.offset_x / ow * self.width;
        local_y += region.offset_y / oh * self.height;
        if region.degrees == 90 {
            local_x2 = local_x + region.height as f32 / ow * self.width;
            local_y2 = local_y + region.width as f32 / oh * self.height;
        } else {
            local_x2 = local_x + region.width as f32 / ow * self.width;
            local_y2 = local_y + region.height as f32 / oh * self.height;
        }

        local_x *= self.scale_x;
        local_y *= self.scale_y;
        local_x2 *= self.scale_x;
        local_y2 *= self.scale_y;

        let cos = (self.rotation * DEG_RAD).cos();
        let sin = (self.rotation * DEG_RAD).sin();
        let lx_cos = local_x * cos + self.x;
        let lx_sin = local_x * sin;
        let ly_cos = local_y * cos + self.y;
        let ly_sin = local_y * sin;
        let lx2_cos = local_x2 * cos + self.x;
        let lx2_sin = local_x2 * sin;
        let ly2_cos = local_y2 * cos + self.y;
        let ly2_sin = local_y2 * sin;

        self.offset[BLX] = lx_cos - ly_sin;
        self.offset[BLY] = ly_cos + lx_sin;
        self.offset[ULX] = lx_cos - ly2_sin;
        self.offset[ULY] = ly2_cos + lx_sin;
        self.offset[URX] = lx2_cos - ly2_sin;
        self.offset[URY] = ly2_cos + lx2_sin;
        self.offset[BRX] = lx2_cos - ly_sin;
        self.offset[BRY] = ly_cos + lx2_sin;

        let pw = page_w.max(1) as f32;
        let ph = page_h.max(1) as f32;
        let u = region.x as f32 / pw;
        let v = (region.y + region.height) as f32 / ph;
        let u2 = (region.x + region.width) as f32 / pw;
        let v2 = region.y as f32 / ph;
        if region.degrees == 90 {
            self.uvs = [u, v2, u2, v2, u2, v, u, v];
        } else {
            self.uvs = [u, v, u, v2, u2, v2, u2, v];
        }
    }

    /// The four world-space corner positions, in order BL, UL, UR, BR.
    #[must_use]
    pub fn compute_world_vertices(&self, bone: &Bone) -> [Vec2; 4] {
        let (a, b, c, d) = (bone.a(), bone.b(), bone.c(), bone.d());
        let (wx, wy) = (bone.world_x(), bone.world_y());
        let mut out = [Vec2::ZERO; 4];
        for (i, slot) in out.iter_mut().enumerate() {
            let ox = self.offset[i * 2];
            let oy = self.offset[i * 2 + 1];
            *slot = Vec2::new(ox * a + oy * b + wx, ox * c + oy * d + wy);
        }
        out
    }
}

/// How a mesh's vertices are bound to bones.
#[derive(Debug, Clone)]
pub enum MeshVertices {
    /// Each vertex is `[x, y]` in the slot bone's local space.
    Unweighted(Vec<f32>),
    /// Weighted skinning. `bones` is, per vertex, `[count, boneIndex...]`;
    /// `vertices` is, per influence, `[vx, vy, weight]`.
    Weighted {
        /// Per-vertex bone-influence layout: `[count, boneIndex...]`.
        bones: Vec<usize>,
        /// Per-influence bind data: `[vx, vy, weight]`.
        vertices: Vec<f32>,
    },
}

/// A textured mesh attachment.
#[derive(Debug, Clone)]
pub struct MeshAttachment {
    /// Attachment name (the key within a skin).
    pub name: String,
    /// Atlas region name this draws.
    pub path: String,
    /// Per-vertex UVs (`2 * vertex_count` values).
    pub uvs: Vec<f32>,
    /// Triangle index list (triples into the vertex array).
    pub triangles: Vec<u16>,
    /// Tint color.
    pub color: Color,
    /// Number of vertices forming the convex hull (for clipping).
    pub hull_length: usize,
    /// Bind-pose vertices.
    vertices: MeshVertices,
    /// Atlas page index (texture), set when the atlas is bound.
    pub page: usize,
    /// Skin whose deform timelines drive this mesh (`None` = default skin). For
    /// an inheriting linked mesh this points at the parent's skin.
    pub deform_skin: Option<String>,
    /// The animated sequence this attachment cycles through, if any.
    pub sequence: Option<Sequence>,
}

impl MeshAttachment {
    /// A mesh attachment from its parts.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        path: impl Into<String>,
        vertices: MeshVertices,
        uvs: Vec<f32>,
        triangles: Vec<u16>,
    ) -> Self {
        Self {
            name: name.into(),
            path: path.into(),
            uvs,
            triangles,
            color: Color::WHITE,
            hull_length: 0,
            vertices,
            page: 0,
            deform_skin: None,
            sequence: None,
        }
    }

    /// Number of vertices in the mesh.
    #[must_use]
    pub fn vertex_count(&self) -> usize {
        self.uvs.len() / 2
    }

    /// Compute world-space positions for every vertex. `slot_bone` is the index
    /// of the bone the slot follows (used for unweighted meshes); `deform`
    /// overrides the local vertices when non-empty (unweighted only).
    #[must_use]
    pub fn compute_world_vertices(
        &self,
        skeleton: &Skeleton,
        slot_bone: usize,
        deform: &[f32],
    ) -> Vec<Vec2> {
        let mut out = Vec::with_capacity(self.vertex_count());
        self.compute_world_vertices_into(skeleton, slot_bone, deform, &mut out);
        out
    }

    /// [`Self::compute_world_vertices`] into a reused buffer, which is cleared
    /// first: a render loop avoids a fresh allocation per mesh per frame.
    pub fn compute_world_vertices_into(
        &self,
        skeleton: &Skeleton,
        slot_bone: usize,
        deform: &[f32],
        out: &mut Vec<Vec2>,
    ) {
        compute_vertices_into(
            &self.vertices,
            self.vertex_count(),
            skeleton,
            slot_bone,
            deform,
            out,
        );
    }

    /// The unweighted setup vertex positions (`2 * vertex_count`), or `None` for
    /// a weighted mesh. Used to build deform timelines.
    #[must_use]
    pub fn setup_vertices(&self) -> Option<&[f32]> {
        match &self.vertices {
            MeshVertices::Unweighted(v) => Some(v),
            MeshVertices::Weighted { .. } => None,
        }
    }

    /// The length of a deform vertex array for this mesh: `2 * vertex_count` for
    /// an unweighted mesh, or `2 *` the total influence count for a weighted one.
    #[must_use]
    pub fn deform_len(&self) -> usize {
        match &self.vertices {
            MeshVertices::Unweighted(v) => v.len(),
            MeshVertices::Weighted { vertices, .. } => (vertices.len() / 3) * 2,
        }
    }

    /// Remap the mesh's `[0, 1]` region-relative UVs into page space using the
    /// bound atlas region, handling a 90-degree rotated region.
    pub fn remap_uvs(&mut self, region: &AtlasRegion, page_w: u32, page_h: u32) {
        let (pw, ph) = (page_w.max(1) as f32, page_h.max(1) as f32);
        let (rx, ry) = (region.x as f32, region.y as f32);
        let (rw, rh) = (region.width as f32, region.height as f32);
        for i in 0..self.vertex_count() {
            let mu = self.uvs[i * 2];
            let mv = self.uvs[i * 2 + 1];
            if region.degrees == 90 {
                // The region is packed rotated 90 degrees: swap and flip the axes.
                self.uvs[i * 2] = (rx + (1.0 - mv) * rw) / pw;
                self.uvs[i * 2 + 1] = (ry + mu * rh) / ph;
            } else {
                self.uvs[i * 2] = (rx + mu * rw) / pw;
                self.uvs[i * 2 + 1] = (ry + mv * rh) / ph;
            }
        }
    }
}

/// Transform a vertex attachment's bind-pose vertices into world space. Shared
/// by [`MeshAttachment`] and [`PathAttachment`]: unweighted vertices follow the
/// slot bone; weighted vertices are a blend across their influence bones.
fn compute_vertices(
    vertices: &MeshVertices,
    count: usize,
    skeleton: &Skeleton,
    slot_bone: usize,
    deform: &[f32],
) -> Vec<Vec2> {
    let mut out = Vec::with_capacity(count);
    compute_vertices_into(vertices, count, skeleton, slot_bone, deform, &mut out);
    out
}

/// [`compute_vertices`] into a reused buffer, which is cleared first.
fn compute_vertices_into(
    vertices: &MeshVertices,
    count: usize,
    skeleton: &Skeleton,
    slot_bone: usize,
    deform: &[f32],
    out: &mut Vec<Vec2>,
) {
    out.clear();
    match vertices {
        MeshVertices::Unweighted(v) => {
            let Some(bone) = skeleton.bone(slot_bone) else {
                return;
            };
            // A deform timeline overrides the local vertices for unweighted
            // meshes.
            let local = if deform.len() >= count * 2 { deform } else { v };
            let (a, b, c, d) = (bone.a(), bone.b(), bone.c(), bone.d());
            let (wx, wy) = (bone.world_x(), bone.world_y());
            for i in 0..count {
                let vx = local[i * 2];
                let vy = local[i * 2 + 1];
                out.push(Vec2::new(vx * a + vy * b + wx, vx * c + vy * d + wy));
            }
        }
        MeshVertices::Weighted { bones, vertices } => {
            // A deform timeline adds a per-influence offset to each bind vertex.
            let mut bi = 0;
            let mut vi = 0;
            let mut fi = 0;
            for _ in 0..count {
                let influences = bones[bi];
                bi += 1;
                let mut wx = 0.0;
                let mut wy = 0.0;
                for _ in 0..influences {
                    let bone_index = bones[bi];
                    bi += 1;
                    let dx = deform.get(fi).copied().unwrap_or(0.0);
                    let dy = deform.get(fi + 1).copied().unwrap_or(0.0);
                    fi += 2;
                    let vx = vertices[vi] + dx;
                    let vy = vertices[vi + 1] + dy;
                    let weight = vertices[vi + 2];
                    vi += 3;
                    if let Some(bone) = skeleton.bone(bone_index) {
                        wx += (vx * bone.a() + vy * bone.b() + bone.world_x()) * weight;
                        wy += (vx * bone.c() + vy * bone.d() + bone.world_y()) * weight;
                    }
                }
                out.push(Vec2::new(wx, wy));
            }
        }
    }
}

/// A path attachment: a composite cubic-Bezier curve whose control points are a
/// vertex set (weighted or not, like a mesh). A path constraint samples
/// positions and tangents along it; this type provides the control-point
/// geometry.
#[derive(Debug, Clone)]
pub struct PathAttachment {
    /// Attachment name (the key within a skin).
    pub name: String,
    /// Whether the start and end knots connect.
    pub closed: bool,
    /// Whether to arc-length-parameterize so movement has constant speed.
    pub constant_speed: bool,
    /// Per-curve lengths, used when `constant_speed`.
    pub lengths: Vec<f32>,
    /// Bezier control points (bind pose).
    vertices: MeshVertices,
    /// Number of control points.
    vertex_count: usize,
}

impl PathAttachment {
    /// A path attachment from its parts.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        vertices: MeshVertices,
        vertex_count: usize,
        lengths: Vec<f32>,
        closed: bool,
        constant_speed: bool,
    ) -> Self {
        Self {
            name: name.into(),
            closed,
            constant_speed,
            lengths,
            vertices,
            vertex_count,
        }
    }

    /// Number of Bezier control points.
    #[must_use]
    pub fn vertex_count(&self) -> usize {
        self.vertex_count
    }

    /// Compute world-space positions for every control point.
    #[must_use]
    pub fn compute_world_vertices(&self, skeleton: &Skeleton, slot_bone: usize) -> Vec<Vec2> {
        compute_vertices(&self.vertices, self.vertex_count, skeleton, slot_bone, &[])
    }
}

/// A bounding-box attachment: a (weighted or unweighted) polygon used for
/// collision/hit queries. Not rendered; the host transforms it to world space.
#[derive(Debug, Clone)]
pub struct BoundingBoxAttachment {
    /// Attachment name (the key within a skin).
    pub name: String,
    /// Polygon vertices (bind pose).
    vertices: MeshVertices,
    /// Number of polygon vertices.
    vertex_count: usize,
}

impl BoundingBoxAttachment {
    /// A bounding-box attachment from its parts.
    #[must_use]
    pub fn new(name: impl Into<String>, vertices: MeshVertices, vertex_count: usize) -> Self {
        Self {
            name: name.into(),
            vertices,
            vertex_count,
        }
    }

    /// Number of polygon vertices.
    #[must_use]
    pub fn vertex_count(&self) -> usize {
        self.vertex_count
    }

    /// Compute world-space positions for every polygon vertex.
    #[must_use]
    pub fn compute_world_vertices(&self, skeleton: &Skeleton, slot_bone: usize) -> Vec<Vec2> {
        compute_vertices(&self.vertices, self.vertex_count, skeleton, slot_bone, &[])
    }
}

/// A point attachment: a position and rotation on a bone (a handle for spawning
/// effects, aiming, and similar). Not rendered.
#[derive(Debug, Clone)]
pub struct PointAttachment {
    /// Attachment name (the key within a skin).
    pub name: String,
    /// Local x offset from the bone.
    pub x: f32,
    /// Local y offset from the bone.
    pub y: f32,
    /// Local rotation, in degrees.
    pub rotation: f32,
}

impl PointAttachment {
    /// A point attachment from its parts.
    #[must_use]
    pub fn new(name: impl Into<String>, x: f32, y: f32, rotation: f32) -> Self {
        Self {
            name: name.into(),
            x,
            y,
            rotation,
        }
    }

    /// The point's world-space position on `bone`.
    #[must_use]
    pub fn compute_world_position(&self, bone: &Bone) -> Vec2 {
        Vec2::new(
            self.x * bone.a() + self.y * bone.b() + bone.world_x(),
            self.x * bone.c() + self.y * bone.d() + bone.world_y(),
        )
    }

    /// The point's world-space rotation on `bone`, in degrees.
    #[must_use]
    pub fn compute_world_rotation(&self, bone: &Bone) -> f32 {
        let (sin, cos) = (self.rotation * DEG_RAD).sin_cos();
        let x = cos * bone.a() + sin * bone.b();
        let y = cos * bone.c() + sin * bone.d();
        y.atan2(x).to_degrees()
    }
}

/// A linked mesh: a mesh that borrows its geometry from a parent mesh in some
/// skin, supplying only its own texture path and tint. Resolved to a plain
/// [`MeshAttachment`] at load time, once every skin is parsed.
#[derive(Debug, Clone)]
pub struct LinkedMeshAttachment {
    /// Attachment name (the key within a skin).
    pub name: String,
    /// Atlas region name this draws.
    pub path: String,
    /// Skin holding the parent mesh, or `None` for the linked mesh's own skin.
    pub skin: Option<String>,
    /// Parent attachment name (within the linked mesh's slot).
    pub parent: String,
    /// Tint color.
    pub color: Color,
    /// Whether this link shares its parent's deform timelines (Spine `deform`).
    pub inherit_deform: bool,
}

impl LinkedMeshAttachment {
    /// A linked-mesh reference from its parts.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        path: impl Into<String>,
        skin: Option<String>,
        parent: impl Into<String>,
        color: Color,
        inherit_deform: bool,
    ) -> Self {
        Self {
            name: name.into(),
            path: path.into(),
            skin,
            parent: parent.into(),
            color,
            inherit_deform,
        }
    }

    /// Resolve to a concrete mesh by borrowing `parent`'s geometry (vertices,
    /// UVs, triangles, hull) while keeping this link's own name, path, and tint.
    /// The page is reset so atlas binding re-resolves it against this path.
    #[must_use]
    pub fn resolve(&self, parent: &MeshAttachment) -> MeshAttachment {
        let mut m = parent.clone();
        m.name = self.name.clone();
        m.path = self.path.clone();
        m.color = self.color;
        m.page = 0;
        m
    }
}

/// A clipping attachment: a polygon that masks the slots from its own slot up to
/// and including `end_slot` (in draw order). Convex polygons clip exactly;
/// concave polygons clip against their convex span (a known simplification).
#[derive(Debug, Clone)]
pub struct ClippingAttachment {
    /// Attachment name (the key within a skin).
    pub name: String,
    /// Name of the slot at which clipping ends (resolved at render time).
    pub end_slot: String,
    /// Clip polygon vertices (bind pose).
    vertices: MeshVertices,
    /// Number of polygon vertices.
    vertex_count: usize,
}

impl ClippingAttachment {
    /// A clipping attachment from its parts.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        end_slot: impl Into<String>,
        vertices: MeshVertices,
        vertex_count: usize,
    ) -> Self {
        Self {
            name: name.into(),
            end_slot: end_slot.into(),
            vertices,
            vertex_count,
        }
    }

    /// Number of clip-polygon vertices.
    #[must_use]
    pub fn vertex_count(&self) -> usize {
        self.vertex_count
    }

    /// Compute the clip polygon's world-space vertices.
    #[must_use]
    pub fn compute_world_vertices(&self, skeleton: &Skeleton, slot_bone: usize) -> Vec<Vec2> {
        compute_vertices(&self.vertices, self.vertex_count, skeleton, slot_bone, &[])
    }

    /// [`Self::compute_world_vertices`] into a reused buffer, which is cleared
    /// first: a render loop avoids a fresh allocation per clip per frame.
    pub fn compute_world_vertices_into(
        &self,
        skeleton: &Skeleton,
        slot_bone: usize,
        out: &mut Vec<Vec2>,
    ) {
        compute_vertices_into(&self.vertices, self.vertex_count, skeleton, slot_bone, &[], out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{BoneData, SkeletonData};
    use std::sync::Arc;

    const EPS: f32 = 1e-3;

    fn close(p: Vec2, x: f32, y: f32) -> bool {
        (p.x - x).abs() < EPS && (p.y - y).abs() < EPS
    }

    fn one_bone_at(x: f32, y: f32) -> Skeleton {
        let data = SkeletonData {
            bones: vec![BoneData {
                index: 0,
                name: "root".into(),
                position: Vec2::new(x, y),
                ..Default::default()
            }],
            ..Default::default()
        };
        let mut sk = Skeleton::new(Arc::new(data));
        sk.update_world_transform();
        sk
    }

    #[test]
    fn region_quad_corners_center_on_the_bone() {
        let sk = one_bone_at(100.0, 50.0);
        let region = AtlasRegion {
            name: "r".into(),
            page: 0,
            x: 0,
            y: 0,
            width: 20,
            height: 10,
            degrees: 0,
            offset_x: 0.0,
            offset_y: 0.0,
            original_width: 20,
            original_height: 10,
            index: -1,
        };
        let mut att = RegionAttachment::new("r", "r");
        att.width = 20.0;
        att.height = 10.0;
        att.update(&region, 64, 64);
        let v = att.compute_world_vertices(sk.bone(0).unwrap());
        // 20x10 quad centered on (100,50): BL(90,45) UL(90,55) UR(110,55) BR(110,45).
        assert!(close(v[0], 90.0, 45.0), "BL {:?}", v[0]);
        assert!(close(v[1], 90.0, 55.0), "UL {:?}", v[1]);
        assert!(close(v[2], 110.0, 55.0), "UR {:?}", v[2]);
        assert!(close(v[3], 110.0, 45.0), "BR {:?}", v[3]);
    }

    #[test]
    fn unweighted_mesh_follows_its_bone() {
        let sk = one_bone_at(10.0, 20.0);
        let mesh = MeshAttachment::new(
            "m",
            "m",
            MeshVertices::Unweighted(vec![0.0, 0.0, 5.0, 0.0, 0.0, 5.0]),
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            vec![0, 1, 2],
        );
        let w = mesh.compute_world_vertices(&sk, 0, &[]);
        // identity bone at (10,20): each local vertex offset by (10,20).
        assert!(close(w[0], 10.0, 20.0));
        assert!(close(w[1], 15.0, 20.0));
        assert!(close(w[2], 10.0, 25.0));
    }

    #[test]
    fn weighted_mesh_blends_two_bones() {
        // Two roots at (0,0) and (100,0); a vertex weighted 50/50 lands at the
        // midpoint of where each bone places its local origin.
        let data = SkeletonData {
            bones: vec![
                BoneData {
                    index: 0,
                    name: "a".into(),
                    ..Default::default()
                },
                BoneData {
                    index: 1,
                    name: "b".into(),
                    position: Vec2::new(100.0, 0.0),
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        let mut sk = Skeleton::new(Arc::new(data));
        sk.update_world_transform();
        let mesh = MeshAttachment::new(
            "m",
            "m",
            MeshVertices::Weighted {
                bones: vec![2, 0, 1], // 1 vertex, influenced by bones 0 and 1
                vertices: vec![0.0, 0.0, 0.5, 0.0, 0.0, 0.5], // (0,0)w.5 via bone0; (0,0)w.5 via bone1
            },
            vec![0.0, 0.0],
            vec![],
        );
        let w = mesh.compute_world_vertices(&sk, 0, &[]);
        assert_eq!(w.len(), 1);
        assert!(close(w[0], 50.0, 0.0), "midpoint {:?}", w[0]);
    }

    #[test]
    fn path_control_points_follow_their_bone() {
        let sk = one_bone_at(10.0, 0.0);
        // 4 control points (one cubic Bezier), unweighted.
        let path = PathAttachment::new(
            "p",
            MeshVertices::Unweighted(vec![0.0, 0.0, 10.0, 0.0, 20.0, 0.0, 30.0, 0.0]),
            4,
            vec![30.0],
            false,
            true,
        );
        assert_eq!(path.vertex_count(), 4);
        let w = path.compute_world_vertices(&sk, 0);
        assert_eq!(w.len(), 4);
        // bone at (10,0): control points offset by +10 in x.
        assert!(close(w[0], 10.0, 0.0));
        assert!(close(w[3], 40.0, 0.0));
    }

    #[test]
    fn rotated_region_remaps_mesh_uvs() {
        let region = AtlasRegion {
            name: "r".into(),
            page: 0,
            x: 10,
            y: 20,
            width: 30,
            height: 40,
            degrees: 90,
            offset_x: 0.0,
            offset_y: 0.0,
            original_width: 40,
            original_height: 30,
            index: -1,
        };
        let mut m = MeshAttachment::new(
            "m",
            "m",
            MeshVertices::Unweighted(vec![0.0, 0.0, 5.0, 5.0]),
            vec![0.0, 0.0, 1.0, 1.0], // mesh UVs (0,0) and (1,1)
            vec![],
        );
        m.remap_uvs(&region, 100, 100);
        // degrees 90: u = (rx + (1 - mv) * rw) / pw, v = (ry + mu * rh) / ph.
        // (0,0) -> u=(10+30)/100=0.4, v=(20+0)/100=0.2.
        assert!((m.uvs[0] - 0.4).abs() < 1e-4, "u0={}", m.uvs[0]);
        assert!((m.uvs[1] - 0.2).abs() < 1e-4, "v0={}", m.uvs[1]);
        // (1,1) -> u=(10+0)/100=0.1, v=(20+40)/100=0.6.
        assert!((m.uvs[2] - 0.1).abs() < 1e-4, "u1={}", m.uvs[2]);
        assert!((m.uvs[3] - 0.6).abs() < 1e-4, "v1={}", m.uvs[3]);
    }

    #[test]
    fn bounding_box_polygon_follows_its_bone() {
        let sk = one_bone_at(10.0, 0.0);
        let bb = BoundingBoxAttachment::new(
            "bb",
            MeshVertices::Unweighted(vec![0.0, 0.0, 10.0, 0.0, 10.0, 10.0]),
            3,
        );
        assert_eq!(bb.vertex_count(), 3);
        let w = bb.compute_world_vertices(&sk, 0);
        assert_eq!(w.len(), 3);
        // bone at (10,0): each polygon vertex offset by +10 in x.
        assert!(close(w[0], 10.0, 0.0), "{:?}", w[0]);
        assert!(close(w[1], 20.0, 0.0), "{:?}", w[1]);
        assert!(close(w[2], 20.0, 10.0), "{:?}", w[2]);
    }

    #[test]
    fn point_attachment_transforms_with_its_bone() {
        let sk = one_bone_at(10.0, 20.0);
        let bone = sk.bone(0).unwrap();
        let p = PointAttachment::new("p", 5.0, 0.0, 90.0);
        let pos = p.compute_world_position(bone);
        assert!(close(pos, 15.0, 20.0), "pos {:?}", pos);
        // identity bone: world rotation equals the point's local rotation.
        assert!((p.compute_world_rotation(bone) - 90.0).abs() < 1e-3);
    }

    #[test]
    fn linked_mesh_borrows_parent_geometry() {
        let mut parent = MeshAttachment::new(
            "wing",
            "wing",
            MeshVertices::Unweighted(vec![0.0, 0.0, 10.0, 0.0, 0.0, 10.0]),
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            vec![0, 1, 2],
        );
        parent.hull_length = 3;
        let link =
            LinkedMeshAttachment::new("wing-blue", "wing-blue", None, "wing", Color::WHITE, true);
        let m = link.resolve(&parent);
        // Identity stays the link's own; geometry is borrowed from the parent.
        assert_eq!(m.name, "wing-blue");
        assert_eq!(m.path, "wing-blue");
        assert_eq!(m.triangles, parent.triangles);
        assert_eq!(m.hull_length, 3);
        assert_eq!(m.vertex_count(), 3);
    }
}
