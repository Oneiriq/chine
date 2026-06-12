//! Clipping geometry and the reusable render scratch.
//!
//! Spine's clipping attachment masks the slots that follow it: the clip
//! polygon is decomposed into convex CCW pieces (the polygon itself when
//! convex, an ear-clipping triangulation otherwise), and every drawn triangle
//! is Sutherland-Hodgman-clipped against each piece.
//!
//! All of that needs working buffers. [`RenderScratch`] owns them — plus the
//! emitted commands themselves — so a render loop that keeps one scratch and
//! calls [`render_with`](crate::render::render_with) every frame reuses every
//! buffer instead of reallocating it.

use std::mem;

use glam::Vec2;

use crate::render::RenderCommand;

/// A clip vertex: a world-space position with its interpolated UV.
pub(crate) type ClipVertex = (Vec2, (f32, f32));

/// Reusable buffers for [`render_with`](crate::render::render_with).
///
/// Holds the emitted commands (rebuilt in place each frame) and every
/// intermediate the clipper needs: the clip attachment's world polygon, its
/// decomposed convex pieces, the ear-clipping working set, and the
/// Sutherland-Hodgman ping-pong buffers. Keep one scratch per render loop;
/// once its buffers have grown to the skeleton's working sizes, steady-state
/// rendering of a clipped skeleton performs no heap allocation. Buffers only
/// grow; dropping the scratch releases them.
#[derive(Debug, Default)]
pub struct RenderScratch {
    /// Grow-only pool of output commands; a frame's commands are the prefix
    /// that [`render_with`](crate::render::render_with) returns, overwritten
    /// in place so their buffers keep their capacity.
    pub(crate) commands: Vec<RenderCommand>,
    /// Build target for a clipped slot's pre-clip geometry, which is then
    /// clipped into the output command.
    pub(crate) staging: RenderCommand,
    /// The active clip attachment's world-space polygon.
    pub(crate) clip_world: Vec<Vec2>,
    /// Flattened vertices of the decomposed convex clip pieces.
    pub(crate) poly_points: Vec<Vec2>,
    /// `(start, len)` spans into [`Self::poly_points`], one per convex piece.
    pub(crate) poly_ranges: Vec<(usize, usize)>,
    /// Ear-clipping working set: the not-yet-clipped vertex indices.
    pub(crate) ear: Vec<usize>,
    /// Sutherland-Hodgman subject polygon (ping).
    pub(crate) subject: Vec<ClipVertex>,
    /// Sutherland-Hodgman per-edge output polygon (pong).
    pub(crate) clipped: Vec<ClipVertex>,
}

impl RenderScratch {
    /// An empty scratch; its buffers grow on first use.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

/// Decompose `world` into convex CCW pieces appended to `points` / `ranges`,
/// replacing the pieces already there (the previous clip). Mirrors the
/// previous owned-`Vec` behavior: a degenerate polygon that yields no piece
/// leaves the previous clip's pieces in place and returns `false`.
pub(crate) fn replace_clip(
    world: &mut [Vec2],
    ear: &mut Vec<usize>,
    points: &mut Vec<Vec2>,
    ranges: &mut Vec<(usize, usize)>,
) -> bool {
    if world.len() < 3 {
        return false;
    }
    make_ccw(world);
    let points_mark = points.len();
    let ranges_mark = ranges.len();
    if is_convex(world) {
        ranges.push((points.len(), world.len()));
        points.extend_from_slice(world);
    } else {
        triangulate_into(world, ear, points, ranges);
    }
    if ranges.len() == ranges_mark {
        // Nothing usable came out (degenerate polygon): keep the old clip.
        points.truncate(points_mark);
        return false;
    }
    // The new pieces supersede the old clip: drop its prefix and rebase the
    // new spans onto the start of the buffer. Shifts in place, no allocation.
    points.drain(..points_mark);
    ranges.drain(..ranges_mark);
    for range in ranges.iter_mut() {
        range.0 -= points_mark;
    }
    true
}

/// Clip `src`'s triangles against the convex pieces in `points` / `ranges`,
/// interpolating UVs at the new edges and unioning the pieces into `dst`
/// (whose buffers are cleared and reused). Returns `false`, leaving `dst`'s
/// buffers empty, if nothing survives.
pub(crate) fn clip_into(
    src: &RenderCommand,
    points: &[Vec2],
    ranges: &[(usize, usize)],
    subject: &mut Vec<ClipVertex>,
    clipped: &mut Vec<ClipVertex>,
    dst: &mut RenderCommand,
) -> bool {
    dst.positions.clear();
    dst.uvs.clear();
    dst.triangles.clear();
    for tri in src.triangles.chunks_exact(3) {
        let idx = [tri[0] as usize, tri[1] as usize, tri[2] as usize];
        let p = [
            src.positions[idx[0]],
            src.positions[idx[1]],
            src.positions[idx[2]],
        ];
        let uv = [
            (src.uvs[idx[0] * 2], src.uvs[idx[0] * 2 + 1]),
            (src.uvs[idx[1] * 2], src.uvs[idx[1] * 2 + 1]),
            (src.uvs[idx[2] * 2], src.uvs[idx[2] * 2 + 1]),
        ];
        for &(start, len) in ranges {
            let polygon = &points[start..start + len];
            clip_triangle(subject, clipped, &p, &uv, polygon);
            if subject.len() < 3 {
                continue;
            }
            let base = dst.positions.len() as u16;
            for (pt, (u, v)) in subject.iter() {
                dst.positions.push(*pt);
                dst.uvs.push(*u);
                dst.uvs.push(*v);
            }
            for k in 1..subject.len() as u16 - 1 {
                dst.triangles
                    .extend_from_slice(&[base, base + k, base + k + 1]);
            }
        }
    }
    if dst.triangles.is_empty() {
        return false;
    }
    dst.color = src.color;
    dst.dark_color = src.dark_color;
    dst.page = src.page;
    dst.blend = src.blend;
    true
}

/// Sutherland-Hodgman clip of one triangle against a convex polygon (CCW),
/// leaving the clipped polygon's vertices (with interpolated UVs) in
/// `subject`; `clipped` is the per-edge ping-pong buffer. Fewer than three
/// remaining vertices means the triangle was clipped away.
fn clip_triangle(
    subject: &mut Vec<ClipVertex>,
    clipped: &mut Vec<ClipVertex>,
    p: &[Vec2; 3],
    uv: &[(f32, f32); 3],
    polygon: &[Vec2],
) {
    subject.clear();
    subject.extend_from_slice(&[(p[0], uv[0]), (p[1], uv[1]), (p[2], uv[2])]);
    let m = polygon.len();
    for i in 0..m {
        if subject.len() < 3 {
            break;
        }
        clip_against_edge(subject, clipped, polygon[i], polygon[(i + 1) % m]);
        mem::swap(subject, clipped);
    }
}

/// Clip a subject polygon against the inside half-plane of directed edge
/// `e1 -> e2` (left side, for a CCW clip polygon) into `out` (cleared first),
/// interpolating UVs at crossings.
fn clip_against_edge(subject: &[ClipVertex], out: &mut Vec<ClipVertex>, e1: Vec2, e2: Vec2) {
    out.clear();
    let n = subject.len();
    let edge = e2 - e1;
    let inside = |pt: Vec2| edge.x * (pt.y - e1.y) - edge.y * (pt.x - e1.x) >= 0.0;
    for i in 0..n {
        let (cur, cur_uv) = subject[i];
        let (prev, prev_uv) = subject[(i + n - 1) % n];
        let cur_in = inside(cur);
        let prev_in = inside(prev);
        if cur_in {
            if !prev_in {
                out.push(edge_intersect(prev, prev_uv, cur, cur_uv, e1, e2));
            }
            out.push((cur, cur_uv));
        } else if prev_in {
            out.push(edge_intersect(prev, prev_uv, cur, cur_uv, e1, e2));
        }
    }
}

/// The point (with interpolated UV) where segment `a -> b` crosses the line
/// through `e1 -> e2`.
fn edge_intersect(
    a: Vec2,
    a_uv: (f32, f32),
    b: Vec2,
    b_uv: (f32, f32),
    e1: Vec2,
    e2: Vec2,
) -> ClipVertex {
    let d = b - a;
    let edge = e2 - e1;
    let denom = edge.x * d.y - edge.y * d.x;
    let t = if denom.abs() > 1e-9 {
        (edge.x * (e1.y - a.y) - edge.y * (e1.x - a.x)) / denom
    } else {
        0.0
    };
    let pt = a + d * t;
    (
        pt,
        (
            a_uv.0 + (b_uv.0 - a_uv.0) * t,
            a_uv.1 + (b_uv.1 - a_uv.1) * t,
        ),
    )
}

/// Reverse `poly` to counter-clockwise winding (positive signed area) if
/// needed, so the clip half-plane tests treat its interior as "inside".
fn make_ccw(poly: &mut [Vec2]) {
    let n = poly.len();
    let mut area = 0.0;
    for i in 0..n {
        let a = poly[i];
        let b = poly[(i + 1) % n];
        area += a.x * b.y - b.x * a.y;
    }
    if area < 0.0 {
        poly.reverse();
    }
}

/// Whether `poly`'s corners all turn the same way (it is convex).
fn is_convex(poly: &[Vec2]) -> bool {
    let n = poly.len();
    let mut sign = 0.0_f32;
    for i in 0..n {
        let a = poly[i];
        let b = poly[(i + 1) % n];
        let c = poly[(i + 2) % n];
        let cross = (b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x);
        if cross.abs() > 1e-6 {
            if sign == 0.0 {
                sign = cross.signum();
            } else if cross.signum() != sign {
                return false;
            }
        }
    }
    true
}

/// Ear-clipping triangulation of a simple CCW polygon: each CCW triangle is
/// appended to `points` / `ranges`. `indices` is the reused working set.
fn triangulate_into(
    poly: &[Vec2],
    indices: &mut Vec<usize>,
    points: &mut Vec<Vec2>,
    ranges: &mut Vec<(usize, usize)>,
) {
    indices.clear();
    indices.extend(0..poly.len());
    while indices.len() > 3 {
        let m = indices.len();
        let mut ear = None;
        for i in 0..m {
            let a = poly[indices[(i + m - 1) % m]];
            let b = poly[indices[i]];
            let c = poly[indices[(i + 1) % m]];
            // A convex corner with no other vertex inside triangle a,b,c is an ear.
            if (b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x) <= 0.0 {
                continue;
            }
            let clear = indices.iter().enumerate().all(|(j, &vi)| {
                j == (i + m - 1) % m
                    || j == i
                    || j == (i + 1) % m
                    || !point_in_triangle(poly[vi], a, b, c)
            });
            if clear {
                ear = Some((i, [a, b, c]));
                break;
            }
        }
        let Some((i, tri)) = ear else {
            break; // degenerate polygon: stop with what we have
        };
        ranges.push((points.len(), 3));
        points.extend_from_slice(&tri);
        indices.remove(i);
    }
    if indices.len() == 3 {
        ranges.push((points.len(), 3));
        points.extend_from_slice(&[poly[indices[0]], poly[indices[1]], poly[indices[2]]]);
    }
}

/// Whether point `p` lies within triangle `a, b, c` (inclusive of edges).
fn point_in_triangle(p: Vec2, a: Vec2, b: Vec2, c: Vec2) -> bool {
    let d1 = tri_sign(p, a, b);
    let d2 = tri_sign(p, b, c);
    let d3 = tri_sign(p, c, a);
    let has_neg = d1 < 0.0 || d2 < 0.0 || d3 < 0.0;
    let has_pos = d1 > 0.0 || d2 > 0.0 || d3 > 0.0;
    !(has_neg && has_pos)
}

/// Signed area (doubled) of triangle `p, a, b`; its sign gives the turn side.
fn tri_sign(p: Vec2, a: Vec2, b: Vec2) -> f32 {
    (p.x - b.x) * (a.y - b.y) - (a.x - b.x) * (p.y - b.y)
}

#[cfg(test)]
mod tests {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::cell::Cell;
    use std::sync::Arc;

    use super::*;
    use crate::attach::{Attachment, ClippingAttachment, MeshAttachment, MeshVertices};
    use crate::data::{BlendMode, BoneData, Color, SkeletonData, SlotData};
    use crate::render::render_with;
    use crate::skel::Skeleton;
    use crate::skin::Skin;

    /// Counts this thread's heap allocations (alloc / realloc / alloc_zeroed)
    /// so a test can assert that a code path performs none. Thread-local, so
    /// parallel tests do not pollute each other's counts.
    struct CountingAlloc;

    thread_local! {
        static ALLOCATIONS: Cell<u64> = const { Cell::new(0) };
    }

    fn count_one() {
        // `try_with` so late allocations during thread teardown stay safe.
        let _ = ALLOCATIONS.try_with(|count| count.set(count.get() + 1));
    }

    fn allocation_count() -> u64 {
        ALLOCATIONS.with(Cell::get)
    }

    // SAFETY: delegates every operation directly to `System`; the counter is a
    // const-initialized thread-local `Cell`, which does not itself allocate.
    unsafe impl GlobalAlloc for CountingAlloc {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            count_one();
            System.alloc(layout)
        }

        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            System.dealloc(ptr, layout);
        }

        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            count_one();
            System.realloc(ptr, layout, new_size)
        }

        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            count_one();
            System.alloc_zeroed(layout)
        }
    }

    #[global_allocator]
    static COUNTING: CountingAlloc = CountingAlloc;

    fn square() -> Vec<Vec2> {
        vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(10.0, 0.0),
            Vec2::new(10.0, 10.0),
            Vec2::new(0.0, 10.0),
        ]
    }

    /// Run [`clip_triangle`] with throwaway buffers, returning the result.
    fn clip_one(p: &[Vec2; 3], uv: &[(f32, f32); 3], polygon: &[Vec2]) -> Vec<ClipVertex> {
        let mut subject = Vec::new();
        let mut clipped = Vec::new();
        clip_triangle(&mut subject, &mut clipped, p, uv, polygon);
        subject
    }

    #[test]
    fn clip_triangle_inside_is_unchanged() {
        let p = [
            Vec2::new(2.0, 2.0),
            Vec2::new(8.0, 2.0),
            Vec2::new(2.0, 8.0),
        ];
        let uv = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)];
        assert_eq!(clip_one(&p, &uv, &square()).len(), 3);
    }

    #[test]
    fn clip_triangle_crosses_right_edge() {
        // A=(5,5) inside, B=(15,5) outside (x>10), C=(5,8) inside.
        let p = [
            Vec2::new(5.0, 5.0),
            Vec2::new(15.0, 5.0),
            Vec2::new(5.0, 8.0),
        ];
        let uv = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)];
        let clipped = clip_one(&p, &uv, &square());
        // Clipped to four vertices including the two right-edge crossings.
        assert_eq!(clipped.len(), 4);
        // The (10,5) crossing lies halfway along A->B, so its UV is (0.5, 0).
        let at = clipped
            .iter()
            .find(|(pt, _)| (pt.x - 10.0).abs() < 1e-3 && (pt.y - 5.0).abs() < 1e-3)
            .expect("a vertex at (10,5)");
        let &(_, (u, v)) = at;
        assert!((u - 0.5).abs() < 1e-3, "u={u}");
        assert!(v.abs() < 1e-3, "v={v}");
        // Nothing escapes the clip square.
        for (pt, _) in &clipped {
            assert!(pt.x <= 10.0 + 1e-3, "{pt:?}");
        }
    }

    #[test]
    fn convex_clip_stays_one_piece() {
        let mut world = square();
        assert!(is_convex(&world));
        let (mut ear, mut points, mut ranges) = (Vec::new(), Vec::new(), Vec::new());
        assert!(replace_clip(&mut world, &mut ear, &mut points, &mut ranges));
        assert_eq!(ranges, vec![(0, 4)]);
        assert_eq!(points.len(), 4);
    }

    #[test]
    fn degenerate_clip_keeps_the_previous_pieces() {
        let (mut ear, mut points, mut ranges) = (Vec::new(), Vec::new(), Vec::new());
        assert!(replace_clip(
            &mut square(),
            &mut ear,
            &mut points,
            &mut ranges
        ));
        // A two-vertex "polygon" yields nothing: the square stays active.
        let mut line = vec![Vec2::ZERO, Vec2::new(5.0, 0.0)];
        assert!(!replace_clip(&mut line, &mut ear, &mut points, &mut ranges));
        assert_eq!(ranges, vec![(0, 4)]);
        // A real triangle replaces it, rebased to the buffer's start.
        let mut tri = vec![Vec2::ZERO, Vec2::new(5.0, 0.0), Vec2::new(0.0, 5.0)];
        assert!(replace_clip(&mut tri, &mut ear, &mut points, &mut ranges));
        assert_eq!(ranges, vec![(0, 3)]);
        assert_eq!(points.len(), 3);
    }

    #[test]
    fn concave_clip_decomposes_and_excludes_the_notch() {
        // L-shape (concave); the top-right region (x>4, y>4) is outside it.
        let mut l = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(10.0, 0.0),
            Vec2::new(10.0, 4.0),
            Vec2::new(4.0, 4.0),
            Vec2::new(4.0, 10.0),
            Vec2::new(0.0, 10.0),
        ];
        assert!(!is_convex(&l));
        let (mut ear, mut points, mut ranges) = (Vec::new(), Vec::new(), Vec::new());
        assert!(replace_clip(&mut l, &mut ear, &mut points, &mut ranges));
        assert!(
            ranges.len() >= 2,
            "concave clip should decompose: {}",
            ranges.len()
        );
        let uv = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)];
        let pieces: Vec<&[Vec2]> = ranges
            .iter()
            .map(|&(start, len)| &points[start..start + len])
            .collect();
        // A triangle entirely in the notch survives against no piece.
        let notch = [
            Vec2::new(6.0, 6.0),
            Vec2::new(9.0, 6.0),
            Vec2::new(6.0, 9.0),
        ];
        let hits = pieces
            .iter()
            .filter(|poly| clip_one(&notch, &uv, poly).len() >= 3)
            .count();
        assert_eq!(hits, 0, "notch triangle must be fully clipped");
        // A triangle inside the L survives against at least one piece.
        let inside = [
            Vec2::new(1.0, 1.0),
            Vec2::new(3.0, 1.0),
            Vec2::new(1.0, 3.0),
        ];
        let hits2 = pieces
            .iter()
            .filter(|poly| clip_one(&inside, &uv, poly).len() >= 3)
            .count();
        assert!(hits2 >= 1, "inside triangle must survive");
    }

    /// A skeleton with a concave (L-shaped) clip on slot 0 masking a mesh
    /// triangle on slot 1 that spills past it: every clip-path branch runs
    /// (ear-clipping decomposition, multi-piece Sutherland-Hodgman clipping).
    fn clipped_skeleton() -> Skeleton {
        let mut skin = Skin::new("default");
        skin.set(
            0,
            "clip",
            Attachment::Clipping(ClippingAttachment::new(
                "clip",
                "m",
                MeshVertices::Unweighted(vec![
                    0.0, 0.0, 10.0, 0.0, 10.0, 4.0, 4.0, 4.0, 4.0, 10.0, 0.0, 10.0,
                ]),
                6,
            )),
        );
        skin.set(
            1,
            "tri",
            Attachment::Mesh(MeshAttachment::new(
                "tri",
                "tri",
                MeshVertices::Unweighted(vec![1.0, 1.0, 30.0, 1.0, 1.0, 30.0]),
                vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                vec![0, 1, 2],
            )),
        );
        let data = SkeletonData {
            bones: vec![BoneData {
                index: 0,
                name: "root".into(),
                ..Default::default()
            }],
            slots: vec![
                SlotData {
                    index: 0,
                    name: "clipslot".into(),
                    bone: 0,
                    color: Color::WHITE,
                    dark_color: None,
                    attachment: Some("clip".into()),
                    blend: BlendMode::Normal,
                },
                SlotData {
                    index: 1,
                    name: "m".into(),
                    bone: 0,
                    color: Color::WHITE,
                    dark_color: None,
                    attachment: Some("tri".into()),
                    blend: BlendMode::Normal,
                },
            ],
            default_skin: skin,
            ..Default::default()
        };
        let mut sk = Skeleton::new(Arc::new(data));
        sk.update_world_transform();
        sk
    }

    // The zero-allocation guarantee: after warm-up frames have grown every
    // scratch buffer, rendering the clipped skeleton again must not touch the
    // heap at all (the counting allocator tallies this thread's allocations).
    #[test]
    fn steady_state_clipped_render_allocates_nothing() {
        let sk = clipped_skeleton();
        let mut scratch = RenderScratch::new();
        let first = render_with(&sk, &mut scratch).len();
        assert!(first > 0, "the clipped mesh must draw");
        let _ = render_with(&sk, &mut scratch);

        let before = allocation_count();
        let commands = render_with(&sk, &mut scratch);
        let after = allocation_count();
        assert_eq!(commands.len(), first);
        assert_eq!(
            after - before,
            0,
            "steady-state clipped rendering must not allocate"
        );
    }
}
