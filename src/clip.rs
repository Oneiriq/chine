//! Clipping geometry and the reusable render scratch.
//!
//! Spine's clipping attachment masks the slots that follow it: the clip
//! polygon is decomposed into convex CCW pieces (the polygon itself when
//! convex, an ear-clipping triangulation otherwise), and every drawn triangle
//! is Sutherland-Hodgman-clipped against each piece.
//!
//! All of that needs working buffers. [`RenderScratch`] owns them, plus the
//! emitted commands themselves, so a render loop that keeps one scratch and
//! calls [`render_with`](crate::render::render_with) every frame reuses every
//! buffer instead of reallocating it.
//!
//! Clipping work grows with the clipped triangles times the clip pieces, and
//! decomposing a concave polygon grows with its vertex count squared. A file
//! can declare large values for both cheaply, so each frame's clipping draws
//! on a fixed work budget, [`CLIP_WORK_BUDGET`].

use std::mem;

use glam::Vec2;

use crate::render::RenderCommand;

/// The clipping work one frame may do, counted in polygon corners visited
/// while decomposing clip polygons and in clip-piece edges applied to drawn
/// triangles. Real rigs use a tiny fraction of it. Once it is spent, the
/// clip polygon decomposition stops early and the frame's remaining clipped
/// geometry is not drawn.
pub(crate) const CLIP_WORK_BUDGET: usize = 1 << 24;

/// A clip vertex: a world-space position with its interpolated UV.
pub(crate) type ClipVertex = (Vec2, (f32, f32));

/// Take `cost` units of work from `budget`. Returns `false`, and empties the
/// budget, when it holds less than that.
pub(crate) fn spend(budget: &mut usize, cost: usize) -> bool {
    match budget.checked_sub(cost) {
        Some(rest) => {
            *budget = rest;
            true
        }
        None => {
            *budget = 0;
            false
        }
    }
}

/// Reusable buffers for [`render_with`](crate::render::render_with).
///
/// Holds the emitted commands (rebuilt in place each frame) and every
/// intermediate the clipper needs: the clip attachment's world polygon, its
/// decomposed convex pieces, the ear-clipping working set, and the
/// Sutherland-Hodgman ping-pong buffers. Keep one scratch per render loop.
/// Once its buffers have grown to the skeleton's working sizes, steady-state
/// rendering of a clipped skeleton performs no heap allocation. Buffers only
/// grow. Dropping the scratch releases them.
#[derive(Debug, Default)]
pub struct RenderScratch {
    /// Grow-only pool of output commands. A frame's commands are the prefix
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
    /// An empty scratch. Its buffers grow on first use.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

/// [`replace_clip_within`] with no work limit.
#[cfg(test)]
pub(crate) fn replace_clip(
    world: &mut [Vec2],
    ear: &mut Vec<usize>,
    points: &mut Vec<Vec2>,
    ranges: &mut Vec<(usize, usize)>,
) -> bool {
    let mut unlimited = usize::MAX;
    replace_clip_within(world, ear, points, ranges, &mut unlimited)
}

/// Decompose `world` into convex CCW pieces appended to `points` / `ranges`,
/// replacing the pieces already there (the previous clip). A polygon with
/// fewer than three vertices, or one that yields no piece (degenerate, or out
/// of `budget` before the first piece), leaves the current pieces in place and
/// returns `false`.
pub(crate) fn replace_clip_within(
    world: &mut [Vec2],
    ear: &mut Vec<usize>,
    points: &mut Vec<Vec2>,
    ranges: &mut Vec<(usize, usize)>,
    budget: &mut usize,
) -> bool {
    // Winding and convexity each visit every corner once.
    if world.len() < 3 || !spend(budget, world.len()) {
        return false;
    }
    make_ccw(world);
    let points_mark = points.len();
    let ranges_mark = ranges.len();
    if is_convex(world) {
        ranges.push((points.len(), world.len()));
        points.extend_from_slice(world);
    } else {
        triangulate_into(world, ear, points, ranges, budget);
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
///
/// A triangle with a corner that has no position or UV pair in `src` is
/// skipped. Clipping stops when `budget` runs out (each piece costs one unit
/// per edge), or once the next clipped polygon would not fit in the `u16`
/// range that output triangles index.
pub(crate) fn clip_into(
    src: &RenderCommand,
    points: &[Vec2],
    ranges: &[(usize, usize)],
    subject: &mut Vec<ClipVertex>,
    clipped: &mut Vec<ClipVertex>,
    dst: &mut RenderCommand,
    budget: &mut usize,
) -> bool {
    dst.positions.clear();
    dst.uvs.clear();
    dst.triangles.clear();
    'triangles: for tri in src.triangles.chunks(3) {
        let Ok(tri) = <&[u16; 3]>::try_from(tri) else {
            continue;
        };
        let Some([a, b, c]) = corners(src, tri) else {
            continue;
        };
        let p = [a.0, b.0, c.0];
        let uv = [a.1, b.1, c.1];
        for &(start, len) in ranges {
            let Some(polygon) = start
                .checked_add(len)
                .and_then(|end| points.get(start..end))
            else {
                continue;
            };
            if !spend(budget, polygon.len()) {
                break 'triangles;
            }
            clip_triangle(subject, clipped, &p, &uv, polygon);
            if subject.len() < 3 {
                continue;
            }
            // The polygon's last vertex index must fit in u16.
            let (Ok(base), Ok(count)) = (
                u16::try_from(dst.positions.len()),
                u16::try_from(subject.len()),
            ) else {
                break 'triangles;
            };
            if base.checked_add(count - 1).is_none() {
                break 'triangles;
            }
            for (pt, (u, v)) in subject.iter() {
                dst.positions.push(*pt);
                dst.uvs.push(*u);
                dst.uvs.push(*v);
            }
            // `base + count - 1` fits in u16, so none of these sums overflow.
            for k in 1..count - 1 {
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

/// The position and UV of each corner of triangle `tri` in `src`, or `None` if
/// a corner indexes past the positions or UVs.
fn corners(src: &RenderCommand, tri: &[u16; 3]) -> Option<[ClipVertex; 3]> {
    let corner = |t: u16| -> Option<ClipVertex> {
        let i = usize::from(t);
        let pos = *src.positions.get(i)?;
        let &[u, v] = src.uvs.get(i * 2..)?.first_chunk::<2>()?;
        Some((pos, (u, v)))
    };
    let [a, b, c] = *tri;
    Some([corner(a)?, corner(b)?, corner(c)?])
}

/// Sutherland-Hodgman clip of one triangle against a convex polygon (CCW),
/// leaving the clipped polygon's vertices (with interpolated UVs) in
/// `subject`. `clipped` is the per-edge ping-pong buffer. Fewer than three
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
/// Each corner visited costs one unit of `budget` and each ear test one unit
/// per remaining vertex. When the budget runs out, triangulation stops with
/// the triangles made so far.
fn triangulate_into(
    poly: &[Vec2],
    indices: &mut Vec<usize>,
    points: &mut Vec<Vec2>,
    ranges: &mut Vec<(usize, usize)>,
    budget: &mut usize,
) {
    indices.clear();
    indices.extend(0..poly.len());
    'clipping: while indices.len() > 3 {
        let m = indices.len();
        let mut ear = None;
        for i in 0..m {
            if !spend(budget, 1) {
                break 'clipping;
            }
            let a = poly[indices[(i + m - 1) % m]];
            let b = poly[indices[i]];
            let c = poly[indices[(i + 1) % m]];
            // A convex corner with no other vertex inside triangle a,b,c is an ear.
            if (b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x) <= 0.0 {
                continue;
            }
            if !spend(budget, m) {
                break 'clipping;
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

/// Signed area (doubled) of triangle `p, a, b`. Its sign gives the turn side.
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

    // SAFETY: delegates every operation directly to `System`. The counter is a
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

    /// A command drawing triangle `(1,1) (3,1) (1,3)` with `triangles`.
    fn small_triangle(triangles: Vec<u16>) -> RenderCommand {
        RenderCommand {
            positions: vec![
                Vec2::new(1.0, 1.0),
                Vec2::new(3.0, 1.0),
                Vec2::new(1.0, 3.0),
            ],
            uvs: vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            triangles,
            ..Default::default()
        }
    }

    /// Run [`clip_into`] against the pieces of `clip` with `budget`.
    fn clip_command(src: &RenderCommand, clip: Vec<Vec2>, budget: &mut usize) -> RenderCommand {
        let mut world = clip;
        let (mut ear, mut points, mut ranges) = (Vec::new(), Vec::new(), Vec::new());
        assert!(replace_clip(&mut world, &mut ear, &mut points, &mut ranges));
        let mut dst = RenderCommand::default();
        let (mut subject, mut clipped) = (Vec::new(), Vec::new());
        clip_into(
            src,
            &points,
            &ranges,
            &mut subject,
            &mut clipped,
            &mut dst,
            budget,
        );
        dst
    }

    /// [`clip_command`] with no work limit.
    fn clip_unbounded(src: &RenderCommand, clip: Vec<Vec2>) -> RenderCommand {
        let mut unlimited = usize::MAX;
        clip_command(src, clip, &mut unlimited)
    }

    // Triangles that index past the command's positions or UVs used to index
    // out of bounds. They are skipped, and the valid triangle is still clipped.
    #[test]
    fn clip_into_skips_triangles_past_the_vertices() {
        let mut src = small_triangle(vec![0, 1, 2, 0, 1, 3, 9, 9, 9, 0]);
        let out = clip_unbounded(&src, square());
        assert_eq!(out.triangles, vec![0, 1, 2]);
        src.uvs.truncate(5);
        let out = clip_unbounded(&src, square());
        assert!(out.triangles.is_empty());
    }

    // 22,000 copies of one triangle clip to 66,000 vertices, more than u16
    // indices can address. The index base used to wrap and then overflow.
    // Output now stops at the last polygon that fits.
    #[test]
    fn clip_output_stops_at_the_u16_index_limit() {
        let src = small_triangle([0, 1, 2].repeat(22_000));
        let out = clip_unbounded(&src, square());
        assert!(out.positions.len() <= 1 << 16);
        assert_eq!(out.positions.len(), 65_535);
        assert!(out
            .triangles
            .iter()
            .all(|&t| usize::from(t) < out.positions.len()));
    }

    // NaN in the clip polygon or the drawn triangle must not panic or loop.
    #[test]
    fn nan_clip_geometry_does_not_panic() {
        let nan = Vec2::new(f32::NAN, 0.0);
        let mut concave_nan = vec![
            Vec2::ZERO,
            Vec2::new(10.0, 0.0),
            nan,
            Vec2::new(4.0, 4.0),
            Vec2::new(0.0, 10.0),
        ];
        let (mut ear, mut points, mut ranges) = (Vec::new(), Vec::new(), Vec::new());
        let _ = replace_clip(&mut concave_nan, &mut ear, &mut points, &mut ranges);
        let mut src = small_triangle(vec![0, 1, 2]);
        src.positions[1] = nan;
        let _ = clip_unbounded(&src, square());
        let mut all_nan = vec![nan; 5];
        let _ = replace_clip(&mut all_nan, &mut ear, &mut points, &mut ranges);
    }

    // A concave clip polygon costs its vertex count squared to decompose, and
    // clipping costs triangles times pieces. Both stop when the frame's budget
    // runs out, keeping what was done.
    #[test]
    fn clip_work_stops_when_the_budget_runs_out() {
        let comb: Vec<Vec2> = (0..400)
            .map(|i: u16| {
                let x = f32::from(i % 200);
                if i < 200 {
                    Vec2::new(x, if i.is_multiple_of(2) { 0.0 } else { 10.0 })
                } else {
                    Vec2::new(199.0 - x, -10.0)
                }
            })
            .collect();
        let (mut ear, mut points, mut ranges) = (Vec::new(), Vec::new(), Vec::new());
        let mut full = comb.clone();
        assert!(replace_clip(&mut full, &mut ear, &mut points, &mut ranges));
        let pieces = ranges.len();
        let mut budget = 5_000;
        let mut partial = comb;
        assert!(replace_clip_within(
            &mut partial,
            &mut ear,
            &mut points,
            &mut ranges,
            &mut budget
        ));
        assert_eq!(budget, 0);
        assert!(ranges.len() < pieces);

        let src = small_triangle([0, 1, 2].repeat(100));
        let mut budget = 4 * 10;
        let out = clip_command(&src, square(), &mut budget);
        assert_eq!(out.triangles.len(), 10 * 3);
        assert_eq!(budget, 0);
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
        // L-shape (concave). The top-right region (x>4, y>4) is outside it.
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
