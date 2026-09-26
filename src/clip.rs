//! Clipping geometry and the reusable render scratch.
//!
//! Spine's clipping attachment masks the slots that follow it: the clip
//! polygon is decomposed into convex CCW pieces (the polygon itself when
//! convex, an ear-clipping triangulation otherwise), and every drawn triangle
//! is Sutherland-Hodgman-clipped against each piece. A clip marked convex
//! uses the polygon's convex hull as its one piece. A clip marked inverse
//! keeps what lies outside that hull instead.
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
    /// An inverse clip's fragment: the part of a triangle outside one edge.
    pub(crate) outside: Vec<ClipVertex>,
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
    replace_clip_within(world, false, ear, points, ranges, &mut unlimited)
}

/// Decompose `world` into convex CCW pieces appended to `points` / `ranges`,
/// replacing the pieces already there (the previous clip). With `hull`, a
/// concave polygon becomes one piece, its convex hull, as Spine does for a
/// clip marked convex or inverse. A polygon with fewer than three vertices,
/// or one that yields no piece (degenerate, or out of `budget` before the
/// first piece), leaves the current pieces in place and returns `false`.
pub(crate) fn replace_clip_within(
    world: &mut [Vec2],
    hull: bool,
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
    } else if hull {
        // Sorting the corners costs about one more visit each.
        if spend(budget, world.len()) {
            let len = convex_hull_into(world, ear, points);
            if len >= 3 {
                ranges.push((points.len() - len, len));
            } else {
                points.truncate(points_mark);
            }
        }
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
            if subject.len() >= 3 && !push_fan(dst, subject) {
                break 'triangles;
            }
        }
    }
    finish(src, dst)
}

/// Clip `src`'s triangles to the outside of the convex CCW `polygon`, for an
/// inverse clip: each triangle keeps the parts that lie outside the polygon,
/// as convex fragments with interpolated UVs, in `dst` (whose buffers are
/// cleared and reused). Returns `false`, leaving `dst`'s buffers empty, if
/// nothing survives.
///
/// Each edge of the polygon splits the rest of the triangle: the part outside
/// the edge is a fragment, and the part inside goes on to the next edge. What
/// is inside every edge is inside the polygon, and is dropped. Skipped
/// triangles, the budget, and the `u16` limit work as in [`clip_into`].
pub(crate) fn clip_outside_into(
    src: &RenderCommand,
    polygon: &[Vec2],
    subject: &mut Vec<ClipVertex>,
    clipped: &mut Vec<ClipVertex>,
    outside: &mut Vec<ClipVertex>,
    dst: &mut RenderCommand,
    budget: &mut usize,
) -> bool {
    dst.positions.clear();
    dst.uvs.clear();
    dst.triangles.clear();
    let m = polygon.len();
    'triangles: for tri in src.triangles.chunks(3) {
        let Ok(tri) = <&[u16; 3]>::try_from(tri) else {
            continue;
        };
        let Some(corners) = corners(src, tri) else {
            continue;
        };
        if !spend(budget, m) {
            break;
        }
        subject.clear();
        subject.extend_from_slice(&corners);
        for i in 0..m {
            split_by_edge(subject, clipped, outside, polygon[i], polygon[(i + 1) % m]);
            if outside.len() >= 3 && !push_fan(dst, outside) {
                break 'triangles;
            }
            mem::swap(subject, clipped);
            if subject.len() < 3 {
                break;
            }
        }
    }
    finish(src, dst)
}

/// Append convex polygon `polygon` to `dst` as a triangle fan. Returns
/// `false`, appending nothing, when its last vertex index would not fit in
/// the `u16` range that triangles index.
fn push_fan(dst: &mut RenderCommand, polygon: &[ClipVertex]) -> bool {
    let (Ok(base), Ok(count)) = (
        u16::try_from(dst.positions.len()),
        u16::try_from(polygon.len()),
    ) else {
        return false;
    };
    if base.checked_add(count - 1).is_none() {
        return false;
    }
    for (pt, (u, v)) in polygon {
        dst.positions.push(*pt);
        dst.uvs.push(*u);
        dst.uvs.push(*v);
    }
    // `base + count - 1` fits in u16, so none of these sums overflow.
    for k in 1..count - 1 {
        dst.triangles
            .extend_from_slice(&[base, base + k, base + k + 1]);
    }
    true
}

/// Copy `src`'s tint, texture, and blend onto clipped command `dst`. Returns
/// `false` when no triangle survived the clip.
fn finish(src: &RenderCommand, dst: &mut RenderCommand) -> bool {
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

/// Split a subject polygon along the line of directed edge `e1 -> e2` into
/// its part on the inside (left) half-plane, in `inside`, and its part on the
/// outside half-plane, in `outside`. Both are cleared first. Points on the
/// line count as inside, as in [`clip_against_edge`]. The crossings, with
/// interpolated UVs, belong to both parts.
fn split_by_edge(
    subject: &[ClipVertex],
    inside: &mut Vec<ClipVertex>,
    outside: &mut Vec<ClipVertex>,
    e1: Vec2,
    e2: Vec2,
) {
    inside.clear();
    outside.clear();
    let n = subject.len();
    let edge = e2 - e1;
    let is_inside = |pt: Vec2| edge.x * (pt.y - e1.y) - edge.y * (pt.x - e1.x) >= 0.0;
    for i in 0..n {
        let (cur, cur_uv) = subject[i];
        let (prev, prev_uv) = subject[(i + n - 1) % n];
        let cur_in = is_inside(cur);
        if cur_in != is_inside(prev) {
            let crossing = edge_intersect(prev, prev_uv, cur, cur_uv, e1, e2);
            inside.push(crossing);
            outside.push(crossing);
        }
        if cur_in {
            inside.push((cur, cur_uv));
        } else {
            outside.push((cur, cur_uv));
        }
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

/// Append the convex hull of `poly` to `points`, counter-clockwise, and
/// return its vertex count (fewer than three for a degenerate polygon).
/// Collinear corners are dropped. `order` is a reused buffer for the corners
/// sorted by position.
fn convex_hull_into(poly: &[Vec2], order: &mut Vec<usize>, points: &mut Vec<Vec2>) -> usize {
    order.clear();
    order.extend(0..poly.len());
    order.sort_unstable_by(|&a, &b| {
        let (p, q) = (poly[a], poly[b]);
        p.x.total_cmp(&q.x).then(p.y.total_cmp(&q.y))
    });
    // Add corner `c` to the chain that starts at index `base`, first dropping
    // the chain's last corners while they do not turn left toward `c`.
    let add = |points: &mut Vec<Vec2>, base: usize, c: Vec2| {
        while let [.., a, b] = points[base..] {
            if (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x) > 0.0 {
                break;
            }
            points.pop();
        }
        points.push(c);
    };
    let start = points.len();
    // The lower hull left to right, then the upper hull right to left from
    // the rightmost corner, which ends the lower hull.
    for &i in order.iter() {
        add(points, start, poly[i]);
    }
    let base = points.len() - 1;
    for &i in order.iter().rev().skip(1) {
        add(points, base, poly[i]);
    }
    // The upper hull ends on the first corner again.
    points.pop();
    points.len().saturating_sub(start)
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
mod tests;
