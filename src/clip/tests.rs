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
        false,
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
    clipped_skeleton_with(false, false)
}

/// [`clipped_skeleton`] with the clip's `convex` and `inverse` flags.
fn clipped_skeleton_with(convex: bool, inverse: bool) -> Skeleton {
    let mut clip = ClippingAttachment::new(
        "clip",
        "m",
        MeshVertices::Unweighted(vec![
            0.0, 0.0, 10.0, 0.0, 10.0, 4.0, 4.0, 4.0, 4.0, 10.0, 0.0, 10.0,
        ]),
        6,
    );
    clip.convex = convex;
    clip.inverse = inverse;
    let mut skin = Skin::new("default");
    skin.set(0, "clip", Attachment::Clipping(clip));
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

/// The area the triangles of `commands` cover (their areas summed).
fn drawn_area(commands: &[RenderCommand]) -> f32 {
    let mut area = 0.0;
    for c in commands {
        for t in c.triangles.chunks(3) {
            let [a, b, d] = [0, 1, 2].map(|k| c.positions[usize::from(t[k])]);
            area += ((b - a).perp_dot(d - a) / 2.0).abs();
        }
    }
    area
}

#[test]
fn convex_hull_drops_inner_and_collinear_corners() {
    let l = [
        Vec2::new(0.0, 0.0),
        Vec2::new(5.0, 0.0),
        Vec2::new(10.0, 0.0),
        Vec2::new(10.0, 4.0),
        Vec2::new(4.0, 4.0),
        Vec2::new(4.0, 10.0),
        Vec2::new(0.0, 10.0),
    ];
    let (mut order, mut points) = (Vec::new(), Vec::new());
    assert_eq!(convex_hull_into(&l, &mut order, &mut points), 5);
    let expected = [
        (0.0, 0.0),
        (10.0, 0.0),
        (10.0, 4.0),
        (4.0, 10.0),
        (0.0, 10.0),
    ];
    assert_eq!(points, expected.map(|(x, y)| Vec2::new(x, y)));
    // A line has no hull.
    points.clear();
    let line = [Vec2::ZERO, Vec2::new(1.0, 1.0), Vec2::new(2.0, 2.0)];
    assert!(convex_hull_into(&line, &mut order, &mut points) < 3);
}

// Spine clips to the convex hull of a clip polygon marked convex, and keeps
// what lies outside that hull for a clip marked inverse. chine read neither
// flag and always clipped to the inside of the exact polygon.
#[test]
fn convex_and_inverse_clips_follow_their_flags() {
    let area = |convex, inverse| {
        let sk = clipped_skeleton_with(convex, inverse);
        drawn_area(&crate::render::render(&sk))
    };
    // The mesh covers 45 of the L-shaped clip, and 63 of its hull.
    assert!(
        (area(false, false) - 45.0).abs() < 1e-2,
        "{}",
        area(false, false)
    );
    assert!(
        (area(true, false) - 63.0).abs() < 1e-2,
        "{}",
        area(true, false)
    );
    // The mesh's area is 420.5, so 357.5 of it lies outside the hull.
    assert!(
        (area(false, true) - 357.5).abs() < 1e-2,
        "{}",
        area(false, true)
    );
    assert!(
        (area(true, true) - 357.5).abs() < 1e-2,
        "{}",
        area(true, true)
    );

    // No part of an inverse clip's output lies inside the hull.
    let sk = clipped_skeleton_with(false, true);
    let commands = crate::render::render(&sk);
    let inside_hull =
        |p: Vec2| p.x > 0.0 && p.y > 0.0 && p.x < 10.0 && p.y < 10.0 && p.x + p.y < 14.0;
    for c in &commands {
        for t in c.triangles.chunks(3) {
            let centroid = t.iter().map(|&i| c.positions[usize::from(i)]).sum::<Vec2>() / 3.0;
            assert!(!inside_hull(centroid), "{centroid:?}");
        }
    }
}

// An inverse clip drops a triangle inside its polygon, keeps one outside it
// whole, and interpolates UVs where it cuts.
#[test]
fn inverse_clip_of_single_triangles() {
    let clip_outside = |src: &RenderCommand| {
        let mut dst = RenderCommand::default();
        let (mut subject, mut clipped, mut outside) = (Vec::new(), Vec::new(), Vec::new());
        let mut budget = usize::MAX;
        clip_outside_into(
            src,
            &square(),
            &mut subject,
            &mut clipped,
            &mut outside,
            &mut dst,
            &mut budget,
        );
        dst
    };
    assert!(clip_outside(&small_triangle(vec![0, 1, 2]))
        .triangles
        .is_empty());

    let mut far = small_triangle(vec![0, 1, 2]);
    for p in &mut far.positions {
        *p += Vec2::new(20.0, 0.0);
    }
    assert_eq!(clip_outside(&far).positions, far.positions);

    // UVs follow x / 20 and y / 20 across a triangle that covers the square.
    let big = RenderCommand {
        positions: vec![
            Vec2::new(-10.0, -10.0),
            Vec2::new(30.0, -10.0),
            Vec2::new(-10.0, 30.0),
        ],
        uvs: vec![-0.5, -0.5, 1.5, -0.5, -0.5, 1.5],
        triangles: vec![0, 1, 2],
        ..Default::default()
    };
    let out = clip_outside(&big);
    assert!((drawn_area(std::slice::from_ref(&out)) - 700.0).abs() < 1e-2);
    for (p, uv) in out.positions.iter().zip(out.uvs.chunks(2)) {
        assert!((uv[0] - p.x / 20.0).abs() < 1e-4, "{p:?} {uv:?}");
        assert!((uv[1] - p.y / 20.0).abs() < 1e-4, "{p:?} {uv:?}");
    }
}

// Convex and inverse clips reuse the scratch buffers too.
#[test]
fn steady_state_hull_clips_allocate_nothing() {
    for (convex, inverse) in [(true, false), (false, true)] {
        let sk = clipped_skeleton_with(convex, inverse);
        let mut scratch = RenderScratch::new();
        let first = render_with(&sk, &mut scratch).len();
        assert!(first > 0);
        let _ = render_with(&sk, &mut scratch);
        let before = allocation_count();
        let _ = render_with(&sk, &mut scratch);
        assert_eq!(allocation_count() - before, 0, "{convex} {inverse}");
    }
}

// Spine clips with the clipping polygon moved by its slot's deform. The
// renderer used the bind-pose polygon.
#[test]
fn clipping_follows_its_deform() {
    let mut sk = clipped_skeleton();
    assert!(!crate::render::render(&sk).is_empty());
    // Move the whole clip polygon far from the mesh.
    let far: Vec<f32> = [
        0.0, 0.0, 10.0, 0.0, 10.0, 4.0, 4.0, 4.0, 4.0, 10.0, 0.0, 10.0,
    ]
    .iter()
    .map(|v| v + 1000.0)
    .collect();
    sk.slot_pose_and_setup(0).unwrap().0.deform = far;
    assert!(crate::render::render(&sk).is_empty());
}

// Spine ignores a clipping attachment that starts while another clip is
// active. The renderer replaced the active clip with it.
#[test]
fn a_clip_inside_an_active_clip_is_ignored() {
    let sk = clipped_skeleton();
    let mut data = sk.data().clone();
    // A second clip, far from the mesh, on a new slot drawn between the
    // first clip and the mesh.
    let far = ClippingAttachment::new(
        "far",
        "m",
        MeshVertices::Unweighted(vec![100.0, 100.0, 110.0, 100.0, 100.0, 110.0]),
        3,
    );
    data.default_skin.set(2, "far", Attachment::Clipping(far));
    data.slots.push(SlotData {
        index: 2,
        name: "farslot".into(),
        bone: 0,
        color: Color::WHITE,
        dark_color: None,
        attachment: Some("far".into()),
        blend: BlendMode::Normal,
    });
    let mut sk = Skeleton::new(Arc::new(data));
    sk.draw_order_mut().copy_from_slice(&[0, 2, 1]);
    sk.update_world_transform();
    let drawn = drawn_area(&crate::render::render(&sk));
    assert!((drawn - 45.0).abs() < 1e-2, "{drawn}");
}
