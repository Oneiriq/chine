use super::*;

use crate::anim::GLOBAL_PHYSICS;

mod deform;
use deform::{
    deform_mesh_info, read_deform_timeline, read_sequence_timeline, sequence_count,
    skip_deform_timeline,
};

/// Wrap a parsed bone timeline in its [`Timeline`] variant.
fn wrap((tl, d): (BoneTimeline, f32), make: fn(BoneTimeline) -> Timeline) -> (Timeline, f32) {
    (make(tl), d)
}

/// Wrap a parsed single-axis bone timeline.
fn axis((tl, d): (BoneTimeline, f32), a: BoneAxis) -> (Timeline, f32) {
    (Timeline::BoneAxis(tl, a), d)
}

/// Read a timeline's keyframe count. Spine never writes an empty timeline (its
/// own reader cannot load one), so zero frames is corrupt.
fn frame_count(r: &mut BinaryReader) -> usize {
    let frames = r.count();
    if frames == 0 {
        corrupt(r);
    }
    frames
}

/// A keyframe count for a timeline whose frames each store at least
/// `frame_bytes` bytes. A count the remaining data cannot hold is corrupt and
/// reads as zero frames.
fn fitting_frames(r: &mut BinaryReader, frames: usize, frame_bytes: usize) -> usize {
    if frames.saturating_mul(frame_bytes) > r.remaining() {
        corrupt(r);
        return 0;
    }
    frames
}

/// The frame count and Bezier capacity to build a curve timeline with.
///
/// A curve timeline has at least one frame, each storing at least
/// `frame_bytes` bytes. The declared Bezier count sizes the curve storage, but
/// a timeline never uses more than `per_segment` Bezier curves between two
/// frames, so the storage is capped there. On a corrupt frame count this
/// records the error and sizes a one-frame placeholder, which the failed load
/// then discards.
fn curve_sizes(
    r: &mut BinaryReader,
    frames: usize,
    bezier_count: usize,
    frame_bytes: usize,
    per_segment: usize,
) -> (usize, usize) {
    match fitting_frames(r, frames, frame_bytes) {
        0 => {
            corrupt(r);
            (1, 0)
        }
        n => (n, bezier_count.min((n - 1).saturating_mul(per_segment))),
    }
}

/// Claim the next Bezier slot. The curve storage holds `capacity` Bezier
/// curves (Spine sizes it from the declared count), so a curve past it is
/// corrupt: this records the error and returns `None`, and the caller skips
/// that curve.
fn next_bezier(r: &mut BinaryReader, bezier: &mut usize, capacity: usize) -> Option<usize> {
    let index = *bezier;
    *bezier += 1;
    if index < capacity {
        Some(index)
    } else {
        corrupt(r);
        None
    }
}

/// A constraint type in Spine's single constraint list.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ConstraintKind {
    Ik,
    Path,
    Transform,
    Physics,
    Slider,
}

/// Spine's single constraint list: for each position, the constraint's type
/// and its index in chine's list for that type. A constraint's `order` is its
/// position in the single list.
pub(super) fn constraint_list(data: &SkeletonData) -> Vec<Option<(ConstraintKind, usize)>> {
    let mut list = vec![None; constraint_count(data)];
    let mut put = |order: usize, kind: ConstraintKind, index: usize| {
        if let Some(entry) = list.get_mut(order) {
            *entry = Some((kind, index));
        }
    };
    for (i, c) in data.ik_constraints.iter().enumerate() {
        put(c.order, ConstraintKind::Ik, i);
    }
    for (i, c) in data.path_constraints.iter().enumerate() {
        put(c.order, ConstraintKind::Path, i);
    }
    for (i, c) in data.transform_constraints.iter().enumerate() {
        put(c.order, ConstraintKind::Transform, i);
    }
    for (i, c) in data.physics_constraints.iter().enumerate() {
        put(c.order, ConstraintKind::Physics, i);
    }
    for (i, c) in data.sliders.iter().enumerate() {
        put(c.order, ConstraintKind::Slider, i);
    }
    list
}

/// Resolve a constraint timeline's `index` into Spine's single constraint list
/// to the constraint's index in chine's list for `kind`. An index past the
/// list, or one that names a constraint of another type, is corrupt.
fn constraint_index(
    r: &mut BinaryReader,
    constraints: &[Option<(ConstraintKind, usize)>],
    index: usize,
    kind: ConstraintKind,
) -> usize {
    match constraints.get(index) {
        Some(&Some((found, i))) if found == kind => i,
        _ => {
            corrupt(r);
            index
        }
    }
}

/// Parse one animation, group by group. A timeline's target index is checked
/// against the table it points into. Constraint timelines index Spine's
/// single constraint list (see [`constraint_list`]) and are stored with the
/// index in chine's list for their type. An unknown timeline type records an
/// error and stops the parse early with the timelines read so far, and the
/// load then fails.
pub(super) fn read_animation(
    r: &mut BinaryReader,
    name: String,
    data: &SkeletonData,
    constraints: &[Option<(ConstraintKind, usize)>],
    strings: &[String],
    nonessential: bool,
) -> Animation {
    let _timeline_count = r.var_usize();
    let mut timelines = Vec::new();
    let mut duration = 0.0_f32;

    // Slot timelines: per animated slot, one or more typed timelines (color,
    // two-color, attachment, alpha).
    let slot_groups = r.count();
    for _ in 0..slot_groups {
        let slot = r.var_usize();
        if slot >= data.slots.len() {
            corrupt(r);
        }
        let count = r.count();
        for _ in 0..count {
            let kind = r.byte();
            let frames = frame_count(r);
            let entry: Option<(Timeline, f32)> = match kind {
                0 => {
                    let (tl, d) = read_slot_attachment_timeline(r, slot, frames, strings);
                    Some((Timeline::Attachment(tl), d))
                }
                1 => {
                    let (tl, d) = read_slot_color_timeline(r, slot, frames, 4);
                    Some((Timeline::SlotColor(tl, true), d))
                }
                2 => {
                    let (tl, d) = read_slot_color_timeline(r, slot, frames, 3);
                    Some((Timeline::SlotColor(tl, false), d))
                }
                3 => {
                    let (tl, d) = read_slot_color_timeline(r, slot, frames, 7);
                    Some((Timeline::SlotTwoColor(tl, true), d))
                }
                4 => {
                    let (tl, d) = read_slot_color_timeline(r, slot, frames, 6);
                    Some((Timeline::SlotTwoColor(tl, false), d))
                }
                5 => {
                    let (tl, d) = read_slot_color_timeline(r, slot, frames, 1);
                    Some((Timeline::SlotAlpha(tl), d))
                }
                _ => {
                    r.fail(BinaryError::UnknownTimelineType(kind));
                    return Animation::new(name, duration, timelines);
                }
            };
            if let Some((timeline, d)) = entry {
                duration = duration.max(d);
                timelines.push(timeline);
            }
        }
    }

    // Bone timelines: one group per animated bone, each with one or more typed
    // timelines (rotate / translate / scale / shear, their single axes, and
    // inherit).
    let bone_groups = r.count();
    for _ in 0..bone_groups {
        let bone = r.var_usize();
        if bone >= data.bones.len() {
            corrupt(r);
        }
        let count = r.count();
        for _ in 0..count {
            let kind = r.byte();
            let frames = frame_count(r);
            let (timeline, d) = match kind {
                0 => wrap(read_bone_timeline1(r, bone, frames), Timeline::Rotate),
                1 => wrap(read_bone_timeline2(r, bone, frames), Timeline::Translate),
                2 => axis(read_bone_timeline1(r, bone, frames), BoneAxis::TranslateX),
                3 => axis(read_bone_timeline1(r, bone, frames), BoneAxis::TranslateY),
                4 => wrap(read_bone_timeline2(r, bone, frames), Timeline::Scale),
                5 => axis(read_bone_timeline1(r, bone, frames), BoneAxis::ScaleX),
                6 => axis(read_bone_timeline1(r, bone, frames), BoneAxis::ScaleY),
                7 => wrap(read_bone_timeline2(r, bone, frames), Timeline::Shear),
                8 => axis(read_bone_timeline1(r, bone, frames), BoneAxis::ShearX),
                9 => axis(read_bone_timeline1(r, bone, frames), BoneAxis::ShearY),
                10 => {
                    let (tl, d) = read_inherit_timeline(r, bone, frames);
                    (Timeline::Inherit(tl), d)
                }
                _ => {
                    r.fail(BinaryError::UnknownTimelineType(kind));
                    return Animation::new(name, duration, timelines);
                }
            };
            duration = duration.max(d);
            timelines.push(timeline);
        }
    }

    // Remaining timeline groups: IK, transform, path, physics, slider,
    // attachment / deform, draw order, and events. Constraint timelines index
    // Spine's single constraint list, and each index must name a constraint
    // of the timeline's type.
    // IK constraint timelines (one per animated IK constraint).
    let ik_groups = r.count();
    for _ in 0..ik_groups {
        let raw = r.var_usize();
        let index = constraint_index(r, constraints, raw, ConstraintKind::Ik);
        let frames = frame_count(r);
        let (tl, d) = read_ik_constraint_timeline(r, index, frames);
        duration = duration.max(d);
        timelines.push(Timeline::Ik(tl));
    }

    // Transform constraint timelines (six mix channels).
    let transform_groups = r.count();
    for _ in 0..transform_groups {
        let raw = r.var_usize();
        let index = constraint_index(r, constraints, raw, ConstraintKind::Transform);
        let frames = frame_count(r);
        let (tl, d) = read_curve_timeline_n(r, index, frames, TRANSFORM_MIX.len());
        duration = duration.max(d);
        timelines.push(Timeline::TransformMix(tl));
    }

    // Path constraint timelines: position, spacing, or mix per inner entry.
    let path_groups = r.count();
    for _ in 0..path_groups {
        let raw = r.var_usize();
        let index = constraint_index(r, constraints, raw, ConstraintKind::Path);
        let count = r.count();
        for _ in 0..count {
            let kind = r.byte();
            let frames = frame_count(r);
            let entry = match kind {
                0 => {
                    let (tl, d) = read_curve_timeline_n(r, index, frames, PATH_POSITION.len());
                    Some((Timeline::PathPosition(tl), d))
                }
                1 => {
                    let (tl, d) = read_curve_timeline_n(r, index, frames, PATH_SPACING.len());
                    Some((Timeline::PathSpacing(tl), d))
                }
                2 => {
                    let (tl, d) = read_curve_timeline_n(r, index, frames, PATH_MIX.len());
                    Some((Timeline::PathMix(tl), d))
                }
                _ => {
                    r.fail(BinaryError::UnknownTimelineType(kind));
                    return Animation::new(name, duration, timelines);
                }
            };
            if let Some((timeline, d)) = entry {
                duration = duration.max(d);
                timelines.push(timeline);
            }
        }
    }

    // Physics constraint timelines: one tunable (or a reset) per inner entry.
    // The stored index is one less than the stream's (a 0 marks a global
    // timeline, wrapping to the global sentinel).
    let physics_groups = r.count();
    for _ in 0..physics_groups {
        // A stream index of 0 marks a global timeline (the GLOBAL_PHYSICS
        // sentinel). Any other value is one-based.
        let raw = r.var_usize();
        let index = match raw.checked_sub(1) {
            None => GLOBAL_PHYSICS,
            Some(index) => constraint_index(r, constraints, index, ConstraintKind::Physics),
        };
        let count = r.count();
        for _ in 0..count {
            let kind = r.byte();
            let frames = frame_count(r);
            if kind == 8 {
                let frames = fitting_frames(r, frames, 4);
                let times: Vec<f32> = (0..frames).map(|_| r.float()).collect();
                duration = duration.max(times.last().copied().unwrap_or(0.0));
                timelines.push(Timeline::PhysicsReset(PhysicsResetTimeline::new(
                    index, times,
                )));
                continue;
            }
            let property = match kind {
                0 => PhysicsProperty::Inertia,
                1 => PhysicsProperty::Strength,
                2 => PhysicsProperty::Damping,
                4 => PhysicsProperty::Mass,
                5 => PhysicsProperty::Wind,
                6 => PhysicsProperty::Gravity,
                7 => PhysicsProperty::Mix,
                _ => {
                    r.fail(BinaryError::UnknownTimelineType(kind));
                    return Animation::new(name, duration, timelines);
                }
            };
            let (tl, d) = read_curve_timeline_n(r, index, frames, 1);
            duration = duration.max(d);
            timelines.push(Timeline::Physics(tl, property));
        }
    }

    // Slider timelines (Spine 4.3): SLIDER_TIME (0) and SLIDER_MIX (1), each a
    // one-value curve setting the slider's pose.
    let slider_groups = r.count();
    for _ in 0..slider_groups {
        let raw = r.var_usize();
        let index = constraint_index(r, constraints, raw, ConstraintKind::Slider);
        let count = r.count();
        for _ in 0..count {
            let kind = r.byte();
            let frames = frame_count(r);
            let (tl, d) = read_curve_timeline_n(r, index, frames, 1);
            duration = duration.max(d);
            match kind {
                0 => timelines.push(Timeline::SliderTime(tl)),
                1 => timelines.push(Timeline::SliderMix(tl)),
                _ => {
                    r.fail(BinaryError::UnknownTimelineType(kind));
                    return Animation::new(name, duration, timelines);
                }
            }
        }
    }

    // Attachment timelines, nested skins -> slots -> attachments. Mesh deforms
    // and sequence (animated attachment) timelines are both built and applied.
    // Skin index 0 is the default skin, and `i` is the named skin `i - 1`.
    let deform_skins = r.count();
    for _ in 0..deform_skins {
        let skin_index = r.var_usize();
        if skin_index > data.skins.len() {
            corrupt(r);
        }
        let slots = r.count();
        for _ in 0..slots {
            let slot = r.var_usize();
            if slot >= data.slots.len() {
                corrupt(r);
            }
            let atts = r.count();
            for _ in 0..atts {
                let att_name = string_ref(r, strings).unwrap_or_default();
                let kind = r.byte();
                let frames = frame_count(r);
                match kind {
                    0 => match deform_mesh_info(data, skin_index, slot, &att_name) {
                        Some((frame_len, setup, tl_skin)) => {
                            let (tl, d) = read_deform_timeline(
                                r, slot, att_name, tl_skin, setup, frame_len, frames,
                            );
                            duration = duration.max(d);
                            timelines.push(Timeline::Deform(tl));
                        }
                        // No matching mesh: consume the bytes to stay aligned.
                        None => skip_deform_timeline(r, frames),
                    },
                    1 => {
                        let count = sequence_count(data, skin_index, slot, &att_name).unwrap_or(0);
                        let (tl, d) = read_sequence_timeline(r, slot, att_name, count, frames);
                        duration = duration.max(d);
                        timelines.push(Timeline::Sequence(tl));
                    }
                    _ => {
                        r.fail(BinaryError::UnknownTimelineType(kind));
                        return Animation::new(name, duration, timelines);
                    }
                }
            }
        }
    }

    // Draw order timeline: per keyframe, a slot permutation.
    let draw_order_count = r.count();
    if draw_order_count != 0 {
        let (tl, d) = read_draw_order_timeline(r, draw_order_count, data.slots.len());
        duration = duration.max(d);
        timelines.push(Timeline::DrawOrder(tl));
    }

    // Draw-order folder timelines (new in Spine 4.3): folder-scoped slot
    // reorders. chine has no folder timeline, so these are consumed to keep the
    // stream aligned.
    let folder_count = r.count();
    for _ in 0..folder_count {
        let folder_slot_count = r.count();
        for _ in 0..folder_slot_count {
            r.var_usize();
        }
        let key_count = r.count();
        for _ in 0..key_count {
            r.float();
            let change_count = r.count();
            for _ in 0..change_count {
                r.var_usize();
                r.var_usize();
            }
        }
    }

    // Event timeline: keyframes that fire named events with per-key overrides.
    let event_count = r.count();
    if event_count != 0 {
        let (tl, d) = read_event_timeline(r, data, event_count);
        duration = duration.max(d);
        timelines.push(Timeline::Event(tl));
    }

    // A nonessential export appends the animation's editor color (RGBA8888).
    if nonessential {
        let _color = r.u32();
    }
    Animation::new(name, duration, timelines)
}

/// Read a one-value bone timeline (rotate / single axis). The stream gives the
/// Bezier-segment count first (curve storage), then per-frame time + value with
/// a stepped / linear / Bezier curve between frames.
pub(super) fn read_bone_timeline1(
    r: &mut BinaryReader,
    bone: usize,
    frames: usize,
) -> (BoneTimeline, f32) {
    let bezier_count = r.count();
    let (frames, beziers) = curve_sizes(r, frames, bezier_count, 8, 1);
    let mut tl = BoneTimeline::one_value(bone, frames, beziers);
    let mut bezier = 0;
    let mut duration = 0.0_f32;
    let mut time = r.float();
    let mut value = r.float();
    let last = frames - 1;
    for frame in 0..frames {
        tl.set_frame1(frame, time, value);
        duration = duration.max(time);
        if frame == last {
            break;
        }
        let time2 = r.float();
        let value2 = r.float();
        match r.byte() {
            1 => tl.set_stepped(frame),
            2 => {
                let cx1 = r.float();
                let cy1 = r.float();
                let cx2 = r.float();
                let cy2 = r.float();
                if let Some(b) = next_bezier(r, &mut bezier, beziers) {
                    tl.set_bezier(b, frame, 0, time, value, cx1, cy1, cx2, cy2, time2, value2);
                }
            }
            _ => {}
        }
        time = time2;
        value = value2;
    }
    (tl, duration)
}

/// Read a bone inherit timeline: per frame a time and an inherit-mode byte.
/// Stepped, so there is no curve data. A mode past the five Spine defines is
/// corrupt.
pub(super) fn read_inherit_timeline(
    r: &mut BinaryReader,
    bone: usize,
    frames: usize,
) -> (InheritTimeline, f32) {
    let frames = fitting_frames(r, frames, 5);
    let mut times = Vec::with_capacity(frames);
    let mut modes = Vec::with_capacity(frames);
    let mut duration = 0.0_f32;
    for _ in 0..frames {
        let time = r.float();
        let mode = r.byte();
        if mode > 4 {
            corrupt(r);
        }
        times.push(time);
        modes.push(inherit_from(usize::from(mode)));
        duration = duration.max(time);
    }
    (InheritTimeline::new(bone, times, modes), duration)
}

/// Read a two-value bone timeline (translate / scale / shear). The stream gives
/// the Bezier-segment count first, then per frame a time and two values, with
/// two Bezier segments per curved frame.
fn read_bone_timeline2(r: &mut BinaryReader, bone: usize, frames: usize) -> (BoneTimeline, f32) {
    let bezier_count = r.count();
    let (frames, beziers) = curve_sizes(r, frames, bezier_count, 12, 2);
    let mut tl = BoneTimeline::two_value(bone, frames, beziers);
    let mut bezier = 0;
    let mut duration = 0.0_f32;
    let mut time = r.float();
    let mut v1 = r.float();
    let mut v2 = r.float();
    let last = frames - 1;
    for frame in 0..frames {
        tl.set_frame2(frame, time, v1, v2);
        duration = duration.max(time);
        if frame == last {
            break;
        }
        let time2 = r.float();
        let nv1 = r.float();
        let nv2 = r.float();
        match r.byte() {
            1 => tl.set_stepped(frame),
            2 => {
                let cx1 = r.float();
                let cy1 = r.float();
                let cx2 = r.float();
                let cy2 = r.float();
                if let Some(b) = next_bezier(r, &mut bezier, beziers) {
                    tl.set_bezier(b, frame, 0, time, v1, cx1, cy1, cx2, cy2, time2, nv1);
                }
                let dx1 = r.float();
                let dy1 = r.float();
                let dx2 = r.float();
                let dy2 = r.float();
                if let Some(b) = next_bezier(r, &mut bezier, beziers) {
                    tl.set_bezier(b, frame, 1, time, v2, dx1, dy1, dx2, dy2, time2, nv2);
                }
            }
            _ => {}
        }
        time = time2;
        v1 = nv1;
        v2 = nv2;
    }
    (tl, duration)
}

/// Read `n` color channels, each a `0..=255` byte scaled to `0..=1`.
fn read_color_channels(r: &mut BinaryReader, n: usize) -> Vec<f32> {
    (0..n).map(|_| f32::from(r.byte()) / 255.0).collect()
}

/// Read a slot color timeline (RGBA / RGB / two-color / alpha): per frame a time
/// and `channels` color components (each a byte), with a stepped / linear /
/// Bezier curve per channel between frames. The stream gives the Bezier-segment
/// count first.
pub(super) fn read_slot_color_timeline(
    r: &mut BinaryReader,
    slot: usize,
    frames: usize,
    channels: usize,
) -> (ConstraintTimeline, f32) {
    let bezier_count = r.count();
    let (frames, beziers) = curve_sizes(
        r,
        frames,
        bezier_count,
        channels.saturating_add(4),
        channels,
    );
    let mut tl = ConstraintTimeline::new(slot, frames, beziers, channels + 1);
    let mut bezier = 0;
    let mut duration = 0.0_f32;
    let mut time = r.float();
    let mut vals = read_color_channels(r, channels);
    let last = frames - 1;
    for frame in 0..frames {
        tl.set_frame(frame, time, &vals);
        duration = duration.max(time);
        if frame == last {
            break;
        }
        let time2 = r.float();
        let vals2 = read_color_channels(r, channels);
        match r.byte() {
            1 => tl.set_stepped(frame),
            2 => {
                for (ch, (&v1, &v2)) in vals.iter().zip(&vals2).enumerate() {
                    let cx1 = r.float();
                    let cy1 = r.float();
                    let cx2 = r.float();
                    let cy2 = r.float();
                    if let Some(b) = next_bezier(r, &mut bezier, beziers) {
                        tl.set_bezier(b, frame, ch, time, v1, cx1, cy1, cx2, cy2, time2, v2);
                    }
                }
            }
            _ => {}
        }
        time = time2;
        vals = vals2;
    }
    (tl, duration)
}

/// Read a curve timeline whose `channels` values are plain floats (transform,
/// path, and physics constraint timelines), with a stepped / linear / Bezier
/// curve per channel between frames.
pub(super) fn read_curve_timeline_n(
    r: &mut BinaryReader,
    constraint: usize,
    frames: usize,
    channels: usize,
) -> (ConstraintTimeline, f32) {
    let bezier_count = r.count();
    let frame_bytes = channels.saturating_add(1).saturating_mul(4);
    let (frames, beziers) = curve_sizes(r, frames, bezier_count, frame_bytes, channels);
    let mut tl = ConstraintTimeline::new(constraint, frames, beziers, channels + 1);
    let mut bezier = 0;
    let mut duration = 0.0_f32;
    let mut time = r.float();
    let mut vals: Vec<f32> = (0..channels).map(|_| r.float()).collect();
    let last = frames - 1;
    for frame in 0..frames {
        tl.set_frame(frame, time, &vals);
        duration = duration.max(time);
        if frame == last {
            break;
        }
        let time2 = r.float();
        let vals2: Vec<f32> = (0..channels).map(|_| r.float()).collect();
        match r.byte() {
            1 => tl.set_stepped(frame),
            2 => {
                for (ch, (&v1, &v2)) in vals.iter().zip(&vals2).enumerate() {
                    let cx1 = r.float();
                    let cy1 = r.float();
                    let cx2 = r.float();
                    let cy2 = r.float();
                    if let Some(b) = next_bezier(r, &mut bezier, beziers) {
                        tl.set_bezier(b, frame, ch, time, v1, cx1, cy1, cx2, cy2, time2, v2);
                    }
                }
            }
            _ => {}
        }
        time = time2;
        vals = vals2;
    }
    (tl, duration)
}

/// Read an IK constraint timeline (Spine 4.3 flags-packed). Each frame packs mix
/// presence, softness presence, bend direction, compress, and stretch into one
/// flags byte (also carrying the following segment's curve type). Mix and
/// softness are the two curved channels.
pub(super) fn read_ik_constraint_timeline(
    r: &mut BinaryReader,
    constraint: usize,
    frames: usize,
) -> (ConstraintTimeline, f32) {
    let bezier_count = r.count();
    let (frames, beziers) = curve_sizes(r, frames, bezier_count, 5, 2);
    let mut tl = ConstraintTimeline::new(constraint, frames, beziers, 6);
    let mut bezier = 0;
    let mut duration = 0.0_f32;
    let last = frames - 1;
    let mut flags = r.byte();
    let mut time = r.float();
    let mut mix = ik_flag_value(r, flags, 1, 2);
    let mut softness = if flags & 4 != 0 { r.float() } else { 0.0 };
    for frame in 0..frames {
        let bend = if flags & 8 != 0 { 1.0 } else { -1.0 };
        let compress = f32::from(flags & 16 != 0);
        let stretch = f32::from(flags & 32 != 0);
        tl.set_frame(frame, time, &[mix, softness, bend, compress, stretch]);
        duration = duration.max(time);
        if frame == last {
            break;
        }
        flags = r.byte();
        let time2 = r.float();
        let mix2 = ik_flag_value(r, flags, 1, 2);
        let softness2 = if flags & 4 != 0 { r.float() } else { 0.0 };
        if flags & 64 != 0 {
            tl.set_stepped(frame);
        } else if flags & 128 != 0 {
            let (cx1, cy1, cx2, cy2) = (r.float(), r.float(), r.float(), r.float());
            if let Some(b) = next_bezier(r, &mut bezier, beziers) {
                tl.set_bezier(b, frame, 0, time, mix, cx1, cy1, cx2, cy2, time2, mix2);
            }
            let (dx1, dy1, dx2, dy2) = (r.float(), r.float(), r.float(), r.float());
            if let Some(b) = next_bezier(r, &mut bezier, beziers) {
                tl.set_bezier(
                    b, frame, 1, time, softness, dx1, dy1, dx2, dy2, time2, softness2,
                );
            }
        }
        time = time2;
        mix = mix2;
        softness = softness2;
    }
    (tl, duration)
}

/// Decode an IK timeline mix from its flags: absent (`0`), a default of `1`, or
/// an explicit float.
fn ik_flag_value(r: &mut BinaryReader, flags: u8, present: u8, explicit: u8) -> f32 {
    if flags & present != 0 {
        if flags & explicit != 0 {
            r.float()
        } else {
            1.0
        }
    } else {
        0.0
    }
}

/// Read a slot attachment-swap timeline: per frame a time and the attachment
/// name shown (`None` hides the slot). Stepped, so there is no curve data.
fn read_slot_attachment_timeline(
    r: &mut BinaryReader,
    slot: usize,
    frames: usize,
    strings: &[String],
) -> (AttachmentTimeline, f32) {
    let frames = fitting_frames(r, frames, 5);
    let mut times = Vec::with_capacity(frames);
    let mut names = Vec::with_capacity(frames);
    let mut duration = 0.0_f32;
    for _ in 0..frames {
        let time = r.float();
        names.push(string_ref(r, strings));
        times.push(time);
        duration = duration.max(time);
    }
    (AttachmentTimeline::new(slot, times, names), duration)
}

/// Read a draw-order timeline: per frame a time and a set of slot moves (each a
/// slot index and an offset), resolved into a full slot permutation. A frame
/// with no moves keeps the setup order.
pub(super) fn read_draw_order_timeline(
    r: &mut BinaryReader,
    frames: usize,
    slot_count: usize,
) -> (DrawOrderTimeline, f32) {
    let frames = fitting_frames(r, frames, 5);
    let mut times = Vec::with_capacity(frames);
    let mut orders = Vec::with_capacity(frames);
    let mut duration = 0.0_f32;
    for _ in 0..frames {
        let time = r.float();
        let change_count = r.count();
        let mut offsets: Vec<(usize, i32)> = Vec::with_capacity(change_count);
        for _ in 0..change_count {
            let slot = r.var_usize();
            // The offset is a signed 32-bit value stored as a var_uint, so
            // this cast reinterprets its bits.
            let offset = r.var_uint() as i32;
            offsets.push((slot, offset));
        }
        let order = if offsets.is_empty() {
            (0..slot_count).collect()
        } else {
            resolve_draw_order(r, slot_count, &mut offsets)
        };
        times.push(time);
        orders.push(order);
        duration = duration.max(time);
    }
    (DrawOrderTimeline::new(times, orders), duration)
}

/// Resolve one draw-order key's slot moves into a full slot permutation.
///
/// Spine moves each slot at most once and keeps it inside the slot list, so
/// the key must name distinct slots in range, move each to a position in
/// range, and move no two slots to the same position. Any other key is
/// corrupt: this records the error and keeps the setup order.
fn resolve_draw_order(
    r: &mut BinaryReader,
    slot_count: usize,
    offsets: &mut [(usize, i32)],
) -> Vec<usize> {
    let in_range = |&(slot, offset): &(usize, i32)| {
        i32::try_from(slot_count).is_ok()
            && slot < slot_count
            && isize::try_from(offset)
                .ok()
                .and_then(|offset| slot.checked_add_signed(offset))
                .is_some_and(|position| position < slot_count)
    };
    offsets.sort_by_key(|&(slot, _)| slot);
    let distinct = offsets
        .iter()
        .zip(offsets.iter().skip(1))
        .all(|(a, b)| a.0 < b.0);
    if distinct && offsets.iter().all(in_range) {
        let order = compute_draw_order(slot_count, offsets);
        // Two slots moved to one position leave a gap the permutation
        // cannot fill, marked by `usize::MAX`.
        if !order.contains(&usize::MAX) {
            return order;
        }
    }
    corrupt(r);
    (0..slot_count).collect()
}

/// Read an event timeline: per frame a time and the fired event. Each event's
/// base values come from the skeleton's [`EventData`] (by index) and are
/// overridden by the keyframe. Volume / balance are only stored when the event
/// has an audio path. An index past the events is corrupt.
pub(super) fn read_event_timeline(
    r: &mut BinaryReader,
    data: &SkeletonData,
    frames: usize,
) -> (EventTimeline, f32) {
    let frames = fitting_frames(r, frames, 11);
    let mut times = Vec::with_capacity(frames);
    let mut events = Vec::with_capacity(frames);
    let mut duration = 0.0_f32;
    for _ in 0..frames {
        let time = r.float();
        let index = r.var_usize();
        let (name, def_string, has_audio, def_volume, def_balance) = match data.events.get(index) {
            Some(e) => (
                e.name.clone(),
                e.string_value.clone(),
                e.audio_path.is_some(),
                e.volume,
                e.balance,
            ),
            None => {
                corrupt(r);
                (String::new(), String::new(), false, 1.0, 0.0)
            }
        };
        let int_value = r.var_int();
        let float_value = r.float();
        let string_value = r.string().unwrap_or(def_string);
        let (volume, balance) = if has_audio {
            (r.float(), r.float())
        } else {
            (def_volume, def_balance)
        };
        events.push(Event {
            name,
            time,
            int_value,
            float_value,
            string_value,
            volume,
            balance,
        });
        times.push(time);
        duration = duration.max(time);
    }
    (EventTimeline::new(times, events), duration)
}
