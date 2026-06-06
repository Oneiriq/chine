use super::*;

use crate::anim::GLOBAL_PHYSICS;

/// Wrap a parsed bone timeline in its [`Timeline`] variant.
fn wrap((tl, d): (BoneTimeline, f32), make: fn(BoneTimeline) -> Timeline) -> (Timeline, f32) {
    (make(tl), d)
}

/// Wrap a parsed single-axis bone timeline.
fn axis((tl, d): (BoneTimeline, f32), a: BoneAxis) -> (Timeline, f32) {
    (Timeline::BoneAxis(tl, a), d)
}

/// Parse one animation. Built group by group; on an unhandled timeline type the
/// parse stops early and returns what it has (later sections are left unread).
pub(super) fn read_animation(
    r: &mut BinaryReader,
    name: String,
    data: &SkeletonData,
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
        let count = r.count();
        for _ in 0..count {
            let kind = r.byte();
            let frames = r.count();
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
    // timelines (rotate / translate / scale / shear and their single axes).
    let bone_groups = r.count();
    for _ in 0..bone_groups {
        let bone = r.var_usize();
        let count = r.count();
        for _ in 0..count {
            let kind = r.byte();
            let frames = r.count();
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
    // attachment / deform, draw order, and events.
    // IK constraint timelines (one per animated IK constraint).
    let ik_groups = r.count();
    for _ in 0..ik_groups {
        let index = r.var_usize();
        let frames = r.count();
        let (tl, d) = read_ik_constraint_timeline(r, index, frames);
        duration = duration.max(d);
        timelines.push(Timeline::Ik(tl));
    }

    // Transform constraint timelines (six mix channels).
    let transform_groups = r.count();
    for _ in 0..transform_groups {
        let index = r.var_usize();
        let frames = r.count();
        let (tl, d) = read_curve_timeline_n(r, index, frames, TRANSFORM_MIX.len());
        duration = duration.max(d);
        timelines.push(Timeline::TransformMix(tl));
    }

    // Path constraint timelines: position, spacing, or mix per inner entry.
    let path_groups = r.count();
    for _ in 0..path_groups {
        let index = r.var_usize();
        let count = r.count();
        for _ in 0..count {
            let kind = r.byte();
            let frames = r.count();
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
        // sentinel); any other value is one-based.
        let raw = r.var_usize();
        let index = if raw == 0 { GLOBAL_PHYSICS } else { raw - 1 };
        let count = r.count();
        for _ in 0..count {
            let kind = r.byte();
            let frames = r.count();
            if kind == 8 {
                let times: Vec<f32> = (0..frames).map(|_| r.float()).collect();
                duration = duration.max(times.last().copied().unwrap_or(0.0));
                timelines.push(Timeline::PhysicsReset(PhysicsResetTimeline::new(index, times)));
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
        let index = r.var_usize();
        let count = r.count();
        for _ in 0..count {
            let kind = r.byte();
            let frames = r.count();
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
    let deform_skins = r.count();
    for _ in 0..deform_skins {
        let skin_index = r.var_usize();
        let slots = r.count();
        for _ in 0..slots {
            let slot = r.var_usize();
            let atts = r.count();
            for _ in 0..atts {
                let att_name = string_ref(r, strings).unwrap_or_default();
                let kind = r.byte();
                let frames = r.count();
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
                        let count =
                            sequence_count(data, skin_index, slot, &att_name).unwrap_or(0);
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
pub(super) fn read_bone_timeline1(r: &mut BinaryReader, bone: usize, frames: usize) -> (BoneTimeline, f32) {
    let bezier_count = r.count();
    let mut tl = BoneTimeline::one_value(bone, frames, bezier_count);
    let mut bezier = 0;
    let mut duration = 0.0_f32;
    let mut time = r.float();
    let mut value = r.float();
    let last = frames.saturating_sub(1);
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
                tl.set_bezier(
                    bezier, frame, 0, time, value, cx1, cy1, cx2, cy2, time2, value2,
                );
                bezier += 1;
            }
            _ => {}
        }
        time = time2;
        value = value2;
    }
    (tl, duration)
}

/// Read a two-value bone timeline (translate / scale / shear). The stream gives
/// the Bezier-segment count first, then per frame a time and two values, with
/// two Bezier segments per curved frame.
fn read_bone_timeline2(r: &mut BinaryReader, bone: usize, frames: usize) -> (BoneTimeline, f32) {
    let bezier_count = r.count();
    let mut tl = BoneTimeline::two_value(bone, frames, bezier_count);
    let mut bezier = 0;
    let mut duration = 0.0_f32;
    let mut time = r.float();
    let mut v1 = r.float();
    let mut v2 = r.float();
    let last = frames.saturating_sub(1);
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
                tl.set_bezier(bezier, frame, 0, time, v1, cx1, cy1, cx2, cy2, time2, nv1);
                bezier += 1;
                let dx1 = r.float();
                let dy1 = r.float();
                let dx2 = r.float();
                let dy2 = r.float();
                tl.set_bezier(bezier, frame, 1, time, v2, dx1, dy1, dx2, dy2, time2, nv2);
                bezier += 1;
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
    let mut tl = ConstraintTimeline::new(slot, frames, bezier_count, channels + 1);
    let mut bezier = 0;
    let mut duration = 0.0_f32;
    let mut time = r.float();
    let mut vals = read_color_channels(r, channels);
    let last = frames.saturating_sub(1);
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
                for ch in 0..channels {
                    let cx1 = r.float();
                    let cy1 = r.float();
                    let cx2 = r.float();
                    let cy2 = r.float();
                    tl.set_bezier(
                        bezier, frame, ch, time, vals[ch], cx1, cy1, cx2, cy2, time2, vals2[ch],
                    );
                    bezier += 1;
                }
            }
            _ => {}
        }
        time = time2;
        vals = vals2;
    }
    (tl, duration)
}

/// Consume one mesh-deform timeline frame: a run of changed vertices encoded as
/// a count, a start offset, then that many float offsets (count `0` means the
/// frame uses the setup vertices).
fn skip_deform_frame(r: &mut BinaryReader) {
    let end = r.count();
    if end != 0 {
        let _start = r.var_usize();
        for _ in 0..end {
            r.float();
        }
    }
}

/// Consume a mesh-deform timeline (Bezier count, then per-frame time and vertex
/// offsets, with stepped / linear / Bezier curves) without building anything.
fn skip_deform_timeline(r: &mut BinaryReader, frames: usize) {
    let _bezier_count = r.var_usize();
    let last = frames.saturating_sub(1);
    r.float(); // first frame time
    skip_deform_frame(r);
    for frame in 0..frames {
        if frame == last {
            break;
        }
        r.float(); // time
        if r.byte() == 2 {
            // Bezier: four control floats.
            for _ in 0..4 {
                r.float();
            }
        }
        skip_deform_frame(r);
    }
}

/// The number of regions in a sequenced region/mesh attachment, for a sequence
/// timeline's index wrapping.
fn sequence_count(data: &SkeletonData, skin_index: usize, slot: usize, name: &str) -> Option<usize> {
    let skin = if skin_index == 0 {
        None
    } else {
        data.skins.get(skin_index - 1)
    };
    match data.attachment(slot, name, skin) {
        Some(Attachment::Region(r)) => r.sequence.as_ref().map(|s| s.count),
        Some(Attachment::Mesh(m)) => m.sequence.as_ref().map(|s| s.count),
        _ => None,
    }
}

/// Read a sequence (flipbook) timeline: per frame a time, a packed mode-and-index
/// int, and a delay. Stepped, so there is no curve data.
fn read_sequence_timeline(
    r: &mut BinaryReader,
    slot: usize,
    attachment: String,
    count: usize,
    frames: usize,
) -> (SequenceTimeline, f32) {
    let mut times = Vec::with_capacity(frames);
    let mut mode_and_index = Vec::with_capacity(frames);
    let mut delays = Vec::with_capacity(frames);
    let mut duration = 0.0_f32;
    for _ in 0..frames {
        let time = r.float();
        mode_and_index.push(r.u32());
        delays.push(r.float());
        times.push(time);
        duration = duration.max(time);
    }
    (
        SequenceTimeline::new(slot, attachment, count, times, mode_and_index, delays),
        duration,
    )
}

/// Resolve a deform timeline's mesh: the setup-pose deform length, the setup
/// vertices to add at apply time (zeros for a weighted mesh), and the
/// timeline's skin name (`None` for the default skin). Returns `None` when no
/// matching mesh is found, so the caller consumes the bytes instead.
fn deform_mesh_info(
    data: &SkeletonData,
    skin_index: usize,
    slot: usize,
    name: &str,
) -> Option<(usize, Vec<f32>, Option<String>)> {
    let skin = if skin_index == 0 {
        None
    } else {
        data.skins.get(skin_index - 1)
    };
    let Some(Attachment::Mesh(mesh)) = data.attachment(slot, name, skin) else {
        return None;
    };
    let frame_len = mesh.deform_len();
    let setup = mesh
        .setup_vertices()
        .map_or_else(|| vec![0.0; frame_len], <[f32]>::to_vec);
    Some((frame_len, setup, skin.map(|s| s.name.clone())))
}

/// Read a mesh-deform timeline into chine's relative-offset model: per frame a
/// time and a sparse run of vertex offsets (zeros elsewhere; the setup vertices
/// are added at apply time), with stepped / linear / Bezier curves. The offsets
/// are read raw, never adding the setup, which matches the JSON loader.
fn read_deform_timeline(
    r: &mut BinaryReader,
    slot: usize,
    attachment: String,
    skin: Option<String>,
    setup: Vec<f32>,
    frame_len: usize,
    frames: usize,
) -> (DeformTimeline, f32) {
    let bezier_count = r.count();
    let last = frames.saturating_sub(1);
    let mut times = Vec::with_capacity(frames);
    let mut offsets = Vec::with_capacity(frames);
    let mut segments: Vec<(u8, [f32; 4])> = Vec::new();
    let mut time = r.float();
    for frame in 0..frames {
        let mut deform = vec![0.0_f32; frame_len];
        let end = r.count();
        if end != 0 {
            let start = r.var_usize();
            for i in 0..end {
                let value = r.float();
                if let Some(v) = deform.get_mut(start + i) {
                    *v = value;
                }
            }
        }
        times.push(time);
        offsets.push(deform);
        if frame == last {
            break;
        }
        let time2 = r.float();
        let segment = match r.byte() {
            1 => (1, [0.0; 4]),
            2 => (2, [r.float(), r.float(), r.float(), r.float()]),
            _ => (0, [0.0; 4]),
        };
        segments.push(segment);
        time = time2;
    }
    let duration = times.last().copied().unwrap_or(0.0);
    let mut tl = DeformTimeline::new(
        slot,
        attachment,
        skin,
        setup,
        times.clone(),
        offsets,
        bezier_count,
    );
    let mut bezier = 0;
    for (frame, &(kind, c)) in segments.iter().enumerate() {
        match kind {
            1 => tl.set_stepped(frame),
            2 => {
                tl.set_bezier(
                    bezier,
                    frame,
                    0,
                    times[frame],
                    0.0,
                    c[0],
                    c[1],
                    c[2],
                    c[3],
                    times[frame + 1],
                    1.0,
                );
                bezier += 1;
            }
            _ => {}
        }
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
    let mut tl = ConstraintTimeline::new(constraint, frames, bezier_count, channels + 1);
    let mut bezier = 0;
    let mut duration = 0.0_f32;
    let mut time = r.float();
    let mut vals: Vec<f32> = (0..channels).map(|_| r.float()).collect();
    let last = frames.saturating_sub(1);
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
                for ch in 0..channels {
                    let cx1 = r.float();
                    let cy1 = r.float();
                    let cx2 = r.float();
                    let cy2 = r.float();
                    tl.set_bezier(
                        bezier, frame, ch, time, vals[ch], cx1, cy1, cx2, cy2, time2, vals2[ch],
                    );
                    bezier += 1;
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
/// flags byte (also carrying the following segment's curve type); mix and
/// softness are the two curved channels.
pub(super) fn read_ik_constraint_timeline(
    r: &mut BinaryReader,
    constraint: usize,
    frames: usize,
) -> (ConstraintTimeline, f32) {
    let bezier_count = r.count();
    let mut tl = ConstraintTimeline::new(constraint, frames, bezier_count, 6);
    let mut bezier = 0;
    let mut duration = 0.0_f32;
    let last = frames.saturating_sub(1);
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
            tl.set_bezier(bezier, frame, 0, time, mix, cx1, cy1, cx2, cy2, time2, mix2);
            bezier += 1;
            let (dx1, dy1, dx2, dy2) = (r.float(), r.float(), r.float(), r.float());
            tl.set_bezier(
                bezier, frame, 1, time, softness, dx1, dy1, dx2, dy2, time2, softness2,
            );
            bezier += 1;
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
    let mut times = Vec::with_capacity(frames);
    let mut orders = Vec::with_capacity(frames);
    let mut duration = 0.0_f32;
    for _ in 0..frames {
        let time = r.float();
        let change_count = r.count();
        let mut offsets: Vec<(usize, i32)> = Vec::with_capacity(change_count);
        for _ in 0..change_count {
            let slot = r.var_usize();
            let offset = r.var_usize() as i32;
            offsets.push((slot, offset));
        }
        let order = if offsets.is_empty() {
            (0..slot_count).collect()
        } else {
            compute_draw_order(slot_count, &mut offsets)
        };
        times.push(time);
        orders.push(order);
        duration = duration.max(time);
    }
    (DrawOrderTimeline::new(times, orders), duration)
}

/// Read an event timeline: per frame a time and the fired event. Each event's
/// base values come from the skeleton's [`EventData`] (by index) and are
/// overridden by the keyframe; volume / balance are only stored when the event
/// has an audio path.
pub(super) fn read_event_timeline(
    r: &mut BinaryReader,
    data: &SkeletonData,
    frames: usize,
) -> (EventTimeline, f32) {
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
            None => (String::new(), String::new(), false, 1.0, 0.0),
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
