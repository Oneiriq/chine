//! Apply functions for slot timelines: color, alpha, two-color, attachment,
//! draw order, deform, and sequence.
//!
//! A timeline whose slot index is out of range for the skeleton, or whose
//! keyframe arrays are empty or shorter than its times, leaves the skeleton
//! unchanged.

use super::*;

/// Slot color timeline: blends the slot's RGB (and alpha when `has_alpha`) tint
/// from its setup color toward the keyed colors. The index is the slot.
pub(super) fn apply_slot_color(
    t: &ConstraintTimeline,
    has_alpha: bool,
    skel: &mut Skeleton,
    time: f32,
    alpha: f32,
    from: MixFrom,
    add: bool,
) {
    let Some((slot, setup)) = skel.slot_pose_and_setup(t.constraint) else {
        return;
    };
    let Some(first) = t.curve.first_time() else {
        return;
    };
    if time < first {
        match from {
            MixFrom::Setup => {
                slot.color.r = setup.color.r;
                slot.color.g = setup.color.g;
                slot.color.b = setup.color.b;
                if has_alpha {
                    slot.color.a = setup.color.a;
                }
            }
            MixFrom::First => {
                slot.color.r += (setup.color.r - slot.color.r) * alpha;
                slot.color.g += (setup.color.g - slot.color.g) * alpha;
                slot.color.b += (setup.color.b - slot.color.b) * alpha;
                if has_alpha {
                    slot.color.a += (setup.color.a - slot.color.a) * alpha;
                }
            }
            MixFrom::Current => {}
        }
        return;
    }
    slot.color.r = absolute_value_with(
        t.curve.value(time, 1),
        alpha,
        from,
        add,
        slot.color.r,
        setup.color.r,
    );
    slot.color.g = absolute_value_with(
        t.curve.value(time, 2),
        alpha,
        from,
        add,
        slot.color.g,
        setup.color.g,
    );
    slot.color.b = absolute_value_with(
        t.curve.value(time, 3),
        alpha,
        from,
        add,
        slot.color.b,
        setup.color.b,
    );
    if has_alpha {
        slot.color.a = absolute_value_with(
            t.curve.value(time, 4),
            alpha,
            from,
            add,
            slot.color.a,
            setup.color.a,
        );
    }
}

/// Slot alpha timeline: blends only the slot tint's alpha channel from its setup
/// value toward the keyed alpha.
pub(super) fn apply_slot_alpha(
    t: &ConstraintTimeline,
    skel: &mut Skeleton,
    time: f32,
    alpha: f32,
    from: MixFrom,
    add: bool,
) {
    let Some((slot, setup)) = skel.slot_pose_and_setup(t.constraint) else {
        return;
    };
    let Some(first) = t.curve.first_time() else {
        return;
    };
    if time < first {
        match from {
            MixFrom::Setup => slot.color.a = setup.color.a,
            MixFrom::First => slot.color.a += (setup.color.a - slot.color.a) * alpha,
            MixFrom::Current => {}
        }
        return;
    }
    slot.color.a = absolute_value_with(
        t.curve.value(time, 1),
        alpha,
        from,
        add,
        slot.color.a,
        setup.color.a,
    );
}

/// Slot two-color timeline: blends the slot's light (RGB or RGBA) and dark (RGB)
/// tints from their setup colors toward the keyed colors.
pub(super) fn apply_slot_two_color(
    t: &ConstraintTimeline,
    light_alpha: bool,
    skel: &mut Skeleton,
    time: f32,
    alpha: f32,
    from: MixFrom,
    add: bool,
) {
    let Some((slot, setup)) = skel.slot_pose_and_setup(t.constraint) else {
        return;
    };
    let Some(first) = t.curve.first_time() else {
        return;
    };
    let setup_dark = setup.dark_color.unwrap_or(Color::new(0.0, 0.0, 0.0, 1.0));
    if time < first {
        match from {
            MixFrom::Setup => {
                slot.color = setup.color;
                slot.dark_color = setup.dark_color;
            }
            MixFrom::First => {
                slot.color.r += (setup.color.r - slot.color.r) * alpha;
                slot.color.g += (setup.color.g - slot.color.g) * alpha;
                slot.color.b += (setup.color.b - slot.color.b) * alpha;
                if light_alpha {
                    slot.color.a += (setup.color.a - slot.color.a) * alpha;
                }
                let mut dark = slot.dark_color.unwrap_or(setup_dark);
                dark.r += (setup_dark.r - dark.r) * alpha;
                dark.g += (setup_dark.g - dark.g) * alpha;
                dark.b += (setup_dark.b - dark.b) * alpha;
                slot.dark_color = Some(dark);
            }
            MixFrom::Current => {}
        }
        return;
    }
    slot.color.r = absolute_value_with(
        t.curve.value(time, 1),
        alpha,
        from,
        add,
        slot.color.r,
        setup.color.r,
    );
    slot.color.g = absolute_value_with(
        t.curve.value(time, 2),
        alpha,
        from,
        add,
        slot.color.g,
        setup.color.g,
    );
    slot.color.b = absolute_value_with(
        t.curve.value(time, 3),
        alpha,
        from,
        add,
        slot.color.b,
        setup.color.b,
    );
    if light_alpha {
        slot.color.a = absolute_value_with(
            t.curve.value(time, 4),
            alpha,
            from,
            add,
            slot.color.a,
            setup.color.a,
        );
    }
    let base = if light_alpha { 4 } else { 3 };
    let mut dark = slot.dark_color.unwrap_or(setup_dark);
    dark.r = absolute_value_with(
        t.curve.value(time, base + 1),
        alpha,
        from,
        add,
        dark.r,
        setup_dark.r,
    );
    dark.g = absolute_value_with(
        t.curve.value(time, base + 2),
        alpha,
        from,
        add,
        dark.g,
        setup_dark.g,
    );
    dark.b = absolute_value_with(
        t.curve.value(time, base + 3),
        alpha,
        from,
        add,
        dark.b,
        setup_dark.b,
    );
    slot.dark_color = Some(dark);
}

/// Slot attachment timeline: a stepped switch to the keyed attachment name.
pub(super) fn apply_attachment(
    t: &AttachmentTimeline,
    skel: &mut Skeleton,
    time: f32,
    from: MixFrom,
) {
    let Some((slot, setup)) = skel.slot_pose_and_setup(t.slot) else {
        return;
    };
    let before_first = match t.times.first() {
        Some(&first) => time < first,
        None => true,
    };
    if before_first {
        if matches!(from, MixFrom::Setup | MixFrom::First) {
            slot.attachment = setup.attachment.clone();
        }
        return;
    }
    if let Some(name) = t.names.get(search_step(&t.times, time)) {
        slot.attachment = name.clone();
    }
}

/// Draw-order timeline: a stepped switch to the keyed slot ordering.
pub(super) fn apply_draw_order(t: &DrawOrderTimeline, skel: &mut Skeleton, time: f32) {
    let Some(&first) = t.times.first() else {
        return;
    };
    if time < first {
        return; // before the first key: keep the setup order
    }
    if let Some(order) = t.orders.get(search_step(&t.times, time)) {
        skel.set_draw_order(order);
    }
}

/// Draw order folder timeline: a stepped switch of the order of the folder's
/// slots among the draw order positions they hold. Before the first key, a
/// mix from the setup pose or from the first application shows the folder in
/// setup order.
pub(super) fn apply_draw_order_folder(
    t: &DrawOrderFolderTimeline,
    skel: &mut Skeleton,
    time: f32,
    from: MixFrom,
) {
    let Some(&first) = t.times.first() else {
        return;
    };
    // The folder position shown at each of the folder's draw order
    // positions. `None` shows the folder in setup order.
    let order = if time < first {
        if matches!(from, MixFrom::Current) {
            return;
        }
        None
    } else {
        let Some(moves) = t.moves.get(search_step(&t.times, time)) else {
            return;
        };
        (!moves.is_empty()).then(|| draw_order_from_sorted(t.slots.len(), moves))
    };
    let mut found = 0;
    for entry in skel.draw_order_mut() {
        if found == t.slots.len() {
            break;
        }
        if t.sorted.binary_search(entry).is_err() {
            continue;
        }
        let position = order
            .as_ref()
            .map_or(Some(found), |order| order.get(found).copied());
        if let Some(&slot) = position.and_then(|position| t.slots.get(position)) {
            *entry = slot;
        }
        found += 1;
    }
}

/// The attachment slot `slot` currently shows, when the timelines of the
/// attachment at `key` drive it. The slot's attachment resolves through the
/// active skin, then the default skin, as rendering resolves it. It is driven
/// when it is the attachment at `key`, or a linked mesh that inherits that
/// attachment's timelines.
fn driven_attachment<'s>(
    skel: &'s Skeleton,
    slot: usize,
    key: &AttachmentKey,
) -> Option<&'s Attachment> {
    let name = skel.slot(slot)?.attachment.as_deref()?;
    let active = skel.active_skin();
    let (skin, attachment) = match active.and_then(|skin| skin.attachment(slot, name)) {
        Some(attachment) => (active.map(|skin| skin.name.as_str()), attachment),
        None => (None, skel.data().default_skin.attachment(slot, name)?),
    };
    let driven = match attachment {
        Attachment::Mesh(MeshAttachment {
            timeline_source: Some(source),
            ..
        }) => source == key,
        _ => key.slot == slot && key.name == name && key.skin.as_deref() == skin,
    };
    driven.then_some(attachment)
}

/// Deform timeline: set the deform buffer of each slot the timeline
/// reaches to the setup vertices plus the interpolated keyframe offsets
/// (scaled by `alpha`). A slot is changed only while it shows the timeline's
/// attachment, or a linked mesh inheriting its timelines. The buffer takes
/// the length of the setup vertices. A keyframe with fewer offsets adds zero
/// to the rest.
pub(super) fn apply_deform(
    t: &DeformTimeline,
    skel: &mut Skeleton,
    time: f32,
    alpha: f32,
    from: MixFrom,
) {
    let before_first = match t.times.first() {
        Some(&first) => time < first,
        None => true,
    };
    if before_first && !matches!(from, MixFrom::Setup) {
        return;
    }
    // Interpolated once, for the first slot that shows the attachment.
    let mut offset: Option<Vec<f32>> = None;
    for index in t.target.slots() {
        if driven_attachment(skel, index, &t.target.key).is_none() {
            continue;
        }
        let Some((slot, _)) = skel.slot_pose_and_setup(index) else {
            continue;
        };
        if before_first {
            slot.deform.clear();
            continue;
        }
        let offset = offset.get_or_insert_with(|| interp_deform(t, time));
        slot.deform.resize(t.setup.len(), 0.0);
        for (i, (d, &s)) in slot.deform.iter_mut().zip(&t.setup).enumerate() {
            *d = s + offset.get(i).copied().unwrap_or(0.0) * alpha;
        }
    }
}

/// The number of regions in the sequence `attachment` shows, or 0 when it
/// has no sequence.
fn sequence_count(attachment: &Attachment) -> usize {
    let sequence = match attachment {
        Attachment::Region(region) => region.sequence.as_ref(),
        Attachment::Mesh(mesh) => mesh.sequence.as_ref(),
        _ => None,
    };
    sequence.map_or(0, |sequence| sequence.count)
}

/// Sequence timeline: advance the sequence frame index of each slot the
/// timeline reaches from the active keyframe by the elapsed time over the
/// delay, wrapped per the sequence mode over the regions of the sequence the
/// slot shows. A slot is changed only while it shows the timeline's
/// attachment, or a linked mesh inheriting its timelines.
pub(super) fn apply_sequence(t: &SequenceTimeline, skel: &mut Skeleton, time: f32) {
    let Some(&first) = t.times.first() else {
        return;
    };
    if time < first {
        return;
    }
    let frame = search_step(&t.times, time);
    let (Some(&frame_time), Some(&mode_and_index), Some(&delay)) = (
        t.times.get(frame),
        t.mode_and_index.get(frame),
        t.delays.get(frame),
    ) else {
        return;
    };
    for index in t.target.slots() {
        let Some(attachment) = driven_attachment(skel, index, &t.target.key) else {
            continue;
        };
        let count = sequence_count(attachment);
        if count == 0 {
            continue;
        }
        let shown = sequence_index(mode_and_index, count, time - frame_time, delay);
        if let Some((slot, _)) = skel.slot_pose_and_setup(index) {
            // An index past `i32::MAX` only comes from a corrupt mode or delay.
            slot.sequence_index = i32::try_from(shown).unwrap_or(i32::MAX);
        }
    }
}

/// The region index a sequence key shows `elapsed` seconds after its time,
/// over a sequence of `count` regions (`count` is not 0).
fn sequence_index(mode_and_index: u32, count: usize, elapsed: f32, delay: f32) -> usize {
    let mut index = (mode_and_index >> 4) as usize;
    let mode = mode_and_index & 0xf;
    if mode != 0 {
        // The cast saturates: a zero delay or an infinite time gives
        // `usize::MAX`, and NaN gives 0. The add saturates to match.
        let advance = (elapsed / delay + 0.000_01) as usize;
        index = index.saturating_add(advance);
        let last = count - 1;
        index = match mode {
            1 => index.min(last),                                      // once
            2 => index % count,                                        // loop
            3 => sequence_pingpong(index, count),                      // pingpong
            4 => last.saturating_sub(index),                           // once reverse
            5 => last - (index % count),                               // loop reverse
            6 => sequence_pingpong(index.saturating_add(last), count), // pingpong reverse
            _ => index,
        };
    }
    index
}

/// Wrap an index across a ping-pong sequence of `count` regions.
fn sequence_pingpong(index: usize, count: usize) -> usize {
    let n = count.saturating_mul(2).saturating_sub(2);
    let i = if n == 0 { 0 } else { index % n };
    if i >= count {
        n - i
    } else {
        i
    }
}

/// Interpolate the deform offset frames at `time`, using the timeline's curve
/// (stepped / linear / Bezier) for the interpolation percent. A missing frame
/// reads as no offsets.
fn interp_deform(t: &DeformTimeline, time: f32) -> Vec<f32> {
    let times = &t.times;
    let last = times.len().saturating_sub(1);
    if times.get(last).is_some_and(|&last_time| time >= last_time) {
        return t.frames.get(last).cloned().unwrap_or_default();
    }
    let i = search_step(times, time);
    let Some(a) = t.frames.get(i) else {
        return Vec::new();
    };
    // A corrupt timeline can have no next frame here. Hold this one.
    let Some(b) = t.frames.get(i + 1) else {
        return a.clone();
    };
    let pct = t.percent(time);
    let n = a.len().max(b.len());
    let mut out = vec![0.0; n];
    for (j, v) in out.iter_mut().enumerate() {
        let av = a.get(j).copied().unwrap_or(0.0);
        let bv = b.get(j).copied().unwrap_or(0.0);
        *v = av + (bv - av) * pct;
    }
    out
}

/// Index of the last keyframe at or before `time` (assumes `time >= times[0]`).
fn search_step(times: &[f32], time: f32) -> usize {
    times.iter().rposition(|&t| t <= time).unwrap_or(0)
}
