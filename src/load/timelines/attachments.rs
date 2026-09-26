//! Attachment timelines for the JSON loader: mesh deform and sequence
//! (flipbook) keys, each keyed by skin, slot, and attachment.

use super::*;

/// One attachment's entry in a skin, slot, and attachment map.
pub(super) struct AttachmentEntry<'a> {
    /// The skin name the map keys the entry under.
    pub(super) skin: &'a str,
    /// The slot index.
    pub(super) slot: usize,
    /// The attachment name.
    pub(super) name: &'a str,
    /// The entry's value: a deform key list in the Spine 4.0 `deform` map,
    /// or a map of timeline kinds under `attachments`.
    pub(super) value: &'a Value,
}

/// Flatten a skin, slot, and attachment map into its entries. An entry for a
/// slot the skeleton does not have is skipped.
pub(super) fn attachment_entries<'a>(map: &'a Value, names: &Names) -> Vec<AttachmentEntry<'a>> {
    let mut entries = Vec::new();
    let Some(skins) = map.as_object() else {
        return entries;
    };
    for (skin, slots) in skins {
        let Some(slots) = slots.as_object() else {
            continue;
        };
        for (slot_name, attachments) in slots {
            let Some(&slot) = names.slots.get(slot_name.as_str()) else {
                continue;
            };
            let Some(attachments) = attachments.as_object() else {
                continue;
            };
            for (name, value) in attachments {
                entries.push(AttachmentEntry {
                    skin,
                    slot,
                    name,
                    value,
                });
            }
        }
    }
    entries
}

/// The attachment an attachment timeline drives: the entry's attachment in
/// the skin the entry names. Spine looks only in that skin. `None` when the
/// skin or the attachment is missing. A mesh's timeline slots join the
/// target.
fn entry_target<'d>(
    entry: &AttachmentEntry,
    data: &'d SkeletonData,
    names: &Names,
) -> Option<(AttachmentTarget, &'d Attachment)> {
    let (skin_name, skin) = if entry.skin == "default" {
        (None, &data.default_skin)
    } else {
        (Some(entry.skin.to_string()), names.skin(data, entry.skin)?)
    };
    let attachment = skin.attachment(entry.slot, entry.name)?;
    let timeline_slots = match attachment {
        Attachment::Mesh(mesh) => Arc::clone(&mesh.timeline_slots),
        _ => Vec::new().into(),
    };
    let key = AttachmentKey {
        skin: skin_name,
        slot: entry.slot,
        name: entry.name.to_string(),
    };
    Some((AttachmentTarget::new(key, timeline_slots), attachment))
}

/// Read one mesh's deform timeline from its key list. `None` when there are no
/// keys or the entry names no mesh.
pub(super) fn read_deform(
    entry: &AttachmentEntry,
    keys: &Value,
    data: &SkeletonData,
    names: &Names,
    budget: &mut Budget,
) -> Result<Option<(Timeline, f32)>, LoadError> {
    let Some(keys) = keys.as_array().filter(|keys| !keys.is_empty()) else {
        return Ok(None);
    };
    let Some((target, Attachment::Mesh(mesh))) = entry_target(entry, data, names) else {
        return Ok(None);
    };
    let frame_len = mesh.deform_len();
    let n = keys.len();
    // The setup vertices plus one full frame per key.
    budget.charge(
        n.saturating_add(1)
            .saturating_mul(frame_len)
            .saturating_mul(size_of::<f32>()),
        "a deform timeline",
    )?;
    // Unweighted: setup = the bind vertices (offsets add to them). Weighted:
    // setup = zeros (offsets are added per-influence to the bind positions in
    // compute_vertices).
    let setup = match mesh.setup_vertices() {
        Some(v) => v.to_vec(),
        None => vec![0.0; frame_len],
    };
    let mut times = Vec::with_capacity(n);
    let mut frames = Vec::with_capacity(n);
    for k in keys {
        times.push(f(k, "time"));
        frames.push(read_deform_frame(k, frame_len));
    }
    let duration = times.last().copied().unwrap_or(0.0);
    let mut tl = DeformTimeline::new(target, setup, times, frames, n);
    let mut bezier = 0;
    for (frame, pair) in keys.windows(2).enumerate() {
        if let [key, next] = pair {
            if let Some(curve) = key.get("curve") {
                let (time, time2) = (f(key, "time"), f(next, "time"));
                bezier = read_curve(curve, &mut tl, bezier, frame, 0, time, time2, 0.0, 1.0);
            }
        }
    }
    Ok(Some((Timeline::Deform(tl), duration)))
}

/// Read one deform keyframe's sparse `offset`/`vertices` into a full `len`-long
/// offset array (zero where unspecified). Values past the end are dropped.
fn read_deform_frame(k: &Value, len: usize) -> Vec<f32> {
    let mut frame = vec![0.0; len];
    let offset = k.get("offset").and_then(Value::as_u64).unwrap_or(0);
    let window = usize::try_from(offset)
        .ok()
        .and_then(|offset| frame.get_mut(offset..));
    if let Some(window) = window {
        for (slot, v) in window.iter_mut().zip(f_array(k, "vertices")) {
            *slot = v;
        }
    }
    frame
}

/// Read one attachment's sequence (flipbook) timeline: per key a packed
/// mode and index, plus a hold delay. `None` when there are no keys or the
/// entry names no attachment.
pub(super) fn read_sequence(
    entry: &AttachmentEntry,
    keys: &Value,
    data: &SkeletonData,
    names: &Names,
) -> Option<(Timeline, f32)> {
    let keys = keys.as_array().filter(|keys| !keys.is_empty())?;
    let (target, _) = entry_target(entry, data, names)?;
    let n = keys.len();
    let mut times = Vec::with_capacity(n);
    let mut mode_and_index = Vec::with_capacity(n);
    let mut delays = Vec::with_capacity(n);
    for k in keys {
        // The index shares a `u32` with the 4-bit mode, so a larger index
        // saturates instead of losing its high bits.
        let index = k.get("index").and_then(Value::as_u64).map_or(0, |i| {
            u32::try_from(i).unwrap_or(u32::MAX).min(u32::MAX >> 4)
        });
        let mode = sequence_mode(k.get("mode").and_then(Value::as_str));
        times.push(f(k, "time"));
        mode_and_index.push((index << 4) | mode);
        delays.push(f(k, "delay"));
    }
    let duration = times.last().copied().unwrap_or(0.0);
    let tl = SequenceTimeline::new(target, times, mode_and_index, delays);
    Some((Timeline::Sequence(tl), duration))
}

/// Map a Spine sequence mode name to its ordinal (`hold` = 0 by default).
fn sequence_mode(name: Option<&str>) -> u32 {
    match name {
        Some("once") => 1,
        Some("loop") => 2,
        Some("pingpong") => 3,
        Some("onceReverse") => 4,
        Some("loopReverse") => 5,
        Some("pingpongReverse") => 6,
        _ => 0, // hold
    }
}
