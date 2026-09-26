use super::*;

use crate::anim::{
    compute_draw_order, Animation, AttachmentTimeline, BoneAxis, BoneTimeline, ConstraintTimeline,
    DeformTimeline, DrawOrderTimeline, EventTimeline, Fallback, InheritTimeline, PhysicsProperty,
    PhysicsResetTimeline, SequenceTimeline, Timeline, GLOBAL_PHYSICS, PATH_MIX, PATH_POSITION,
    PATH_SPACING, TRANSFORM_MIX,
};
use crate::event::Event;

mod curves;
use curves::{
    read_curve, read_curve_timeline, read_ik_timeline, read_slot_rgb_timeline,
    read_slot_rgba_timeline, read_slot_two_color_timeline, read_timeline1, read_timeline2,
    value_channel,
};

/// Parse one animation: bone, slot, deform, event, draw-order, and constraint
/// timelines. `names` resolves the names timelines refer to, and `budget`
/// caps the data that deform, draw order, and event keys expand to.
pub(super) fn parse_animation(
    name: &str,
    anim: &Value,
    data: &SkeletonData,
    names: &Names,
    budget: &mut Budget,
) -> Result<Animation, LoadError> {
    let mut timelines = Vec::new();
    let mut duration = 0.0_f32;
    if let Some(bones) = anim.get("bones").and_then(Value::as_object) {
        for (bone_name, props) in bones {
            let bone = names.bone(bone_name)?;
            let Some(props) = props.as_object() else {
                continue;
            };
            for (prop, keys) in props {
                let Some(keys) = keys.as_array() else {
                    continue;
                };
                if keys.is_empty() {
                    continue;
                }
                let (timeline, dur) = match prop.as_str() {
                    "rotate" => {
                        let (tl, d) = read_timeline1(keys, bone, 0.0);
                        (Timeline::Rotate(tl), d)
                    }
                    "translate" => {
                        let (tl, d) = read_timeline2(keys, bone, "x", "y", 0.0);
                        (Timeline::Translate(tl), d)
                    }
                    "scale" => {
                        let (tl, d) = read_timeline2(keys, bone, "x", "y", 1.0);
                        (Timeline::Scale(tl), d)
                    }
                    "shear" => {
                        let (tl, d) = read_timeline2(keys, bone, "x", "y", 0.0);
                        (Timeline::Shear(tl), d)
                    }
                    "translatex" => {
                        let (tl, d) = read_timeline1(keys, bone, 0.0);
                        (Timeline::BoneAxis(tl, BoneAxis::TranslateX), d)
                    }
                    "translatey" => {
                        let (tl, d) = read_timeline1(keys, bone, 0.0);
                        (Timeline::BoneAxis(tl, BoneAxis::TranslateY), d)
                    }
                    "scalex" => {
                        let (tl, d) = read_timeline1(keys, bone, 1.0);
                        (Timeline::BoneAxis(tl, BoneAxis::ScaleX), d)
                    }
                    "scaley" => {
                        let (tl, d) = read_timeline1(keys, bone, 1.0);
                        (Timeline::BoneAxis(tl, BoneAxis::ScaleY), d)
                    }
                    "shearx" => {
                        let (tl, d) = read_timeline1(keys, bone, 0.0);
                        (Timeline::BoneAxis(tl, BoneAxis::ShearX), d)
                    }
                    "sheary" => {
                        let (tl, d) = read_timeline1(keys, bone, 0.0);
                        (Timeline::BoneAxis(tl, BoneAxis::ShearY), d)
                    }
                    "inherit" => {
                        let times: Vec<f32> = keys.iter().map(|k| f(k, "time")).collect();
                        let modes = keys
                            .iter()
                            .map(|k| parse_inherit(k.get("inherit")))
                            .collect();
                        let d = times.iter().copied().fold(0.0, f32::max);
                        (
                            Timeline::Inherit(InheritTimeline::new(bone, times, modes)),
                            d,
                        )
                    }
                    _ => continue, // unknown bone channels are skipped
                };
                duration = duration.max(dur);
                timelines.push(timeline);
            }
        }
    }
    if let Some(iks) = anim.get("ik").and_then(Value::as_object) {
        for (cname, keys) in iks {
            let Some(keys) = keys.as_array() else {
                continue;
            };
            if keys.is_empty() {
                continue;
            }
            let idx = lookup(&names.ik, cname)?;
            let (tl, dur) = read_ik_timeline(keys, idx);
            duration = duration.max(dur);
            timelines.push(Timeline::Ik(tl));
        }
    }
    if let Some(tcs) = anim.get("transform").and_then(Value::as_object) {
        for (cname, keys) in tcs {
            let Some(keys) = keys.as_array() else {
                continue;
            };
            if keys.is_empty() {
                continue;
            }
            let idx = lookup(&names.transform, cname)?;
            let (tl, dur) = read_curve_timeline(keys, idx, TRANSFORM_MIX);
            duration = duration.max(dur);
            timelines.push(Timeline::TransformMix(tl));
        }
    }
    if let Some(pcs) = anim.get("path").and_then(Value::as_object) {
        for (cname, channels) in pcs {
            let idx = lookup(&names.path, cname)?;
            let Some(channels) = channels.as_object() else {
                continue;
            };
            for (channel, keys) in channels {
                let Some(keys) = keys.as_array() else {
                    continue;
                };
                if keys.is_empty() {
                    continue;
                }
                match channel.as_str() {
                    "position" => {
                        let (tl, dur) = read_curve_timeline(keys, idx, PATH_POSITION);
                        duration = duration.max(dur);
                        timelines.push(Timeline::PathPosition(tl));
                    }
                    "spacing" => {
                        let (tl, dur) = read_curve_timeline(keys, idx, PATH_SPACING);
                        duration = duration.max(dur);
                        timelines.push(Timeline::PathSpacing(tl));
                    }
                    "mix" => {
                        let (tl, dur) = read_curve_timeline(keys, idx, PATH_MIX);
                        duration = duration.max(dur);
                        timelines.push(Timeline::PathMix(tl));
                    }
                    _ => {}
                }
            }
        }
    }

    if let Some(pcs) = anim.get("physics").and_then(Value::as_object) {
        for (cname, channels) in pcs {
            // An empty constraint name marks a global timeline (it drives every
            // physics constraint whose matching global flag is set). The "reset"
            // channel is still skipped below.
            let idx = if cname.is_empty() {
                GLOBAL_PHYSICS
            } else {
                lookup(&names.physics, cname)?
            };
            let Some(channels) = channels.as_object() else {
                continue;
            };
            for (channel, keys) in channels {
                let Some(keys) = keys.as_array() else {
                    continue;
                };
                if keys.is_empty() {
                    continue;
                }
                if channel == "reset" {
                    let times: Vec<f32> = keys.iter().map(|k| f(k, "time")).collect();
                    duration = duration.max(times.last().copied().unwrap_or(0.0));
                    timelines.push(Timeline::PhysicsReset(PhysicsResetTimeline::new(
                        idx, times,
                    )));
                    continue;
                }
                let (property, default) = match channel.as_str() {
                    "inertia" => (PhysicsProperty::Inertia, 0.5),
                    "strength" => (PhysicsProperty::Strength, 100.0),
                    "damping" => (PhysicsProperty::Damping, 0.85),
                    "mass" => (PhysicsProperty::Mass, 1.0),
                    "wind" => (PhysicsProperty::Wind, 0.0),
                    "gravity" => (PhysicsProperty::Gravity, 0.0),
                    "mix" => (PhysicsProperty::Mix, 1.0),
                    _ => continue, // Unknown channels are skipped.
                };
                let (tl, dur) = read_curve_timeline(keys, idx, &value_channel(default));
                duration = duration.max(dur);
                timelines.push(Timeline::Physics(tl, property));
            }
        }
    }

    if let Some(slots) = anim.get("slots").and_then(Value::as_object) {
        for (slot_name, channels) in slots {
            let idx = names.slot(slot_name)?;
            let Some(channels) = channels.as_object() else {
                continue;
            };
            for (channel, keys) in channels {
                let Some(keys) = keys.as_array() else {
                    continue;
                };
                if keys.is_empty() {
                    continue;
                }
                match channel.as_str() {
                    "rgba" => {
                        let (tl, dur) = read_slot_rgba_timeline(keys, idx);
                        duration = duration.max(dur);
                        timelines.push(Timeline::SlotColor(tl, true));
                    }
                    "rgb" => {
                        let (tl, dur) = read_slot_rgb_timeline(keys, idx);
                        duration = duration.max(dur);
                        timelines.push(Timeline::SlotColor(tl, false));
                    }
                    "rgba2" => {
                        let (tl, dur) = read_slot_two_color_timeline(keys, idx, true);
                        duration = duration.max(dur);
                        timelines.push(Timeline::SlotTwoColor(tl, true));
                    }
                    "rgb2" => {
                        let (tl, dur) = read_slot_two_color_timeline(keys, idx, false);
                        duration = duration.max(dur);
                        timelines.push(Timeline::SlotTwoColor(tl, false));
                    }
                    "attachment" => {
                        let mut times = Vec::with_capacity(keys.len());
                        let mut names = Vec::with_capacity(keys.len());
                        for k in keys {
                            times.push(f(k, "time"));
                            names.push(k.get("name").and_then(Value::as_str).map(String::from));
                        }
                        duration = duration.max(times.last().copied().unwrap_or(0.0));
                        timelines.push(Timeline::Attachment(AttachmentTimeline::new(
                            idx, times, names,
                        )));
                    }
                    "alpha" => {
                        // A one-channel value curve over the slot tint's alpha.
                        let (tl, dur) = read_curve_timeline(keys, idx, &value_channel(0.0));
                        duration = duration.max(dur);
                        timelines.push(Timeline::SlotAlpha(tl));
                    }
                    _ => {} // unknown slot channels are skipped.
                }
            }
        }
    }

    if let Some(dos) = anim
        .get("drawOrder")
        .or_else(|| anim.get("draworder"))
        .and_then(Value::as_array)
    {
        let slot_count = data.slots.len();
        let mut times = Vec::with_capacity(dos.len());
        let mut orders = Vec::with_capacity(dos.len());
        for k in dos {
            // Each key holds a full slot ordering.
            budget.charge(
                slot_count.saturating_mul(size_of::<usize>()),
                "a draw order key",
            )?;
            times.push(f(k, "time"));
            orders.push(read_draw_order(k, slot_count, names)?);
        }
        if let Some(&dur) = times.last() {
            duration = duration.max(dur);
            timelines.push(Timeline::DrawOrder(DrawOrderTimeline::new(times, orders)));
        }
    }

    if let Some(events) = anim.get("events").and_then(Value::as_array) {
        if !events.is_empty() {
            let (tl, dur) = read_event_timeline(events, data, names, budget)?;
            duration = duration.max(dur);
            timelines.push(Timeline::Event(tl));
        }
    }

    if let Some(deforms) = anim.get("deform").and_then(Value::as_object) {
        for (skin_name, slots) in deforms {
            let skin = if skin_name == "default" {
                None
            } else {
                names.skin(data, skin_name)
            };
            let Some(slots) = slots.as_object() else {
                continue;
            };
            for (slot_name, attachments) in slots {
                let Some(&slot_idx) = names.slots.get(slot_name.as_str()) else {
                    continue;
                };
                let Some(attachments) = attachments.as_object() else {
                    continue;
                };
                for (att_name, keys) in attachments {
                    let Some(keys) = keys.as_array() else {
                        continue;
                    };
                    if keys.is_empty() {
                        continue;
                    }
                    let Some(Attachment::Mesh(mesh)) = data.attachment(slot_idx, att_name, skin)
                    else {
                        continue;
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
                    // Unweighted: setup = the bind vertices (offsets add to them).
                    // Weighted: setup = zeros (offsets are added per-influence to
                    // the bind positions in compute_vertices).
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
                    duration = duration.max(times.last().copied().unwrap_or(0.0));
                    let tl_skin = if skin_name == "default" {
                        None
                    } else {
                        Some(skin_name.clone())
                    };
                    let mut tl = DeformTimeline::new(
                        slot_idx,
                        att_name.clone(),
                        tl_skin,
                        setup,
                        times,
                        frames,
                        n,
                    );
                    let mut bezier = 0;
                    let mut frame = 0;
                    while frame + 1 < n {
                        if let Some(curve) = keys[frame].get("curve") {
                            let time = f(&keys[frame], "time");
                            let time2 = f(&keys[frame + 1], "time");
                            bezier =
                                read_curve(curve, &mut tl, bezier, frame, 0, time, time2, 0.0, 1.0);
                        }
                        frame += 1;
                    }
                    timelines.push(Timeline::Deform(tl));
                }
            }
        }
    }

    // Slider constraint timelines (Spine 4.3): "time" and "mix", each a
    // one-value curve keyed by slider name.
    if let Some(sliders) = anim.get("slider").and_then(Value::as_object) {
        for (slider_name, channels) in sliders {
            // The JSON loader does not parse slider constraints (only the binary
            // loader does), so a slider timeline with no matching constraint is
            // skipped, not an error.
            let Some(idx) = data.sliders.iter().position(|s| s.name == *slider_name) else {
                continue;
            };
            let Some(channels) = channels.as_object() else {
                continue;
            };
            for (channel, keys) in channels {
                let Some(keys) = keys.as_array() else {
                    continue;
                };
                if keys.is_empty() {
                    continue;
                }
                match channel.as_str() {
                    "time" => {
                        let (tl, dur) = read_curve_timeline(keys, idx, &value_channel(1.0));
                        duration = duration.max(dur);
                        timelines.push(Timeline::SliderTime(tl));
                    }
                    "mix" => {
                        let (tl, dur) = read_curve_timeline(keys, idx, &value_channel(1.0));
                        duration = duration.max(dur);
                        timelines.push(Timeline::SliderMix(tl));
                    }
                    _ => {}
                }
            }
        }
    }

    // Attachment sequence (flipbook) timelines: skin -> slot -> attachment ->
    // "sequence", each keyframe a packed mode/index plus a hold delay.
    if let Some(attachments) = anim.get("attachments").and_then(Value::as_object) {
        for (skin_name, slots) in attachments {
            let skin = if skin_name == "default" {
                None
            } else {
                names.skin(data, skin_name)
            };
            let Some(slots) = slots.as_object() else {
                continue;
            };
            for (slot_name, atts) in slots {
                let Some(&slot_idx) = names.slots.get(slot_name.as_str()) else {
                    continue;
                };
                let Some(atts) = atts.as_object() else {
                    continue;
                };
                for (att_name, channels) in atts {
                    let Some(seq_keys) = channels.get("sequence").and_then(Value::as_array) else {
                        continue;
                    };
                    if seq_keys.is_empty() {
                        continue;
                    }
                    let count = match data.attachment(slot_idx, att_name, skin) {
                        Some(Attachment::Region(r)) => r.sequence.as_ref().map_or(0, |s| s.count),
                        Some(Attachment::Mesh(m)) => m.sequence.as_ref().map_or(0, |s| s.count),
                        _ => 0,
                    };
                    let n = seq_keys.len();
                    let mut times = Vec::with_capacity(n);
                    let mut mode_and_index = Vec::with_capacity(n);
                    let mut delays = Vec::with_capacity(n);
                    for k in seq_keys {
                        // The index shares a `u32` with the 4-bit mode, so a
                        // larger index saturates instead of losing its high bits.
                        let index = k.get("index").and_then(Value::as_u64).map_or(0, |i| {
                            u32::try_from(i).unwrap_or(u32::MAX).min(u32::MAX >> 4)
                        });
                        let mode = sequence_mode(k.get("mode").and_then(Value::as_str));
                        times.push(f(k, "time"));
                        mode_and_index.push((index << 4) | mode);
                        delays.push(f(k, "delay"));
                    }
                    duration = duration.max(times.last().copied().unwrap_or(0.0));
                    timelines.push(Timeline::Sequence(SequenceTimeline::new(
                        slot_idx,
                        att_name.clone(),
                        count,
                        times,
                        mode_and_index,
                        delays,
                    )));
                }
            }
        }
    }

    Ok(Animation::new(name, duration, timelines))
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

/// Read one draw-order keyframe's `offsets` into a full slot ordering (the setup
/// order when there are no offsets). Offsets naming an unknown slot are skipped.
/// Spine moves each listed slot to `slot + offset`, so an offset that lands
/// outside the slot list, a slot listed twice, or two slots moved to one
/// position is corrupt data and is rejected.
fn read_draw_order(k: &Value, slot_count: usize, names: &Names) -> Result<Vec<usize>, LoadError> {
    let corrupt = || LoadError::Schema("draw order offsets do not form a slot order".to_string());
    let mut offsets: Vec<(usize, i32)> = Vec::new();
    if let Some(offs) = k.get("offsets").and_then(Value::as_array) {
        for o in offs {
            let Some(slot_name) = o.get("slot").and_then(Value::as_str) else {
                continue;
            };
            let Some(&slot_index) = names.slots.get(slot_name) else {
                continue;
            };
            let offset = o.get("offset").and_then(Value::as_i64).unwrap_or(0);
            let in_range = i64::try_from(slot_index)
                .ok()
                .and_then(|slot| slot.checked_add(offset))
                .and_then(|target| usize::try_from(target).ok())
                .is_some_and(|target| target < slot_count);
            if !in_range {
                return Err(corrupt());
            }
            let offset = i32::try_from(offset).map_err(|_| corrupt())?;
            offsets.push((slot_index, offset));
        }
    }
    if offsets.is_empty() {
        return Ok((0..slot_count).collect());
    }
    offsets.sort_unstable_by_key(|&(slot, _)| slot);
    if offsets
        .windows(2)
        .any(|pair| matches!(pair, [a, b] if a.0 == b.0))
    {
        return Err(corrupt());
    }
    let order = compute_draw_order(slot_count, &mut offsets);
    // A position no slot filled keeps an out-of-range marker: two slots moved
    // to the same position.
    if order.iter().any(|&slot| slot >= slot_count) {
        return Err(corrupt());
    }
    Ok(order)
}

/// Read an animation event timeline: each keyframe's event, resolving its value
/// overrides against the named event's setup defaults.
fn read_event_timeline(
    keys: &[Value],
    data: &SkeletonData,
    names: &Names,
    budget: &mut Budget,
) -> Result<(EventTimeline, f32), LoadError> {
    let mut times = Vec::with_capacity(keys.len());
    let mut events = Vec::with_capacity(keys.len());
    let mut duration = 0.0_f32;
    for k in keys {
        let time = f(k, "time");
        let name = k.get("name").and_then(Value::as_str).unwrap_or("");
        let setup = names.events.get(name).and_then(|&i| data.events.get(i));
        let (di, df, ds, dv, db) = match setup {
            Some(e) => (
                e.int_value,
                e.float_value,
                e.string_value.as_str(),
                e.volume,
                e.balance,
            ),
            None => (0, 0.0, "", 1.0, 0.0),
        };
        let string_value = match k.get("string").and_then(Value::as_str) {
            Some(s) => s.to_string(),
            None => {
                // The key copies its event's setup string.
                budget.charge(ds.len(), "an event key")?;
                ds.to_string()
            }
        };
        events.push(Event {
            name: name.to_string(),
            time,
            int_value: k
                .get("int")
                .and_then(Value::as_i64)
                .map_or(di, |v| v as i32),
            float_value: f_or(k, "float", df),
            string_value,
            volume: f_or(k, "volume", dv),
            balance: f_or(k, "balance", db),
        });
        times.push(time);
        duration = duration.max(time);
    }
    Ok((EventTimeline::new(times, events), duration))
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
