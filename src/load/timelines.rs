use super::*;

use crate::anim::{
    compute_draw_order, sort_draw_order_moves, Animation, AttachmentTarget, AttachmentTimeline,
    BoneAxis, BoneTimeline, ConstraintTimeline, DeformTimeline, DrawOrderFolderTimeline,
    DrawOrderTimeline, EventTimeline, Fallback, InheritTimeline, PhysicsProperty,
    PhysicsResetTimeline, SequenceTimeline, Timeline, GLOBAL_PHYSICS, PATH_MIX, PATH_POSITION,
    PATH_SPACING, TRANSFORM_MIX,
};
use crate::attach::AttachmentKey;
use crate::event::Event;

mod attachments;
use attachments::{attachment_entries, read_deform, read_sequence};
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

    // Draw order folder timelines (Spine 4.3): the draw order of a folder of
    // slots, each keyed like the draw order over the folder's slots.
    if let Some(folders) = anim.get("drawOrderFolder").and_then(Value::as_array) {
        for folder in folders {
            if let Some((tl, dur)) = read_draw_order_folder(folder, names)? {
                duration = duration.max(dur);
                timelines.push(tl);
            }
        }
    }

    if let Some(events) = anim.get("events").and_then(Value::as_array) {
        if !events.is_empty() {
            let (tl, dur) = read_event_timeline(events, data, names, budget)?;
            duration = duration.max(dur);
            timelines.push(Timeline::Event(tl));
        }
    }

    // Spine 4.0 exports keep deform timelines in their own "deform" map. Later
    // exports nest them under "attachments", read below.
    if let Some(map) = anim.get("deform") {
        for entry in attachment_entries(map, names) {
            if let Some((tl, d)) = read_deform(&entry, entry.value, data, names, budget)? {
                duration = duration.max(d);
                timelines.push(tl);
            }
        }
    }

    // Slider constraint timelines (Spine 4.3): "time" and "mix", each a
    // one-value curve keyed by slider name.
    if let Some(sliders) = anim.get("slider").and_then(Value::as_object) {
        for (slider_name, channels) in sliders {
            let idx = lookup(&names.sliders, slider_name)?;
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

    // Attachment timelines, keyed by skin, slot, and attachment: mesh deforms
    // and sequence (flipbook) keys.
    if let Some(map) = anim.get("attachments") {
        for entry in attachment_entries(map, names) {
            if let Some(keys) = entry.value.get("deform") {
                if let Some((tl, d)) = read_deform(&entry, keys, data, names, budget)? {
                    duration = duration.max(d);
                    timelines.push(tl);
                }
            }
            if let Some(keys) = entry.value.get("sequence") {
                if let Some((tl, d)) = read_sequence(&entry, keys, data, names) {
                    duration = duration.max(d);
                    timelines.push(tl);
                }
            }
        }
    }

    Ok(Animation::new(name, duration, timelines))
}

/// Read one draw order folder timeline: its `slots` (in setup order), then
/// per key a time and `offsets` that move folder slots by an offset within
/// the folder. `None` when the folder has no slots or no keys.
///
/// # Errors
/// Returns [`LoadError::BadReference`] for an unknown slot, and
/// [`LoadError::Schema`] for a slot listed twice in the folder, or for key
/// offsets that name a slot outside the folder or do not order it.
fn read_draw_order_folder(
    folder: &Value,
    names: &Names,
) -> Result<Option<(Timeline, f32)>, LoadError> {
    let corrupt = |problem: &str| LoadError::Schema(format!("draw order folder {problem}"));
    let mut slots = Vec::new();
    for slot in folder
        .get("slots")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        slots.push(names.slot(slot.as_str().unwrap_or_default())?);
    }
    let mut sorted = slots.clone();
    sorted.sort_unstable();
    if sorted.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(corrupt("lists a slot twice"));
    }
    // A slot's position in the folder.
    let positions: HashMap<usize, usize> = slots.iter().enumerate().map(|(i, &s)| (s, i)).collect();
    let keys = folder.get("keys").and_then(Value::as_array);
    let mut times = Vec::new();
    let mut moves = Vec::new();
    for k in keys.into_iter().flatten() {
        let mut key = Vec::new();
        for offset in k
            .get("offsets")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let slot = names.slot(str_ref(offset, "slot")?)?;
            let position = *positions
                .get(&slot)
                .ok_or_else(|| corrupt("moves a slot outside the folder"))?;
            let by = offset.get("offset").and_then(Value::as_i64).unwrap_or(0);
            let by = i32::try_from(by).map_err(|_| corrupt("moves a slot out of range"))?;
            key.push((position, by));
        }
        if !sort_draw_order_moves(slots.len(), &mut key) {
            return Err(corrupt("offsets do not form an order"));
        }
        times.push(f(k, "time"));
        moves.push(key);
    }
    let Some(&duration) = times.last() else {
        return Ok(None);
    };
    if slots.is_empty() {
        return Ok(None);
    }
    let timeline = DrawOrderFolderTimeline::new(slots, times, moves);
    Ok(Some((Timeline::DrawOrderFolder(timeline), duration)))
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
