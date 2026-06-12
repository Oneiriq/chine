use super::*;

use crate::anim::{
    compute_draw_order, Animation, AttachmentTimeline, BoneAxis, BoneTimeline, ConstraintTimeline,
    DeformTimeline, DrawOrderTimeline, EventTimeline, PhysicsProperty, PhysicsResetTimeline,
    SequenceTimeline, Timeline, GLOBAL_PHYSICS, PATH_MIX, PATH_POSITION, PATH_SPACING,
    TRANSFORM_MIX,
};
use crate::event::Event;

/// Parse one animation: bone, slot, deform, event, draw-order, and constraint
/// timelines.
pub(super) fn parse_animation(
    name: &str,
    anim: &Value,
    data: &SkeletonData,
) -> Result<Animation, LoadError> {
    let mut timelines = Vec::new();
    let mut duration = 0.0_f32;
    if let Some(bones) = anim.get("bones").and_then(Value::as_object) {
        for (bone_name, props) in bones {
            let bone = data
                .find_bone(bone_name)
                .ok_or_else(|| LoadError::BadReference(bone_name.clone()))?;
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
            let idx = data
                .ik_constraints
                .iter()
                .position(|c| c.name == *cname)
                .ok_or_else(|| LoadError::BadReference(cname.clone()))?;
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
            let idx = data
                .transform_constraints
                .iter()
                .position(|c| c.name == *cname)
                .ok_or_else(|| LoadError::BadReference(cname.clone()))?;
            let (tl, dur) = read_curve_timeline(keys, idx, TRANSFORM_MIX);
            duration = duration.max(dur);
            timelines.push(Timeline::TransformMix(tl));
        }
    }
    if let Some(pcs) = anim.get("path").and_then(Value::as_object) {
        for (cname, channels) in pcs {
            let idx = data
                .path_constraints
                .iter()
                .position(|c| c.name == *cname)
                .ok_or_else(|| LoadError::BadReference(cname.clone()))?;
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
                data.physics_constraints
                    .iter()
                    .position(|c| c.name == *cname)
                    .ok_or_else(|| LoadError::BadReference(cname.clone()))?
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
                let (tl, dur) = read_curve_timeline(keys, idx, &[("value", default)]);
                duration = duration.max(dur);
                timelines.push(Timeline::Physics(tl, property));
            }
        }
    }

    if let Some(slots) = anim.get("slots").and_then(Value::as_object) {
        for (slot_name, channels) in slots {
            let idx = data
                .find_slot(slot_name)
                .ok_or_else(|| LoadError::BadReference(slot_name.clone()))?;
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
                        let (tl, dur) = read_curve_timeline(keys, idx, &[("value", 0.0)]);
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
            times.push(f(k, "time"));
            orders.push(read_draw_order(k, slot_count, data));
        }
        if let Some(&dur) = times.last() {
            duration = duration.max(dur);
            timelines.push(Timeline::DrawOrder(DrawOrderTimeline::new(times, orders)));
        }
    }

    if let Some(events) = anim.get("events").and_then(Value::as_array) {
        if !events.is_empty() {
            let (tl, dur) = read_event_timeline(events, data);
            duration = duration.max(dur);
            timelines.push(Timeline::Event(tl));
        }
    }

    if let Some(deforms) = anim.get("deform").and_then(Value::as_object) {
        for (skin_name, slots) in deforms {
            let skin = if skin_name == "default" {
                None
            } else {
                data.find_skin(skin_name)
            };
            let Some(slots) = slots.as_object() else {
                continue;
            };
            for (slot_name, attachments) in slots {
                let Some(slot_idx) = data.find_slot(slot_name) else {
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
                    // Unweighted: setup = the bind vertices (offsets add to them).
                    // Weighted: setup = zeros (offsets are added per-influence to
                    // the bind positions in compute_vertices).
                    let setup = match mesh.setup_vertices() {
                        Some(v) => v.to_vec(),
                        None => vec![0.0; frame_len],
                    };
                    let n = keys.len();
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
            // JSON slider constraint setup is not parsed yet (binary only), so a
            // slider animation with no matching constraint is skipped, not an error.
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
                        let (tl, dur) = read_curve_timeline(keys, idx, &[("value", 1.0)]);
                        duration = duration.max(dur);
                        timelines.push(Timeline::SliderTime(tl));
                    }
                    "mix" => {
                        let (tl, dur) = read_curve_timeline(keys, idx, &[("value", 1.0)]);
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
                data.find_skin(skin_name)
            };
            let Some(slots) = slots.as_object() else {
                continue;
            };
            for (slot_name, atts) in slots {
                let Some(slot_idx) = data.find_slot(slot_name) else {
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
                        let index = k.get("index").and_then(Value::as_u64).unwrap_or(0) as u32;
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

/// Read a one-value (rotate) timeline. Returns the timeline and its last
/// keyframe time. Mirrors Spine's `readTimeline` for `CurveTimeline1`.
fn read_timeline1(keys: &[Value], bone: usize, default_value: f32) -> (BoneTimeline, f32) {
    let n = keys.len();
    let mut tl = BoneTimeline::one_value(bone, n, n);
    let mut bezier = 0;
    let mut duration = 0.0_f32;
    let mut frame = 0;
    while frame < n {
        let k = &keys[frame];
        let time = f(k, "time");
        let value = f_or(k, "value", default_value);
        tl.set_frame1(frame, time, value);
        duration = duration.max(time);
        if frame + 1 < n {
            if let Some(curve) = k.get("curve") {
                let next = &keys[frame + 1];
                let time2 = f(next, "time");
                let value2 = f_or(next, "value", default_value);
                bezier = read_curve(curve, &mut tl, bezier, frame, 0, time, time2, value, value2);
            }
        }
        frame += 1;
    }
    (tl, duration)
}

/// Read a two-value (translate / scale) timeline. Mirrors Spine's `readTimeline`
/// for `BoneTimeline2`.
fn read_timeline2(
    keys: &[Value],
    bone: usize,
    name1: &str,
    name2: &str,
    default_value: f32,
) -> (BoneTimeline, f32) {
    let n = keys.len();
    let mut tl = BoneTimeline::two_value(bone, n, n * 2);
    let mut bezier = 0;
    let mut duration = 0.0_f32;
    let mut frame = 0;
    while frame < n {
        let k = &keys[frame];
        let time = f(k, "time");
        let v1 = f_or(k, name1, default_value);
        let v2 = f_or(k, name2, default_value);
        tl.set_frame2(frame, time, v1, v2);
        duration = duration.max(time);
        if frame + 1 < n {
            if let Some(curve) = k.get("curve") {
                let next = &keys[frame + 1];
                let time2 = f(next, "time");
                let nv1 = f_or(next, name1, default_value);
                let nv2 = f_or(next, name2, default_value);
                bezier = read_curve(curve, &mut tl, bezier, frame, 0, time, time2, v1, nv1);
                bezier = read_curve(curve, &mut tl, bezier, frame, 1, time, time2, v2, nv2);
            }
        }
        frame += 1;
    }
    (tl, duration)
}

/// A timeline whose curve interpolation the loader can configure (stepped or
/// Bezier), implemented by bone and constraint timelines so [`read_curve`] works
/// for both.
trait CurveBuilder {
    fn stepped(&mut self, frame: usize);
    #[allow(clippy::too_many_arguments)]
    fn bezier(
        &mut self,
        bezier: usize,
        frame: usize,
        value: usize,
        time1: f32,
        value1: f32,
        cx1: f32,
        cy1: f32,
        cx2: f32,
        cy2: f32,
        time2: f32,
        value2: f32,
    );
}

impl CurveBuilder for BoneTimeline {
    fn stepped(&mut self, frame: usize) {
        self.set_stepped(frame);
    }
    #[allow(clippy::too_many_arguments)]
    fn bezier(
        &mut self,
        bezier: usize,
        frame: usize,
        value: usize,
        time1: f32,
        value1: f32,
        cx1: f32,
        cy1: f32,
        cx2: f32,
        cy2: f32,
        time2: f32,
        value2: f32,
    ) {
        self.set_bezier(
            bezier, frame, value, time1, value1, cx1, cy1, cx2, cy2, time2, value2,
        );
    }
}

impl CurveBuilder for ConstraintTimeline {
    fn stepped(&mut self, frame: usize) {
        self.set_stepped(frame);
    }
    #[allow(clippy::too_many_arguments)]
    fn bezier(
        &mut self,
        bezier: usize,
        frame: usize,
        value: usize,
        time1: f32,
        value1: f32,
        cx1: f32,
        cy1: f32,
        cx2: f32,
        cy2: f32,
        time2: f32,
        value2: f32,
    ) {
        self.set_bezier(
            bezier, frame, value, time1, value1, cx1, cy1, cx2, cy2, time2, value2,
        );
    }
}

impl CurveBuilder for DeformTimeline {
    fn stepped(&mut self, frame: usize) {
        self.set_stepped(frame);
    }
    #[allow(clippy::too_many_arguments)]
    fn bezier(
        &mut self,
        bezier: usize,
        frame: usize,
        value: usize,
        time1: f32,
        value1: f32,
        cx1: f32,
        cy1: f32,
        cx2: f32,
        cy2: f32,
        time2: f32,
        value2: f32,
    ) {
        self.set_bezier(
            bezier, frame, value, time1, value1, cx1, cy1, cx2, cy2, time2, value2,
        );
    }
}

/// Apply one keyframe's `curve` field (absent = linear, `"stepped"`, or a Bezier
/// array; the array holds 4 floats per value channel at offset `value_ord * 4`).
/// Mirrors Spine's `readCurve`.
#[allow(clippy::too_many_arguments)]
fn read_curve(
    curve: &Value,
    tl: &mut impl CurveBuilder,
    bezier: usize,
    frame: usize,
    value_ord: usize,
    time1: f32,
    time2: f32,
    value1: f32,
    value2: f32,
) -> usize {
    if let Some(s) = curve.as_str() {
        if s == "stepped" {
            tl.stepped(frame);
        }
        return bezier;
    }
    let Some(arr) = curve.as_array() else {
        return bezier;
    };
    let base = value_ord * 4;
    let at = |i: usize| arr.get(base + i).and_then(Value::as_f64).unwrap_or(0.0) as f32;
    tl.bezier(
        bezier,
        frame,
        value_ord,
        time1,
        value1,
        at(0),
        at(1),
        at(2),
        at(3),
        time2,
        value2,
    );
    bezier + 1
}

/// Read a generic N-channel constraint timeline (all channels Bezier-curved).
/// `channels` is `(json field, default)` per channel.
fn read_curve_timeline(
    keys: &[Value],
    constraint: usize,
    channels: &[(&str, f32)],
) -> (ConstraintTimeline, f32) {
    let n = keys.len();
    let nc = channels.len();
    let mut tl = ConstraintTimeline::new(constraint, n, n * nc, nc + 1);
    let mut bezier = 0;
    let mut duration = 0.0_f32;
    let mut frame = 0;
    while frame < n {
        let k = &keys[frame];
        let time = f(k, "time");
        let values: Vec<f32> = channels
            .iter()
            .map(|(field, def)| f_or(k, field, *def))
            .collect();
        tl.set_frame(frame, time, &values);
        duration = duration.max(time);
        if frame + 1 < n {
            if let Some(curve) = k.get("curve") {
                let next = &keys[frame + 1];
                let time2 = f(next, "time");
                for (ci, (field, def)) in channels.iter().enumerate() {
                    let v2 = f_or(next, field, *def);
                    bezier = read_curve(
                        curve, &mut tl, bezier, frame, ci, time, time2, values[ci], v2,
                    );
                }
            }
        }
        frame += 1;
    }
    (tl, duration)
}

/// Read an IK constraint timeline: mix and softness are Bezier-curved; bend
/// direction, compress, and stretch are stored stepped.
fn read_ik_timeline(keys: &[Value], constraint: usize) -> (ConstraintTimeline, f32) {
    let n = keys.len();
    let mut tl = ConstraintTimeline::new(constraint, n, n * 2, 6);
    let mut bezier = 0;
    let mut duration = 0.0_f32;
    let mut frame = 0;
    while frame < n {
        let k = &keys[frame];
        let time = f(k, "time");
        let mix = f_or(k, "mix", 1.0);
        let softness = f(k, "softness");
        let bend = if bool_or(k, "bendPositive", true) {
            1.0
        } else {
            -1.0
        };
        let compress = f32::from(bool_or(k, "compress", false));
        let stretch = f32::from(bool_or(k, "stretch", false));
        tl.set_frame(frame, time, &[mix, softness, bend, compress, stretch]);
        duration = duration.max(time);
        if frame + 1 < n {
            if let Some(curve) = k.get("curve") {
                let next = &keys[frame + 1];
                let time2 = f(next, "time");
                let mix2 = f_or(next, "mix", 1.0);
                let soft2 = f(next, "softness");
                bezier = read_curve(curve, &mut tl, bezier, frame, 0, time, time2, mix, mix2);
                bezier = read_curve(
                    curve, &mut tl, bezier, frame, 1, time, time2, softness, soft2,
                );
            }
        }
        frame += 1;
    }
    (tl, duration)
}

/// Read a slot RGBA color timeline (four channels from each keyframe's `color`
/// hex string).
fn read_slot_rgba_timeline(keys: &[Value], slot: usize) -> (ConstraintTimeline, f32) {
    let n = keys.len();
    let mut tl = ConstraintTimeline::new(slot, n, n * 4, 5);
    let mut bezier = 0;
    let mut duration = 0.0_f32;
    let mut frame = 0;
    while frame < n {
        let k = &keys[frame];
        let time = f(k, "time");
        let c = parse_color(k.get("color").and_then(Value::as_str), Color::WHITE);
        tl.set_frame(frame, time, &[c.r, c.g, c.b, c.a]);
        duration = duration.max(time);
        if frame + 1 < n {
            if let Some(curve) = k.get("curve") {
                let next = &keys[frame + 1];
                let time2 = f(next, "time");
                let c2 = parse_color(next.get("color").and_then(Value::as_str), Color::WHITE);
                bezier = read_curve(curve, &mut tl, bezier, frame, 0, time, time2, c.r, c2.r);
                bezier = read_curve(curve, &mut tl, bezier, frame, 1, time, time2, c.g, c2.g);
                bezier = read_curve(curve, &mut tl, bezier, frame, 2, time, time2, c.b, c2.b);
                bezier = read_curve(curve, &mut tl, bezier, frame, 3, time, time2, c.a, c2.a);
            }
        }
        frame += 1;
    }
    (tl, duration)
}

/// Read a slot RGB color timeline (three channels from each keyframe's `color`
/// hex string; the slot's alpha is left unchanged).
fn read_slot_rgb_timeline(keys: &[Value], slot: usize) -> (ConstraintTimeline, f32) {
    let n = keys.len();
    let mut tl = ConstraintTimeline::new(slot, n, n * 3, 4);
    let mut bezier = 0;
    let mut duration = 0.0_f32;
    let mut frame = 0;
    while frame < n {
        let k = &keys[frame];
        let time = f(k, "time");
        let c = parse_color(k.get("color").and_then(Value::as_str), Color::WHITE);
        tl.set_frame(frame, time, &[c.r, c.g, c.b]);
        duration = duration.max(time);
        if frame + 1 < n {
            if let Some(curve) = k.get("curve") {
                let next = &keys[frame + 1];
                let time2 = f(next, "time");
                let c2 = parse_color(next.get("color").and_then(Value::as_str), Color::WHITE);
                bezier = read_curve(curve, &mut tl, bezier, frame, 0, time, time2, c.r, c2.r);
                bezier = read_curve(curve, &mut tl, bezier, frame, 1, time, time2, c.g, c2.g);
                bezier = read_curve(curve, &mut tl, bezier, frame, 2, time, time2, c.b, c2.b);
            }
        }
        frame += 1;
    }
    (tl, duration)
}

/// Read a slot two-color timeline: the light tint (RGB, or RGBA when
/// `light_alpha`) from each keyframe's `light` hex, then the dark tint (RGB)
/// from `dark`.
fn read_slot_two_color_timeline(
    keys: &[Value],
    slot: usize,
    light_alpha: bool,
) -> (ConstraintTimeline, f32) {
    let nc = if light_alpha { 7 } else { 6 };
    let n = keys.len();
    let mut tl = ConstraintTimeline::new(slot, n, n * nc, nc + 1);
    let channels = |k: &Value| -> Vec<f32> {
        let light = parse_color(k.get("light").and_then(Value::as_str), Color::WHITE);
        let dark = parse_color(
            k.get("dark").and_then(Value::as_str),
            Color::new(0.0, 0.0, 0.0, 1.0),
        );
        let mut v = vec![light.r, light.g, light.b];
        if light_alpha {
            v.push(light.a);
        }
        v.extend([dark.r, dark.g, dark.b]);
        v
    };
    let mut bezier = 0;
    let mut duration = 0.0_f32;
    let mut frame = 0;
    while frame < n {
        let k = &keys[frame];
        let time = f(k, "time");
        let vals = channels(k);
        tl.set_frame(frame, time, &vals);
        duration = duration.max(time);
        if frame + 1 < n {
            if let Some(curve) = k.get("curve") {
                let next = &keys[frame + 1];
                let time2 = f(next, "time");
                let next_vals = channels(next);
                for ci in 0..nc {
                    bezier = read_curve(
                        curve,
                        &mut tl,
                        bezier,
                        frame,
                        ci,
                        time,
                        time2,
                        vals[ci],
                        next_vals[ci],
                    );
                }
            }
        }
        frame += 1;
    }
    (tl, duration)
}

/// Read one draw-order keyframe's `offsets` into a full slot ordering (the setup
/// order when there are no offsets).
fn read_draw_order(k: &Value, slot_count: usize, data: &SkeletonData) -> Vec<usize> {
    let mut offsets: Vec<(usize, i32)> = Vec::new();
    if let Some(offs) = k.get("offsets").and_then(Value::as_array) {
        for o in offs {
            let Some(slot_name) = o.get("slot").and_then(Value::as_str) else {
                continue;
            };
            let Some(slot_index) = data.find_slot(slot_name) else {
                continue;
            };
            let offset = o.get("offset").and_then(Value::as_i64).unwrap_or(0) as i32;
            offsets.push((slot_index, offset));
        }
    }
    if offsets.is_empty() {
        return (0..slot_count).collect();
    }
    compute_draw_order(slot_count, &mut offsets)
}

/// Read an animation event timeline: each keyframe's event, resolving its value
/// overrides against the named event's setup defaults.
fn read_event_timeline(keys: &[Value], data: &SkeletonData) -> (EventTimeline, f32) {
    let mut times = Vec::with_capacity(keys.len());
    let mut events = Vec::with_capacity(keys.len());
    let mut duration = 0.0_f32;
    for k in keys {
        let time = f(k, "time");
        let name = k.get("name").and_then(Value::as_str).unwrap_or("");
        let setup = data.events.iter().find(|e| e.name == name);
        let (di, df, ds, dv, db) = match setup {
            Some(e) => (
                e.int_value,
                e.float_value,
                e.string_value.clone(),
                e.volume,
                e.balance,
            ),
            None => (0, 0.0, String::new(), 1.0, 0.0),
        };
        events.push(Event {
            name: name.to_string(),
            time,
            int_value: k
                .get("int")
                .and_then(Value::as_i64)
                .map_or(di, |v| v as i32),
            float_value: f_or(k, "float", df),
            string_value: k
                .get("string")
                .and_then(Value::as_str)
                .map_or(ds, String::from),
            volume: f_or(k, "volume", dv),
            balance: f_or(k, "balance", db),
        });
        times.push(time);
        duration = duration.max(time);
    }
    (EventTimeline::new(times, events), duration)
}

/// Read one deform keyframe's sparse `offset`/`vertices` into a full `len`-long
/// offset array (zero where unspecified).
fn read_deform_frame(k: &Value, len: usize) -> Vec<f32> {
    let mut frame = vec![0.0; len];
    let offset = k.get("offset").and_then(Value::as_u64).unwrap_or(0) as usize;
    for (j, v) in f_array(k, "vertices").into_iter().enumerate() {
        if offset + j < len {
            frame[offset + j] = v;
        }
    }
    frame
}
