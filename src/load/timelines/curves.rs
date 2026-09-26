//! Keyframe curve reading for the JSON loader: one- and two-value bone
//! timelines, N-channel constraint timelines, IK and slot color timelines,
//! and the stepped or Bezier `curve` field they share.

use super::*;

/// Read a one-value (rotate) timeline. Returns the timeline and its last
/// keyframe time. Mirrors Spine's `readTimeline` for `CurveTimeline1`.
pub(super) fn read_timeline1(
    keys: &[Value],
    bone: usize,
    default_value: f32,
) -> (BoneTimeline, f32) {
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
pub(super) fn read_timeline2(
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
pub(super) trait CurveBuilder {
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
/// array, which holds 4 floats per value channel at offset `value_ord * 4`).
/// Mirrors Spine's `readCurve`.
#[allow(clippy::too_many_arguments)]
pub(super) fn read_curve(
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

/// One keyframe's channel values: each channel's key, or its fallback when
/// the key is absent.
fn channel_values(k: &Value, channels: &[(&str, Fallback)]) -> Vec<f32> {
    let mut values: Vec<f32> = Vec::with_capacity(channels.len());
    for &(field, fallback) in channels {
        let default = match fallback {
            Fallback::Value(v) => v,
            Fallback::Channel(i) => values.get(i).copied().unwrap_or(0.0),
        };
        values.push(f_or(k, field, default));
    }
    values
}

/// Read a generic N-channel constraint timeline (all channels Bezier-curved).
/// `channels` is `(json field, fallback)` per channel.
pub(super) fn read_curve_timeline(
    keys: &[Value],
    constraint: usize,
    channels: &[(&str, Fallback)],
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
        let values = channel_values(k, channels);
        tl.set_frame(frame, time, &values);
        duration = duration.max(time);
        if frame + 1 < n {
            if let Some(curve) = k.get("curve") {
                let next = &keys[frame + 1];
                let time2 = f(next, "time");
                let next_values = channel_values(next, channels);
                for (ci, (&v1, &v2)) in values.iter().zip(&next_values).enumerate() {
                    bezier = read_curve(curve, &mut tl, bezier, frame, ci, time, time2, v1, v2);
                }
            }
        }
        frame += 1;
    }
    (tl, duration)
}

/// A one-channel curve timeline layout whose key is `value`.
pub(super) fn value_channel(default: f32) -> [(&'static str, Fallback); 1] {
    [("value", Fallback::Value(default))]
}

/// Read an IK constraint timeline: mix and softness are Bezier-curved. Bend
/// direction, compress, and stretch are stored stepped.
pub(super) fn read_ik_timeline(keys: &[Value], constraint: usize) -> (ConstraintTimeline, f32) {
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
pub(super) fn read_slot_rgba_timeline(keys: &[Value], slot: usize) -> (ConstraintTimeline, f32) {
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
/// hex string, and the slot's alpha is left unchanged).
pub(super) fn read_slot_rgb_timeline(keys: &[Value], slot: usize) -> (ConstraintTimeline, f32) {
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
pub(super) fn read_slot_two_color_timeline(
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
