//! Keyframe curves: stepped / linear / Bezier interpolation.
//!
//! [`Curve`] re-implements Spine 4.3's `CurveTimeline` storage and evaluation,
//! so interpolated values match: keyframe `frames` interleave a time with one or
//! two values, and a parallel `curves` array holds either a per-frame
//! interpolation type (`LINEAR` / `STEPPED`) or an index into
//! forward-difference-sampled Bezier segments.
//!
//! Every read and write is bounds-checked. A frame or Bezier index past the
//! storage is skipped on write, and a read that finds no data falls back to
//! holding the frame's value, so corrupt counts cannot panic.

use super::MixFrom;

/// Curve-type sentinels stored in `curves`. Types `>= BEZIER` index the packed
/// Bezier segments.
const LINEAR: f32 = 0.0;
const STEPPED: f32 = 1.0;
const BEZIER: usize = 2;
/// Floats stored per Bezier curve (10 sampled points, x/y interleaved, less 2).
const BEZIER_SIZE: usize = 18;

/// One sampled Bezier segment: 9 `(x, y)` points, interleaved.
type BezierTable = [f32; BEZIER_SIZE];

/// Shared keyframe storage with stepped / linear / Bezier interpolation.
#[derive(Debug, Clone)]
pub(crate) struct Curve {
    /// Interleaved `[time, v1, (v2)]` per frame, with `entries` floats per frame.
    frames: Vec<f32>,
    /// Per-frame interpolation type, followed by packed Bezier segments.
    curves: Vec<f32>,
    /// Floats per frame: 2 for one-value timelines, 3 for two-value.
    entries: usize,
}

impl Curve {
    /// Storage for `frame_count` frames (`entries` floats each) and up to
    /// `bezier_count` Bezier curves. Sizes that overflow `usize` give a curve
    /// with no frames, which timelines apply as a no-op.
    pub(crate) fn new(frame_count: usize, bezier_count: usize, entries: usize) -> Self {
        // Each Bezier segment belongs to one value of one frame, so a curve can
        // use at most `frame_count * (entries - 1)` of them. The cap keeps a
        // corrupt declared count from sizing the allocation.
        let bezier_count = bezier_count.min(frame_count.saturating_mul(entries.saturating_sub(1)));
        let frames_len = frame_count.checked_mul(entries);
        let curves_len = bezier_count
            .checked_mul(BEZIER_SIZE)
            .and_then(|n| n.checked_add(frame_count));
        let (Some(frames_len), Some(curves_len)) = (frames_len, curves_len) else {
            return Self {
                frames: Vec::new(),
                curves: Vec::new(),
                entries,
            };
        };
        let mut curves = vec![0.0; curves_len];
        // The final frame has no following frame. STEPPED holds its value.
        if let Some(last) = frame_count.checked_sub(1).and_then(|i| curves.get_mut(i)) {
            *last = STEPPED;
        }
        Self {
            frames: vec![0.0; frames_len],
            curves,
            entries,
        }
    }

    fn frame_count(&self) -> usize {
        self.frames.len().checked_div(self.entries).unwrap_or(0)
    }

    /// The time of the first keyframe, or `None` when the curve has no frames.
    pub(crate) fn first_time(&self) -> Option<f32> {
        self.frames.first().copied()
    }

    /// The `entries` floats of `frame`, or `None` past the last frame.
    fn frame_mut(&mut self, frame: usize) -> Option<&mut [f32]> {
        let start = frame.checked_mul(self.entries)?;
        let end = start.checked_add(self.entries)?;
        self.frames.get_mut(start..end)
    }

    /// Set a one-value frame (`entries == 2`).
    pub(crate) fn set_frame1(&mut self, frame: usize, time: f32, value: f32) {
        if let Some([t, v, ..]) = self.frame_mut(frame) {
            *t = time;
            *v = value;
        }
    }

    /// Set a two-value frame (`entries == 3`).
    pub(crate) fn set_frame2(&mut self, frame: usize, time: f32, value1: f32, value2: f32) {
        if let Some([t, v1, v2, ..]) = self.frame_mut(frame) {
            *t = time;
            *v1 = value1;
            *v2 = value2;
        }
    }

    /// Mark `frame` as stepped interpolation.
    pub(crate) fn set_stepped(&mut self, frame: usize) {
        if frame < self.frame_count() {
            if let Some(curve) = self.curves.get_mut(frame) {
                *curve = STEPPED;
            }
        }
    }

    /// Store one Bezier segment table for `frame`'s value ordinal `value`
    /// (0-based), as forward-difference sampling of the cubic. A frame or
    /// segment index past the curve's storage is skipped, leaving the frame's
    /// interpolation unchanged.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn set_bezier(
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
        let frame_count = self.frame_count();
        if frame >= frame_count {
            return;
        }
        let Some(i0) = bezier
            .checked_mul(BEZIER_SIZE)
            .and_then(|n| n.checked_add(frame_count))
        else {
            return;
        };
        // The curve type stores the table start as an `f32`. A start that does
        // not survive that round trip (possible past 2^24) would read back as
        // a different table, so the segment is skipped.
        let Some(start) = i0.checked_add(BEZIER) else {
            return;
        };
        let curve_type = start as f32;
        if curve_type as usize != start {
            return;
        }
        let Some(table) = i0
            .checked_add(BEZIER_SIZE)
            .and_then(|end| self.curves.get_mut(i0..end))
            .and_then(|s| <&mut BezierTable>::try_from(s).ok())
        else {
            return;
        };
        let tmpx = (time1 - cx1 * 2.0 + cx2) * 0.03;
        let tmpy = (value1 - cy1 * 2.0 + cy2) * 0.03;
        let dddx = ((cx1 - cx2) * 3.0 - time1 + time2) * 0.006;
        let dddy = ((cy1 - cy2) * 3.0 - value1 + value2) * 0.006;
        let mut ddx = tmpx * 2.0 + dddx;
        let mut ddy = tmpy * 2.0 + dddy;
        let mut dx = (cx1 - time1) * 0.3 + tmpx + dddx * 0.166_666_67;
        let mut dy = (cy1 - value1) * 0.3 + tmpy + dddy * 0.166_666_67;
        let mut x = time1 + dx;
        let mut y = value1 + dy;
        for point in table.chunks_mut(2) {
            if let [px, py] = point {
                *px = x;
                *py = y;
            }
            dx += ddx;
            dy += ddy;
            ddx += dddx;
            ddy += dddy;
            x += dx;
            y += dy;
        }
        if value == 0 {
            if let Some(curve) = self.curves.get_mut(frame) {
                *curve = curve_type;
            }
        }
    }

    /// The interpolation type of the frame starting at float index `i`. A
    /// missing entry reads as STEPPED, which holds the frame's value.
    fn curve_type(&self, i: usize) -> f32 {
        i.checked_div(self.entries)
            .and_then(|frame| self.curves.get(frame))
            .copied()
            .unwrap_or(STEPPED)
    }

    /// The Bezier table starting at `seg` in `curves`, if it fits.
    fn bezier_table(&self, seg: usize) -> Option<&BezierTable> {
        let end = seg.checked_add(BEZIER_SIZE)?;
        self.curves.get(seg..end)?.try_into().ok()
    }

    /// Interpolated value at `time` for the value at `value_offset` (1 or 2).
    /// `time` must be `>= first_time()`. A curve with no frames reads as `0`.
    pub(crate) fn value(&self, time: f32, value_offset: usize) -> f32 {
        let entries = self.entries;
        let i = search(&self.frames, time, entries);
        let (Some(&before), Some(&v)) = (
            self.frames.get(i),
            self.frames.get(i.saturating_add(value_offset)),
        ) else {
            return 0.0;
        };
        let curve_type = self.curve_type(i);
        if curve_type == STEPPED {
            return v;
        }
        // The last frame has no next frame to interpolate toward, so it holds.
        let next = i.saturating_add(entries);
        let (Some(&next_time), Some(&next_v)) = (
            self.frames.get(next),
            self.frames.get(next.saturating_add(value_offset)),
        ) else {
            return v;
        };
        let linear = |v: f32| v + (time - before) / (next_time - before) * (next_v - v);
        if curve_type == LINEAR {
            return linear(v);
        }
        let table = (curve_type as usize)
            .checked_sub(BEZIER)
            .zip(value_offset.checked_sub(1))
            .and_then(|(seg, ordinal)| seg.checked_add(ordinal.checked_mul(BEZIER_SIZE)?))
            .and_then(|seg| self.bezier_table(seg));
        match table {
            Some(table) => bezier_value(table, time, before, v, next_time, next_v),
            None => linear(v),
        }
    }

    /// The interpolation percent (0..1) within the segment containing `time`,
    /// respecting the segment's curve type (stepped / linear / Bezier). Drives
    /// lerps of an external value array (e.g. mesh deform), where the Bezier `y`
    /// samples encode the percent. `time` must be `>= first_time()` and before
    /// the last frame time. At the last frame it reads as stepped (`0`).
    /// Mirrors Spine's `CurveTimeline.getCurvePercent`.
    pub(crate) fn percent(&self, time: f32) -> f32 {
        let entries = self.entries;
        let i = search(&self.frames, time, entries);
        let curve_type = self.curve_type(i);
        let (Some(&x), Some(&next_x)) = (
            self.frames.get(i),
            self.frames.get(i.saturating_add(entries)),
        ) else {
            return 0.0;
        };
        if curve_type == LINEAR {
            (time - x) / (next_x - x)
        } else if curve_type == STEPPED {
            0.0
        } else {
            match (curve_type as usize)
                .checked_sub(BEZIER)
                .and_then(|seg| self.bezier_table(seg))
            {
                Some(table) => bezier_percent(table, time, x, next_x),
                None => (time - x) / (next_x - x),
            }
        }
    }

    /// Value for a property whose timeline value is **added** to the setup value
    /// (rotate / translate / shear). Spine `getRelativeValue`. A curve with no
    /// frames leaves `current` unchanged.
    pub(crate) fn relative_value(
        &self,
        time: f32,
        alpha: f32,
        from: MixFrom,
        add: bool,
        current: f32,
        setup: f32,
    ) -> f32 {
        let Some(first) = self.first_time() else {
            return current;
        };
        if time < first {
            return before_first_key(from, alpha, current, setup);
        }
        let value = self.value(time, 1);
        match from {
            MixFrom::Setup => setup + value * alpha,
            _ => current + (if add { value } else { value + setup - current }) * alpha,
        }
    }

    /// Value for an absolute property (replaces the setup value rather than
    /// adding to it). Spine `getAbsoluteValue`. A curve with no frames leaves
    /// `current` unchanged.
    pub(crate) fn absolute_value(
        &self,
        time: f32,
        alpha: f32,
        from: MixFrom,
        add: bool,
        current: f32,
        setup: f32,
    ) -> f32 {
        let Some(first) = self.first_time() else {
            return current;
        };
        if time < first {
            return before_first_key(from, alpha, current, setup);
        }
        absolute_value_with(self.value(time, 1), alpha, from, add, current, setup)
    }

    /// Set a frame with `time` followed by one value per channel. Values past
    /// the frame's channel count are ignored.
    pub(crate) fn set_frame_n(&mut self, frame: usize, time: f32, values: &[f32]) {
        let Some((t, channels)) = self.frame_mut(frame).and_then(<[f32]>::split_first_mut) else {
            return;
        };
        *t = time;
        for (dst, v) in channels.iter_mut().zip(values) {
            *dst = *v;
        }
    }

    /// The raw (stepped) value at `value_offset` in the frame containing `time`.
    /// A value outside the frames reads as `0`.
    pub(crate) fn frame_value(&self, time: f32, value_offset: usize) -> f32 {
        let i = search(&self.frames, time, self.entries);
        self.frames
            .get(i.saturating_add(value_offset))
            .copied()
            .unwrap_or(0.0)
    }
}

/// The `(x, y)` sample points of a Bezier table, in order.
fn sample_points(table: &BezierTable) -> impl Iterator<Item = (f32, f32)> + '_ {
    let xs = table.iter().step_by(2).copied();
    let ys = table.iter().skip(1).step_by(2).copied();
    xs.zip(ys)
}

/// Linear interpolation across a sampled Bezier table that runs from the frame
/// point `(frame_x, frame_y)` to the next frame point `(next_x, next_y)`.
/// Transcribed from Spine's getBezierValue.
fn bezier_value(
    table: &BezierTable,
    time: f32,
    frame_x: f32,
    frame_y: f32,
    next_x: f32,
    next_y: f32,
) -> f32 {
    let (mut x, mut y) = (table[0], table[1]);
    if x > time {
        return frame_y + (time - frame_x) / (x - frame_x) * (y - frame_y);
    }
    for (px, py) in sample_points(table).skip(1) {
        if px >= time {
            return y + (time - x) / (px - x) * (py - y);
        }
        x = px;
        y = py;
    }
    y + (time - x) / (next_x - x) * (next_y - y)
}

/// Sample the Bezier percent table for one segment. The packed pairs are
/// `(time, percent)`. Before and after the table it lerps to the segment's
/// endpoints (percent 0 at the start frame, 1 at the next). Mirrors the
/// percent branch of Spine's `getCurvePercent`.
fn bezier_percent(table: &BezierTable, time: f32, frame_x: f32, next_x: f32) -> f32 {
    let (mut x, mut y) = (table[0], table[1]);
    if x > time {
        return y * (time - frame_x) / (x - frame_x);
    }
    for (px, py) in sample_points(table).skip(1) {
        if px >= time {
            return y + (time - x) / (px - x) * (py - y);
        }
        x = px;
        y = py;
    }
    y + (1.0 - y) * (time - x) / (next_x - x)
}

/// First frame start index (stride `step`) whose time is `<= time`. `time` must
/// be `>= frames[0]`. Spine `Timeline.search`. Returns `0` when there are no
/// frames.
fn search(frames: &[f32], time: f32, step: usize) -> usize {
    if step == 0 {
        return 0;
    }
    let n = frames.len();
    let mut i = step;
    while i < n {
        if frames[i] > time {
            return i - step;
        }
        i += step;
    }
    n.saturating_sub(step)
}

/// Value before the first keyframe, per mix source. Spine `beforeFirstKey`.
fn before_first_key(from: MixFrom, alpha: f32, current: f32, setup: f32) -> f32 {
    match from {
        MixFrom::Setup => setup,
        MixFrom::First => current + (setup - current) * alpha,
        MixFrom::Current => current,
    }
}

/// Mix a precomputed timeline `value` into an absolute property (the value
/// replaces, rather than adds to, the base). Spine's `getAbsoluteValue`
/// value-overload, used by the multi-channel constraint timelines.
pub(crate) fn absolute_value_with(
    value: f32,
    alpha: f32,
    from: MixFrom,
    add: bool,
    current: f32,
    setup: f32,
) -> f32 {
    let base = if matches!(from, MixFrom::Setup) {
        setup
    } else {
        current
    };
    if add {
        base + value * alpha
    } else {
        base + (value - base) * alpha
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_interpolates_between_frames() {
        let mut c = Curve::new(2, 0, 2);
        c.set_frame1(0, 0.0, 0.0);
        c.set_frame1(1, 1.0, 90.0);
        assert!((c.value(0.0, 1) - 0.0).abs() < 1e-4);
        assert!((c.value(0.5, 1) - 45.0).abs() < 1e-4);
        // Past the last frame clamps to its value (forced STEPPED).
        assert!((c.value(2.0, 1) - 90.0).abs() < 1e-4);
    }

    #[test]
    fn stepped_holds_the_earlier_value() {
        let mut c = Curve::new(2, 0, 2);
        c.set_frame1(0, 0.0, 10.0);
        c.set_frame1(1, 1.0, 90.0);
        c.set_stepped(0);
        assert!((c.value(0.99, 1) - 10.0).abs() < 1e-4);
    }

    #[test]
    fn bezier_symmetric_curve_passes_through_midpoint() {
        let mut c = Curve::new(2, 1, 2);
        c.set_frame1(0, 0.0, 0.0);
        c.set_frame1(1, 1.0, 90.0);
        // Symmetric S-curve about (0.5, 45).
        c.set_bezier(0, 0, 0, 0.0, 0.0, 0.25, 0.0, 0.75, 90.0, 1.0, 90.0);
        assert!((c.value(0.0, 1) - 0.0).abs() < 0.5);
        assert!((c.value(0.5, 1) - 45.0).abs() < 0.5);
        assert!((c.value(1.0, 1) - 90.0).abs() < 0.5);
    }

    #[test]
    fn two_value_curve_interpolates_each_channel() {
        let mut c = Curve::new(2, 0, 3);
        c.set_frame2(0, 0.0, 0.0, 0.0);
        c.set_frame2(1, 1.0, 10.0, 20.0);
        assert!((c.value(0.5, 1) - 5.0).abs() < 1e-4);
        assert!((c.value(0.5, 2) - 10.0).abs() < 1e-4);
    }

    #[test]
    fn percent_respects_linear_and_stepped() {
        let mut c = Curve::new(2, 0, 2);
        c.set_frame1(0, 0.0, 0.0);
        c.set_frame1(1, 2.0, 0.0);
        // Linear: halfway through the segment in time -> 0.5.
        assert!((c.percent(1.0) - 0.5).abs() < 1e-4);
        // Stepped: 0 until the next frame.
        c.set_stepped(0);
        assert!((c.percent(1.0)).abs() < 1e-4);
        assert!((c.percent(1.9)).abs() < 1e-4);
    }

    #[test]
    fn percent_bezier_passes_through_midpoint() {
        let mut c = Curve::new(2, 1, 2);
        c.set_frame1(0, 0.0, 0.0);
        c.set_frame1(1, 1.0, 0.0);
        // Symmetric percent S-curve from 0 to 1 about (0.5, 0.5).
        c.set_bezier(0, 0, 0, 0.0, 0.0, 0.25, 0.0, 0.75, 1.0, 1.0, 1.0);
        assert!((c.percent(0.5) - 0.5).abs() < 0.05);
    }

    #[test]
    fn declared_bezier_count_is_capped_by_the_frames() {
        // Two one-value frames can use at most two segments.
        let mut c = Curve::new(2, 1_000_000, 2);
        assert_eq!(c.curves.len(), 2 + 2 * BEZIER_SIZE);
        c.set_frame1(0, 0.0, 0.0);
        c.set_frame1(1, 1.0, 90.0);
        c.set_bezier(0, 0, 0, 0.0, 0.0, 0.25, 0.0, 0.75, 90.0, 1.0, 90.0);
        assert!((c.value(0.5, 1) - 45.0).abs() < 0.5);
    }
}
