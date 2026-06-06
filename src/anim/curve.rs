//! Keyframe curves: stepped / linear / Bezier interpolation.
//!
//! [`Curve`] re-implements Spine 4.3's `CurveTimeline` storage and evaluation,
//! so interpolated values match: keyframe `frames` interleave a time with one or
//! two values, and a parallel `curves` array holds either a per-frame
//! interpolation type (`LINEAR` / `STEPPED`) or an index into
//! forward-difference-sampled Bezier segments.

use super::MixFrom;

/// Curve-type sentinels stored in `curves`. Types `>= BEZIER` index the packed
/// Bezier segments.
const LINEAR: f32 = 0.0;
const STEPPED: f32 = 1.0;
const BEZIER: usize = 2;
/// Floats stored per Bezier curve (10 sampled points, x/y interleaved, less 2).
const BEZIER_SIZE: usize = 18;

/// Shared keyframe storage with stepped / linear / Bezier interpolation.
#[derive(Debug, Clone)]
pub(crate) struct Curve {
    /// Interleaved `[time, v1, (v2)]` per frame; `entries` floats per frame.
    frames: Vec<f32>,
    /// Per-frame interpolation type, followed by packed Bezier segments.
    curves: Vec<f32>,
    /// Floats per frame: 2 for one-value timelines, 3 for two-value.
    entries: usize,
}

impl Curve {
    /// Storage for `frame_count` frames (`entries` floats each) and up to
    /// `bezier_count` Bezier curves.
    pub(crate) fn new(frame_count: usize, bezier_count: usize, entries: usize) -> Self {
        let mut curves = vec![0.0; frame_count + bezier_count * BEZIER_SIZE];
        // The final frame has no following frame; STEPPED keeps reads in bounds.
        curves[frame_count - 1] = STEPPED;
        Self {
            frames: vec![0.0; frame_count * entries],
            curves,
            entries,
        }
    }

    fn frame_count(&self) -> usize {
        self.frames.len() / self.entries
    }

    /// The time of the first keyframe.
    pub(crate) fn first_time(&self) -> f32 {
        self.frames[0]
    }

    /// Set a one-value frame (`entries == 2`).
    pub(crate) fn set_frame1(&mut self, frame: usize, time: f32, value: f32) {
        let i = frame * self.entries;
        self.frames[i] = time;
        self.frames[i + 1] = value;
    }

    /// Set a two-value frame (`entries == 3`).
    pub(crate) fn set_frame2(&mut self, frame: usize, time: f32, value1: f32, value2: f32) {
        let i = frame * self.entries;
        self.frames[i] = time;
        self.frames[i + 1] = value1;
        self.frames[i + 2] = value2;
    }

    /// Mark `frame` as stepped interpolation.
    pub(crate) fn set_stepped(&mut self, frame: usize) {
        self.curves[frame] = STEPPED;
    }

    /// Store one Bezier segment table for `frame`'s value ordinal `value`
    /// (0-based), as forward-difference sampling of the cubic.
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
        let i0 = self.frame_count() + bezier * BEZIER_SIZE;
        if value == 0 {
            self.curves[frame] = (BEZIER + i0) as f32;
        }
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
        let n = i0 + BEZIER_SIZE;
        let mut i = i0;
        while i < n {
            self.curves[i] = x;
            self.curves[i + 1] = y;
            dx += ddx;
            dy += ddy;
            ddx += dddx;
            ddy += dddy;
            x += dx;
            y += dy;
            i += 2;
        }
    }

    /// Interpolated value at `time` for the value at `value_offset` (1 or 2).
    /// `time` must be `>= first_time()`.
    pub(crate) fn value(&self, time: f32, value_offset: usize) -> f32 {
        let entries = self.entries;
        let i = search(&self.frames, time, entries);
        let curve_type = self.curves[i / entries];
        if curve_type == LINEAR {
            let before = self.frames[i];
            let v = self.frames[i + value_offset];
            v + (time - before) / (self.frames[i + entries] - before)
                * (self.frames[i + entries + value_offset] - v)
        } else if curve_type == STEPPED {
            self.frames[i + value_offset]
        } else {
            let seg = curve_type as usize - BEZIER + (value_offset - 1) * BEZIER_SIZE;
            self.bezier_value(time, i, value_offset, seg)
        }
    }

    /// Linear interpolation across a sampled Bezier table. Transcribed from
    /// Spine's getBezierValue.
    fn bezier_value(&self, time: f32, frame_index: usize, value_offset: usize, seg: usize) -> f32 {
        let curves = &self.curves;
        if curves[seg] > time {
            let x = self.frames[frame_index];
            let y = self.frames[frame_index + value_offset];
            return y + (time - x) / (curves[seg] - x) * (curves[seg + 1] - y);
        }
        let n = seg + BEZIER_SIZE;
        let mut i = seg + 2;
        while i < n {
            if curves[i] >= time {
                let x = curves[i - 2];
                let y = curves[i - 1];
                return y + (time - x) / (curves[i] - x) * (curves[i + 1] - y);
            }
            i += 2;
        }
        let fi = frame_index + self.entries;
        let x = curves[n - 2];
        let y = curves[n - 1];
        y + (time - x) / (self.frames[fi] - x) * (self.frames[fi + value_offset] - y)
    }

    /// The interpolation percent (0..1) within the segment containing `time`,
    /// respecting the segment's curve type (stepped / linear / Bezier). Drives
    /// lerps of an external value array (e.g. mesh deform), where the Bezier `y`
    /// samples encode the percent. `time` must be `>= first_time()` and before
    /// the last frame time. Mirrors Spine's `CurveTimeline.getCurvePercent`.
    pub(crate) fn percent(&self, time: f32) -> f32 {
        let entries = self.entries;
        let i = search(&self.frames, time, entries);
        let frame = i / entries;
        let curve_type = self.curves[frame];
        let x = self.frames[i];
        let next_x = self.frames[i + entries];
        if curve_type == LINEAR {
            (time - x) / (next_x - x)
        } else if curve_type == STEPPED {
            0.0
        } else {
            self.bezier_percent(time, x, next_x, curve_type as usize - BEZIER)
        }
    }

    /// Sample the Bezier percent table for one segment. The packed pairs are
    /// `(time, percent)`; before and after the table it lerps to the segment's
    /// endpoints (percent 0 at the start frame, 1 at the next). Mirrors the
    /// percent branch of Spine's `getCurvePercent`.
    fn bezier_percent(&self, time: f32, frame_x: f32, next_x: f32, seg: usize) -> f32 {
        let curves = &self.curves;
        if curves[seg] > time {
            return curves[seg + 1] * (time - frame_x) / (curves[seg] - frame_x);
        }
        let n = seg + BEZIER_SIZE;
        let mut i = seg + 2;
        while i < n {
            if curves[i] >= time {
                let x = curves[i - 2];
                let y = curves[i - 1];
                return y + (time - x) / (curves[i] - x) * (curves[i + 1] - y);
            }
            i += 2;
        }
        let x = curves[n - 2];
        let y = curves[n - 1];
        y + (1.0 - y) * (time - x) / (next_x - x)
    }

    /// Value for a property whose timeline value is **added** to the setup value
    /// (rotate / translate / shear). Spine `getRelativeValue`.
    pub(crate) fn relative_value(
        &self,
        time: f32,
        alpha: f32,
        from: MixFrom,
        add: bool,
        current: f32,
        setup: f32,
    ) -> f32 {
        if time < self.frames[0] {
            return before_first_key(from, alpha, current, setup);
        }
        let value = self.value(time, 1);
        match from {
            MixFrom::Setup => setup + value * alpha,
            _ => current + (if add { value } else { value + setup - current }) * alpha,
        }
    }

    /// Value for an absolute property (replaces the setup value rather than
    /// adding to it). Spine `getAbsoluteValue`.
    pub(crate) fn absolute_value(
        &self,
        time: f32,
        alpha: f32,
        from: MixFrom,
        add: bool,
        current: f32,
        setup: f32,
    ) -> f32 {
        if time < self.frames[0] {
            return before_first_key(from, alpha, current, setup);
        }
        absolute_value_with(self.value(time, 1), alpha, from, add, current, setup)
    }

    /// Set a frame with `time` followed by one value per channel.
    pub(crate) fn set_frame_n(&mut self, frame: usize, time: f32, values: &[f32]) {
        let i = frame * self.entries;
        self.frames[i] = time;
        for (k, v) in values.iter().enumerate() {
            self.frames[i + 1 + k] = *v;
        }
    }

    /// The raw (stepped) value at `value_offset` in the frame containing `time`.
    pub(crate) fn frame_value(&self, time: f32, value_offset: usize) -> f32 {
        let i = search(&self.frames, time, self.entries);
        self.frames[i + value_offset]
    }
}

/// First frame start index (stride `step`) whose time is `<= time`. `time` must
/// be `>= frames[0]`. Spine `Timeline.search`.
fn search(frames: &[f32], time: f32, step: usize) -> usize {
    let n = frames.len();
    let mut i = step;
    while i < n {
        if frames[i] > time {
            return i - step;
        }
        i += step;
    }
    n - step
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
}
