//! Constraints: solvers that reposition bones after forward kinematics.
//!
//! Constraints run inside [`crate::skel::Skeleton::update_world_transform`],
//! after the FK pass and in their defined order. Each adjusts the *local* pose
//! of its constrained bones; those bones are then re-solved by FK. Because the
//! local pose is modified in place, the host must reset bones to their setup or
//! animated pose each frame before posing (Spine separates an animated pose from
//! an applied pose; chine collapses them by re-applying the pose every frame).
//!
//! Each constraint kind lives in its own submodule: [`ik`], [`transform`],
//! [`path`], [`physics`], and [`slider`].

pub(crate) mod ik;
pub(crate) mod path;
pub(crate) mod physics;
pub(crate) mod slider;
pub(crate) mod transform;

/// How a constraint adjusts Y scale when stretching or compressing
/// (Spine 4.3 `ScaleYMode`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ScaleYMode {
    /// Leave Y scale unchanged.
    #[default]
    None,
    /// Scale Y by the same factor as X.
    Uniform,
    /// Adjust Y scale to preserve area.
    Volume,
}

/// Clamp `value` to `[min, max]`. For ordered bounds this matches `f32::clamp`
/// exactly. Unlike `f32::clamp` it does not panic when a bound is NaN or
/// `min > max`, which constraint data read from a file can produce.
fn clamp(value: f32, min: f32, max: f32) -> f32 {
    let mut value = value;
    if value < min {
        value = min;
    }
    if value > max {
        value = max;
    }
    value
}

#[cfg(test)]
mod tests {
    use super::clamp;

    #[test]
    fn clamp_matches_std_for_ordered_bounds() {
        for &(v, lo, hi) in &[
            (5.0_f32, 0.0_f32, 1.0_f32),
            (-5.0, 0.0, 1.0),
            (0.5, 0.0, 1.0),
            (f32::NAN, 0.0, 1.0),
            (f32::INFINITY, -1.0, 1.0),
            (2.0, 2.0, 2.0),
        ] {
            let ours = clamp(v, lo, hi);
            let std = v.clamp(lo, hi);
            assert!(ours.to_bits() == std.to_bits() || (ours.is_nan() && std.is_nan()));
        }
    }

    #[test]
    fn clamp_tolerates_nan_and_reversed_bounds() {
        assert_eq!(clamp(5.0, f32::NAN, 1.0), 1.0);
        assert_eq!(clamp(5.0, 0.0, f32::NAN), 5.0);
        assert_eq!(clamp(0.5, 1.0, -1.0), -1.0);
    }
}
