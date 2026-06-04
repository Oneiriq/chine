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

pub mod ik;
pub mod path;
pub mod physics;
pub mod slider;
pub mod transform;

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
