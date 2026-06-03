//! `chine`: a pure-Rust Spine 4.3 skeletal animation runtime.
//!
//! `chine` loads Spine skeleton exports (JSON / binary `.skel`) together with an
//! atlas, poses and animates a skeleton, and emits renderer-agnostic draw data,
//! leaving all GPU work to the host engine.
//!
//! It is a clean-room reimplementation and is **not** affiliated with or
//! endorsed by Esoteric Software. Using Spine skeleton data requires a valid
//! Spine Editor license.
//!
//! # Pipeline
//!
//! 1. Load a JSON (or, later, binary `.skel`) export plus an atlas into a
//!    `SkeletonData`: the immutable, shareable rig.
//! 2. Instantiate a `Skeleton` from it: a posable instance.
//! 3. Each frame, `AnimationState::update` and `apply` drive the skeleton's
//!    local pose, `Skeleton::update_world_transform` computes the world pose
//!    (forward kinematics plus constraints, and later physics), and the
//!    `RenderCommand` stream feeds the host renderer.
//!
//! The world transform of each bone is a 2x2 matrix `(a, b, c, d)` plus a world
//! position `(world_x, world_y)`, computed root-to-children, the same model the
//! official runtimes use.
#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod anim;
pub mod atlas;
pub mod attach;
pub mod constraint;
pub mod data;
#[cfg(feature = "json")]
pub mod load;
pub mod render;
pub mod skel;
pub mod skin;
