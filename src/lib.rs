//! `chine`: a pure-Rust Spine 4.3 skeletal animation runtime.
//!
//! `chine` loads Spine skeleton exports (JSON / binary `.skel`) together with an
//! atlas, poses and animates a skeleton, and emits renderer-agnostic draw data,
//! leaving all GPU work to the host engine.
//!
//! It is a from-scratch reimplementation that references the official Spine 4.3
//! runtimes, and is **not** affiliated with or endorsed by Esoteric Software.
//! Using Spine skeleton data requires a valid Spine Editor license.
//!
//! # Pipeline
//!
//! 1. Load a JSON or binary `.skel` export plus an atlas into a
//!    `SkeletonData`: the immutable, shareable rig.
//! 2. Instantiate a `Skeleton` from it: a posable instance.
//! 3. Each frame, `AnimationState::update` and `apply` drive the skeleton's
//!    local pose, `Skeleton::update_world_transform` computes the world pose
//!    (forward kinematics plus IK, transform, path, physics, and slider
//!    constraints), and the `RenderCommand` stream feeds the host renderer.
//!
//! The world transform of each bone is a 2x2 matrix `(a, b, c, d)` plus a world
//! position `(world_x, world_y)`, computed root-to-children, the same model the
//! official runtimes use.
#![warn(missing_docs)]
#![warn(unreachable_pub)]
#![warn(clippy::all)]
// Without a loader feature, the loader-fed timeline variants, channel tables,
// and data constructors are unused by design: the crate is then a manual
// pose/render runtime over a hand-built SkeletonData. Allow that dead code only
// in the no-loader configuration; the loader builds still enforce every warning.
#![cfg_attr(
    not(any(feature = "json", feature = "binary")),
    allow(dead_code, unused_imports)
)]

pub mod anim;
pub mod atlas;
pub(crate) mod attach;
#[cfg(feature = "binary")]
pub mod binary;
pub(crate) mod clip;
pub(crate) mod constraint;
pub mod data;
pub(crate) mod event;
pub(crate) mod link;
#[cfg(feature = "json")]
pub mod load;
pub mod render;
pub mod skel;
pub(crate) mod skin;
