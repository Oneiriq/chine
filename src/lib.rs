//! `chine` — a pure-Rust Spine 4.3 skeletal animation runtime.
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
//! ```text
//! load (json / binary + atlas)  ->  SkeletonData   (immutable rig)
//!   instantiate                 ->  Skeleton       (posable instance)
//!   per frame:
//!     AnimationState::update + apply  ->  Skeleton local pose
//!     Skeleton::update_world_transform ->  world pose (+ constraints, physics)
//!     RenderCommand stream             ->  host renderer
//! ```
//!
//! The world transform of each bone is a 2x2 matrix `(a, b, c, d)` plus a world
//! position `(world_x, world_y)`, computed root-to-children — the same model the
//! official runtimes use.
#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod data;
pub mod skel;
