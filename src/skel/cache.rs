//! The update cache: the order in which a [`super::Skeleton`] computes bone
//! world transforms and applies its constraints.
//!
//! The cache is built once from the rig data. The data may be malformed (a
//! binary export stores bone parents and constraint targets as raw indices, and
//! a host can build a `SkeletonData` by hand), so every walk here tolerates
//! out-of-range indices and cyclic parents: a bad reference is skipped, a
//! parent cycle is cut where it closes, and no walk recurses.

use crate::constraint::ik::IkConstraintData;
use crate::constraint::path::PathConstraintData;
use crate::constraint::physics::PhysicsConstraintData;
use crate::constraint::slider::SliderData;
use crate::constraint::transform::TransformConstraintData;
use crate::data::SkeletonData;

/// The cache length past which no further constraint is sorted in.
///
/// Each constraint can re-sort a whole bone chain, so a hostile rig with many
/// constraints over a deep hierarchy would otherwise build a cache (and do
/// per-frame work) that grows with bones times constraints. Real rigs stay
/// orders of magnitude below this. Constraints past the limit are not applied.
pub(super) const MAX_CONSTRAINT_CACHE: usize = 1 << 20;

/// One entry in a skeleton's update cache: compute a bone's world transform,
/// or apply a constraint. Built by [`build_update_cache`] in dependency order.
#[derive(Debug, Clone, Copy)]
pub(super) enum Updatable {
    /// Compute the world transform for the bone at this index.
    Bone(usize),
    /// Apply the IK constraint at this index.
    Ik(usize),
    /// Apply the transform constraint at this index.
    Transform(usize),
    /// Apply the path constraint at this index.
    Path(usize),
    /// Apply the physics constraint at this index.
    Physics(usize),
    /// Apply the slider constraint at this index.
    Slider(usize),
}

/// Build the ordered update cache: a topological interleaving of bone
/// world-transform updates and constraint applications, mirroring Spine's
/// `Skeleton.updateCache`. Bones a constraint reads are computed before it,
/// and bones it modifies are recomputed after.
///
/// Every `Bone` entry indexes `data.bones`, and every constraint entry indexes
/// its constraint list. An IK, transform, path, or physics constraint whose
/// bone or slot references fall outside the rig is left out, so those solvers
/// only see valid indices. A slider is kept either way: it reads its bone
/// through a checked lookup and does nothing when the bone is missing.
pub(super) fn build_update_cache(data: &SkeletonData) -> Vec<Updatable> {
    let n = data.bones.len();
    let parents: Vec<Option<usize>> = data
        .bones
        .iter()
        .map(|b| valid_parent(b.parent, n))
        .collect();
    let mut children: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (i, parent) in parents.iter().enumerate() {
        if let Some(kids) = parent.and_then(|p| children.get_mut(p)) {
            kids.push(i);
        }
    }

    // Merge constraints of all kinds and process them in global order.
    let mut ordered: Vec<(usize, Updatable)> = Vec::new();
    for (i, ik) in data.ik_constraints.iter().enumerate() {
        ordered.push((ik.order, Updatable::Ik(i)));
    }
    for (i, tc) in data.transform_constraints.iter().enumerate() {
        ordered.push((tc.order, Updatable::Transform(i)));
    }
    for (i, pc) in data.path_constraints.iter().enumerate() {
        ordered.push((pc.order, Updatable::Path(i)));
    }
    for (i, pc) in data.physics_constraints.iter().enumerate() {
        ordered.push((pc.order, Updatable::Physics(i)));
    }
    for (i, sl) in data.sliders.iter().enumerate() {
        ordered.push((sl.order, Updatable::Slider(i)));
    }
    ordered.sort_by_key(|(order, _)| *order);

    let mut sorted = vec![false; n];
    let mut cache = Vec::new();
    let mut walk = Walk {
        parents: &parents,
        children: &children,
        sorted: &mut sorted,
        cache: &mut cache,
    };
    for (_, kind) in ordered {
        if walk.cache.len() >= MAX_CONSTRAINT_CACHE {
            break;
        }
        match kind {
            Updatable::Ik(i) => {
                if let Some(ik) = data.ik_constraints.get(i) {
                    if ik.target < n && all_in_range(&ik.bones, n) {
                        walk.sort_ik(ik, i);
                    }
                }
            }
            Updatable::Transform(i) => {
                if let Some(tc) = data.transform_constraints.get(i) {
                    if tc.source < n && all_in_range(&tc.bones, n) {
                        walk.sort_transform(tc, i);
                    }
                }
            }
            Updatable::Path(i) => {
                if let Some(pc) = data.path_constraints.get(i) {
                    let slot_bone = data.slots.get(pc.slot).map(|s| s.bone);
                    if let Some(slot_bone) = slot_bone.filter(|&b| b < n) {
                        if all_in_range(&pc.bones, n) {
                            walk.sort_path(pc, i, slot_bone);
                        }
                    }
                }
            }
            Updatable::Physics(i) => {
                if let Some(pd) = data.physics_constraints.get(i) {
                    if pd.bone < n {
                        walk.sort_physics(pd, i);
                    }
                }
            }
            Updatable::Slider(i) => {
                if let Some(slider) = data.sliders.get(i) {
                    walk.sort_slider(slider, i);
                }
            }
            Updatable::Bone(_) => {}
        }
    }
    for i in 0..n {
        walk.sort_bone(i);
    }
    cache
}

/// `parent` if it names a bone of a rig with `bone_count` bones, else `None`
/// (an out-of-range parent is treated as no parent).
pub(super) fn valid_parent(parent: Option<usize>, bone_count: usize) -> Option<usize> {
    parent.filter(|&p| p < bone_count)
}

/// Whether every index in `indices` is below `len`.
fn all_in_range(indices: &[usize], len: usize) -> bool {
    indices.iter().all(|&i| i < len)
}

/// The bone hierarchy and the cache under construction, shared by the sort
/// steps.
struct Walk<'a> {
    parents: &'a [Option<usize>],
    children: &'a [Vec<usize>],
    sorted: &'a mut [bool],
    cache: &'a mut Vec<Updatable>,
}

impl Walk<'_> {
    /// Set a bone's sorted flag, ignoring an out-of-range index.
    fn set_sorted(&mut self, bone: usize, value: bool) {
        if let Some(flag) = self.sorted.get_mut(bone) {
            *flag = value;
        }
    }

    /// Sort an IK constraint into the cache: its target and constrained bones
    /// are computed before it, the constrained bones recomputed after.
    fn sort_ik(&mut self, ik: &IkConstraintData, idx: usize) {
        let Some(&parent) = ik.bones.first() else {
            return;
        };
        self.sort_bone(ik.target);
        self.sort_bone(parent);
        self.cache.push(Updatable::Ik(idx));
        self.set_sorted(parent, false);
        self.sort_reset(parent);
    }

    /// Sort a transform constraint into the cache. For world targets the
    /// constrained bones are computed before it and kept (their world is the
    /// constraint's output), and their descendants are recomputed. For local
    /// targets the constrained bones themselves are recomputed afterward.
    fn sort_transform(&mut self, tc: &TransformConstraintData, idx: usize) {
        if !tc.local_source {
            self.sort_bone(tc.source);
        }
        let world_target = !tc.local_target;
        if world_target {
            for &b in &tc.bones {
                self.sort_bone(b);
            }
        }
        self.cache.push(Updatable::Transform(idx));
        for &b in &tc.bones {
            self.sort_reset(b);
        }
        for &b in &tc.bones {
            self.set_sorted(b, world_target);
        }
    }

    /// Sort a path constraint into the cache. The slot bone and constrained
    /// bones are computed before it. The constrained bones keep their world
    /// (the constraint's output) and their descendants are recomputed.
    fn sort_path(&mut self, pc: &PathConstraintData, idx: usize, slot_bone: usize) {
        self.sort_bone(slot_bone);
        for &b in &pc.bones {
            self.sort_bone(b);
        }
        self.cache.push(Updatable::Path(idx));
        for &b in &pc.bones {
            self.sort_reset(b);
        }
        for &b in &pc.bones {
            self.set_sorted(b, true);
        }
    }

    /// Sort a physics constraint into the cache. Its bone is computed before
    /// it. The constraint then writes that bone's world transform, so the bone
    /// keeps its computed slot (the constraint's output) and only its
    /// descendants are recomputed afterward.
    fn sort_physics(&mut self, pd: &PhysicsConstraintData, idx: usize) {
        self.sort_bone(pd.bone);
        self.cache.push(Updatable::Physics(idx));
        self.sort_reset(pd.bone);
    }

    /// Sort a slider into the cache: its source bone is computed before it so
    /// its property can be read. The animation it scrubs is applied when the
    /// slider runs.
    fn sort_slider(&mut self, slider: &SliderData, idx: usize) {
        if let Some(bone) = slider.bone {
            self.sort_bone(bone);
        }
        self.cache.push(Updatable::Slider(idx));
    }

    /// Add `bone` and any unsorted ancestors to the cache once, parents first.
    ///
    /// The walk climbs the parent chain, marking each unsorted bone as it goes,
    /// and stops at a sorted bone, a root, or an out-of-range index. Marking
    /// first means a parent cycle ends the walk where it closes. The bones are
    /// appended child first, so the appended run is reversed to put parents
    /// first.
    fn sort_bone(&mut self, bone: usize) {
        let start = self.cache.len();
        let mut current = Some(bone);
        while let Some(b) = current {
            match self.sorted.get_mut(b) {
                Some(flag) if !*flag => *flag = true,
                _ => break,
            }
            self.cache.push(Updatable::Bone(b));
            current = self.parents.get(b).copied().flatten();
        }
        if let Some(run) = self.cache.get_mut(start..) {
            run.reverse();
        }
    }

    /// Mark `bone`'s descendants unsorted so they are recomputed after a
    /// constraint.
    ///
    /// A child's subtree is visited only if the child was sorted. Each child is
    /// marked unsorted before its own children are visited, which gives the
    /// same result as the recursive form on a tree and ends on a parent cycle.
    fn sort_reset(&mut self, bone: usize) {
        let mut stack = vec![bone];
        while let Some(b) = stack.pop() {
            let Some(kids) = self.children.get(b) else {
                continue;
            };
            for &child in kids {
                if let Some(flag) = self.sorted.get_mut(child) {
                    if *flag {
                        stack.push(child);
                    }
                    *flag = false;
                }
            }
        }
    }
}
