//! Linked-mesh resolution, shared by both loaders.
//!
//! A Spine export can declare a "linked mesh" that borrows another mesh's
//! vertices, triangles, and UVs (only its color, sequence, and deform differ).
//! Both the JSON and binary loaders build [`LinkedMeshAttachment`] placeholders
//! while reading skins. This pass replaces each with the concrete
//! [`MeshAttachment`] it resolves to, once every skin is loaded and before any
//! deform timeline binds to it.
//!
//! [`LinkedMeshAttachment`]: crate::attach::LinkedMeshAttachment

use std::collections::HashMap;

use crate::attach::{Attachment, MeshAttachment};
use crate::data::SkeletonData;
use crate::skin::Skin;

/// The geometry values (see `MeshAttachment::geometry_len`) that resolving
/// linked meshes may copy in total.
///
/// Each resolved link owns a copy of its parent's geometry. A link costs a few
/// bytes in a file while its parent can be large, so many links to one big
/// mesh could otherwise exhaust memory. Real rigs copy far less than this.
const MAX_LINKED_GEOMETRY: usize = 1 << 26;

/// Resolve every linked mesh to its parent mesh's geometry, once all skins are
/// parsed. A linked mesh whose parent is missing, or whose copy would pass
/// [`MAX_LINKED_GEOMETRY`], is left in place (and skipped at render and
/// atlas-binding time).
pub(crate) fn resolve_linked_meshes(data: &mut SkeletonData) {
    resolve_linked_meshes_within(data, MAX_LINKED_GEOMETRY);
}

/// [`resolve_linked_meshes`] with `budget` geometry values to copy.
fn resolve_linked_meshes_within(data: &mut SkeletonData, mut budget: usize) {
    let mut parents: HashMap<(String, usize, String), MeshAttachment> = HashMap::new();
    collect_meshes(&data.default_skin, &mut parents);
    for skin in &data.skins {
        collect_meshes(skin, &mut parents);
    }
    resolve_skin(&mut data.default_skin, &parents, &mut budget);
    for skin in &mut data.skins {
        resolve_skin(skin, &parents, &mut budget);
    }
}

/// Record every mesh attachment keyed by `(skin, slot, name)` so linked meshes
/// can find their parent geometry.
fn collect_meshes(skin: &Skin, out: &mut HashMap<(String, usize, String), MeshAttachment>) {
    for (slot, name, att) in skin.iter() {
        if let Attachment::Mesh(m) = att {
            out.insert((skin.name.clone(), slot, name.to_string()), m.clone());
        }
    }
}

/// Replace each linked mesh in `skin` with the concrete mesh it resolves to,
/// while `budget` covers the parent's geometry.
fn resolve_skin(
    skin: &mut Skin,
    parents: &HashMap<(String, usize, String), MeshAttachment>,
    budget: &mut usize,
) {
    let skin_name = skin.name.clone();
    for (slot, _name, att) in skin.iter_mut() {
        if let Attachment::LinkedMesh(lm) = att {
            let parent_skin = lm.skin.clone().unwrap_or_else(|| skin_name.clone());
            if let Some(parent) = parents.get(&(parent_skin, slot, lm.parent.clone())) {
                let Some(rest) = budget.checked_sub(parent.geometry_len()) else {
                    continue;
                };
                *budget = rest;
                let mut m = lm.resolve(parent);
                // An inheriting link shares the parent's deform source. Otherwise
                // it uses its own skin's deforms.
                m.deform_skin = if lm.inherit_deform {
                    parent.deform_skin.clone()
                } else if skin_name == "default" {
                    None
                } else {
                    Some(skin_name.clone())
                };
                *att = Attachment::Mesh(m);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attach::{LinkedMeshAttachment, MeshVertices};
    use crate::data::Color;

    fn rig_with_links(links: usize) -> SkeletonData {
        let mut skin = Skin::new("default");
        let mesh = MeshAttachment::new(
            "m",
            "m",
            MeshVertices::Unweighted(vec![0.0, 0.0, 10.0, 0.0, 0.0, 10.0]),
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            vec![0, 1, 2],
        );
        skin.set(0, "m", Attachment::Mesh(mesh));
        for i in 0..links {
            let name = format!("link{i}");
            let link = LinkedMeshAttachment::new(&name, &name, None, "m", Color::WHITE, true);
            skin.set(0, name, Attachment::LinkedMesh(link));
        }
        SkeletonData {
            default_skin: skin,
            ..Default::default()
        }
    }

    fn count(data: &SkeletonData, linked: bool) -> usize {
        data.default_skin
            .iter()
            .filter(|(_, _, att)| matches!(att, Attachment::LinkedMesh(_)) == linked)
            .count()
    }

    #[test]
    fn links_resolve_to_their_parent_mesh() {
        let mut data = rig_with_links(3);
        resolve_linked_meshes(&mut data);
        assert_eq!(count(&data, true), 0);
        assert_eq!(count(&data, false), 4);
    }

    // Many cheap links to one large mesh each copied the whole mesh, so a small
    // file could demand gigabytes. Copies stop once the budget is spent, and
    // the remaining links stay unresolved (they draw nothing).
    #[test]
    fn linked_mesh_copies_stop_at_the_budget() {
        let mut data = rig_with_links(10);
        // The parent holds 15 geometry values, so 50 pays for three copies.
        resolve_linked_meshes_within(&mut data, 50);
        assert_eq!(count(&data, true), 7);
        assert_eq!(count(&data, false), 4);
    }
}
