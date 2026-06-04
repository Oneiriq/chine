//! Linked-mesh resolution, shared by both loaders.
//!
//! A Spine export can declare a "linked mesh" that borrows another mesh's
//! vertices, triangles, and UVs (only its color, sequence, and deform differ).
//! Both the JSON and binary loaders build [`LinkedMeshAttachment`] placeholders
//! while reading skins; this pass replaces each with the concrete
//! [`MeshAttachment`] it resolves to, once every skin is loaded and before any
//! deform timeline binds to it.
//!
//! [`LinkedMeshAttachment`]: crate::attach::LinkedMeshAttachment

use std::collections::HashMap;

use crate::attach::{Attachment, MeshAttachment};
use crate::data::SkeletonData;
use crate::skin::Skin;

/// Resolve every linked mesh to its parent mesh's geometry, once all skins are
/// parsed. A linked mesh whose parent is missing is left in place (and skipped
/// at render and atlas-binding time).
pub(crate) fn resolve_linked_meshes(data: &mut SkeletonData) {
    let mut parents: HashMap<(String, usize, String), MeshAttachment> = HashMap::new();
    collect_meshes(&data.default_skin, &mut parents);
    for skin in &data.skins {
        collect_meshes(skin, &mut parents);
    }
    resolve_skin(&mut data.default_skin, &parents);
    for skin in &mut data.skins {
        resolve_skin(skin, &parents);
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

/// Replace each linked mesh in `skin` with the concrete mesh it resolves to.
fn resolve_skin(skin: &mut Skin, parents: &HashMap<(String, usize, String), MeshAttachment>) {
    let skin_name = skin.name.clone();
    for (slot, _name, att) in skin.iter_mut() {
        if let Attachment::LinkedMesh(lm) = att {
            let parent_skin = lm.skin.clone().unwrap_or_else(|| skin_name.clone());
            if let Some(parent) = parents.get(&(parent_skin, slot, lm.parent.clone())) {
                let mut m = lm.resolve(parent);
                // An inheriting link shares the parent's deform source; otherwise
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
