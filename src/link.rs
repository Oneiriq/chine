//! Linked-mesh resolution, shared by both loaders.
//!
//! A Spine export can declare a "linked mesh" that borrows a source mesh's
//! vertices, triangles, and UVs. Only its path, color, and sequence are its
//! own. The source can sit in another skin and in another slot. Both the JSON
//! and binary loaders build [`LinkedMeshAttachment`] placeholders while
//! reading skins. This pass replaces each with the concrete [`MeshAttachment`]
//! it resolves to, once every skin is loaded and before any timeline binds to
//! it.
//!
//! A link that inherits its source's timelines names the source as its
//! timeline source. When the link sits in another slot than its source, that
//! slot joins the source's timeline slots, so the source's deform and
//! sequence timelines reach the link there too.
//!
//! [`LinkedMeshAttachment`]: crate::attach::LinkedMeshAttachment

use std::collections::hash_map::Entry;
use std::collections::HashMap;

use crate::attach::{Attachment, AttachmentKey, LinkedMeshAttachment, MeshAttachment};
use crate::data::SkeletonData;
use crate::skin::Skin;

/// The geometry values (see `MeshAttachment::geometry_len`) that resolving
/// linked meshes may copy in total.
///
/// Each resolved link owns a copy of its source's geometry. A link costs a few
/// bytes in a file while its source can be large, so many links to one big
/// mesh could otherwise exhaust memory. Real rigs copy far less than this.
const MAX_LINKED_GEOMETRY: usize = 1 << 26;

/// Where the source mesh of `link`, shown in `slot`, lives. A link names the
/// default skin by leaving its skin unset or by naming "default".
fn source_key(link: &LinkedMeshAttachment, slot: usize) -> AttachmentKey {
    AttachmentKey {
        skin: link.skin.clone().filter(|skin| skin != "default"),
        slot: link.source_slot.unwrap_or(slot),
        name: link.source.clone(),
    }
}

/// The named skins by name. The first skin with a name wins, as a lookup by
/// name finds it.
fn skins_by_name(data: &SkeletonData) -> HashMap<String, usize> {
    let mut by_name = HashMap::with_capacity(data.skins.len());
    for (i, skin) in data.skins.iter().enumerate() {
        by_name.entry(skin.name.clone()).or_insert(i);
    }
    by_name
}

/// The skin `name` names: the default skin for `None`.
fn skin<'d>(
    data: &'d SkeletonData,
    by_name: &HashMap<String, usize>,
    name: Option<&str>,
) -> Option<&'d Skin> {
    match name {
        None => Some(&data.default_skin),
        Some(name) => data.skins.get(*by_name.get(name)?),
    }
}

/// The mesh at `key`, if the skin it names holds a mesh there.
fn mesh_at<'d>(
    data: &'d SkeletonData,
    by_name: &HashMap<String, usize>,
    key: &AttachmentKey,
) -> Option<&'d MeshAttachment> {
    match skin(data, by_name, key.skin.as_deref())?.attachment(key.slot, &key.name)? {
        Attachment::Mesh(mesh) => Some(mesh),
        _ => None,
    }
}

/// Call `f` with every linked mesh whose source exists, and its source mesh.
/// The JSON loader charges its load limit this way.
#[cfg(feature = "json")]
pub(crate) fn for_each_link_source(
    data: &SkeletonData,
    mut f: impl FnMut(&LinkedMeshAttachment, &MeshAttachment),
) {
    let by_name = skins_by_name(data);
    for skin in std::iter::once(&data.default_skin).chain(&data.skins) {
        for (slot, _, attachment) in skin.iter() {
            if let Attachment::LinkedMesh(link) = attachment {
                if let Some(source) = mesh_at(data, &by_name, &source_key(link, slot)) {
                    f(link, source);
                }
            }
        }
    }
}

/// Resolve every linked mesh to its source mesh's geometry, once all skins
/// are parsed. A linked mesh whose source is missing, or whose copy would pass
/// [`MAX_LINKED_GEOMETRY`], is left in place (and skipped at render and
/// atlas-binding time).
pub(crate) fn resolve_linked_meshes(data: &mut SkeletonData) {
    resolve_linked_meshes_within(data, MAX_LINKED_GEOMETRY);
}

/// [`resolve_linked_meshes`] with `budget` geometry values to copy.
fn resolve_linked_meshes_within(data: &mut SkeletonData, mut budget: usize) {
    let by_name = skins_by_name(data);

    // Copy out each source mesh once, before any link is replaced.
    let mut sources: HashMap<AttachmentKey, MeshAttachment> = HashMap::new();
    for skin in std::iter::once(&data.default_skin).chain(&data.skins) {
        for (slot, _, attachment) in skin.iter() {
            if let Attachment::LinkedMesh(link) = attachment {
                if let Entry::Vacant(entry) = sources.entry(source_key(link, slot)) {
                    if let Some(mesh) = mesh_at(data, &by_name, entry.key()) {
                        entry.insert(mesh.clone());
                    }
                }
            }
        }
    }

    // Replace each link, and note the slots that join a source's timeline
    // slots.
    let mut timeline_slots: HashMap<AttachmentKey, Vec<usize>> = HashMap::new();
    for skin in std::iter::once(&mut data.default_skin).chain(&mut data.skins) {
        for (slot, _, attachment) in skin.iter_mut() {
            let Attachment::LinkedMesh(link) = attachment else {
                continue;
            };
            let key = source_key(link, slot);
            let Some(source) = sources.get(&key) else {
                continue;
            };
            let Some(rest) = budget.checked_sub(source.geometry_len()) else {
                continue;
            };
            budget = rest;
            let mesh = link.resolve(source, &key);
            if link.inherit_timelines && key.slot != slot {
                timeline_slots.entry(key).or_default().push(slot);
            }
            *attachment = Attachment::Mesh(mesh);
        }
    }

    for (key, slots) in timeline_slots {
        let skin = match key.skin.as_deref() {
            None => Some(&mut data.default_skin),
            Some(name) => by_name.get(name).and_then(|&i| data.skins.get_mut(i)),
        };
        if let Some(Attachment::Mesh(mesh)) =
            skin.and_then(|skin| skin.attachment_mut(key.slot, &key.name))
        {
            let mut all = mesh.timeline_slots.to_vec();
            all.extend(slots);
            all.sort_unstable();
            all.dedup();
            mesh.timeline_slots = all.into();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attach::{LinkedMeshAttachment, MeshVertices, Sequence};
    use crate::data::Color;

    fn mesh(name: &str) -> MeshAttachment {
        MeshAttachment::new(
            name,
            name,
            MeshVertices::Unweighted(vec![0.0, 0.0, 10.0, 0.0, 0.0, 10.0]),
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            vec![0, 1, 2],
        )
    }

    fn rig_with_links(links: usize) -> SkeletonData {
        let mut skin = Skin::new("default");
        skin.set(0, "m", Attachment::Mesh(mesh("m")));
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

    fn resolved<'a>(skin: &'a Skin, slot: usize, name: &str) -> &'a MeshAttachment {
        match skin.attachment(slot, name) {
            Some(Attachment::Mesh(m)) => m,
            other => panic!("{name} did not resolve: {other:?}"),
        }
    }

    #[test]
    fn links_resolve_to_their_source_mesh() {
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
        // The source holds 15 geometry values, so 50 pays for three copies.
        resolve_linked_meshes_within(&mut data, 50);
        assert_eq!(count(&data, true), 7);
        assert_eq!(count(&data, false), 4);
    }

    // A link found its source in its own skin and slot only. Spine looks in
    // the skin the link names (the default skin when it names none) and in
    // the slot it names, so links across skins or slots drew nothing.
    #[test]
    fn links_find_sources_in_other_skins_and_slots() {
        let mut default = Skin::new("default");
        default.set(0, "arm", Attachment::Mesh(mesh("arm")));
        let mut boy = Skin::new("boy");
        boy.set(1, "sleeve", Attachment::Mesh(mesh("sleeve")));
        let mut girl = Skin::new("girl");
        let mut across_skins = LinkedMeshAttachment::new(
            "girl/sleeve",
            "girl/sleeve",
            Some("boy".into()),
            "sleeve",
            Color::WHITE,
            true,
        );
        across_skins.sequence = Some(Sequence::new(3, 1, 2, 0));
        girl.set(1, "sleeve", Attachment::LinkedMesh(across_skins));
        let mut across_slots =
            LinkedMeshAttachment::new("hand", "hand", None, "arm", Color::WHITE, true);
        across_slots.source_slot = Some(0);
        girl.set(2, "hand", Attachment::LinkedMesh(across_slots));
        let mut data = SkeletonData {
            default_skin: default,
            skins: vec![boy, girl],
            ..Default::default()
        };

        resolve_linked_meshes(&mut data);

        let girl = &data.skins[1];
        let sleeve = resolved(girl, 1, "sleeve");
        assert_eq!(sleeve.path, "girl/sleeve");
        // The link keeps its own sequence.
        assert_eq!(sleeve.sequence.as_ref().map(|s| s.count), Some(3));
        let boy_sleeve = AttachmentKey {
            skin: Some("boy".into()),
            slot: 1,
            name: "sleeve".into(),
        };
        assert_eq!(sleeve.timeline_source.as_ref(), Some(&boy_sleeve));

        let hand = resolved(girl, 2, "hand");
        assert_eq!(hand.triangles, vec![0, 1, 2]);
        let default_arm = AttachmentKey {
            skin: None,
            slot: 0,
            name: "arm".into(),
        };
        assert_eq!(hand.timeline_source.as_ref(), Some(&default_arm));
        // The source's timelines now also reach slot 2.
        let arm = resolved(&data.default_skin, 0, "arm");
        assert_eq!(&arm.timeline_slots[..], [2]);
        // A source in its link's slot gains no timeline slot.
        assert!(resolved(&data.skins[0], 1, "sleeve")
            .timeline_slots
            .is_empty());
    }

    // A link that does not inherit its source's timelines keeps its own.
    #[test]
    fn non_inheriting_links_keep_their_own_timelines() {
        let mut skin = Skin::new("default");
        skin.set(0, "arm", Attachment::Mesh(mesh("arm")));
        let mut link = LinkedMeshAttachment::new("hand", "hand", None, "arm", Color::WHITE, false);
        link.source_slot = Some(0);
        skin.set(1, "hand", Attachment::LinkedMesh(link));
        let mut data = SkeletonData {
            default_skin: skin,
            ..Default::default()
        };
        resolve_linked_meshes(&mut data);
        assert_eq!(
            resolved(&data.default_skin, 1, "hand").timeline_source,
            None
        );
        assert!(resolved(&data.default_skin, 0, "arm")
            .timeline_slots
            .is_empty());
    }
}
