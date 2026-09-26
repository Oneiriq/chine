//! Skins: named sets of attachments.
//!
//! A [`Skin`] maps `(slot index, attachment name)` to an [`Attachment`]. A
//! skeleton's *default* skin holds its base attachments. *Named* skins (e.g.
//! character variants) override or add to it. Resolving an attachment checks
//! the active skin first, then the default.
//!
//! A named skin also lists the skin-required bones and constraints it
//! activates: while it is the active skin, those bones and constraints apply.

use std::collections::HashMap;

use crate::attach::Attachment;

/// A constraint named by its type and its index in that type's list in
/// [`crate::data::SkeletonData`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SkinConstraint {
    /// An IK constraint.
    Ik(usize),
    /// A transform constraint.
    Transform(usize),
    /// A path constraint.
    Path(usize),
    /// A physics constraint.
    Physics(usize),
    /// A slider.
    Slider(usize),
}

/// A named set of attachments keyed by `(slot index, attachment name)`.
#[derive(Debug, Clone, Default)]
pub struct Skin {
    /// Skin name (`"default"` for the base skin).
    pub name: String,
    /// The skin-required bones (indices into the skeleton's bones) this skin
    /// activates. Their ancestors are activated too.
    pub bones: Vec<usize>,
    /// The skin-required constraints this skin activates.
    pub constraints: Vec<SkinConstraint>,
    // slot index -> (attachment name -> attachment)
    attachments: HashMap<usize, HashMap<String, Attachment>>,
}

impl Skin {
    /// A new, empty skin with the given name.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            bones: Vec::new(),
            constraints: Vec::new(),
            attachments: HashMap::new(),
        }
    }

    /// Set the attachment shown for `(slot, name)`.
    pub fn set(&mut self, slot: usize, name: impl Into<String>, attachment: Attachment) {
        self.attachments
            .entry(slot)
            .or_default()
            .insert(name.into(), attachment);
    }

    /// The attachment for `(slot, name)`, if this skin defines it.
    #[must_use]
    pub fn attachment(&self, slot: usize, name: &str) -> Option<&Attachment> {
        self.attachments.get(&slot)?.get(name)
    }

    /// Mutable access to the attachment for `(slot, name)`, for loaders that
    /// finish an attachment once every skin is read.
    pub(crate) fn attachment_mut(&mut self, slot: usize, name: &str) -> Option<&mut Attachment> {
        self.attachments.get_mut(&slot)?.get_mut(name)
    }

    /// `true` if the skin defines no attachments.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.attachments.is_empty()
    }

    /// Mutable iterator over every attachment, for binding to an atlas.
    pub(crate) fn attachments_mut(&mut self) -> impl Iterator<Item = &mut Attachment> {
        self.attachments.values_mut().flat_map(|m| m.values_mut())
    }

    /// Iterate every `(slot, name, attachment)` this skin defines.
    pub(crate) fn iter(&self) -> impl Iterator<Item = (usize, &str, &Attachment)> {
        self.attachments
            .iter()
            .flat_map(|(slot, m)| m.iter().map(move |(name, att)| (*slot, name.as_str(), att)))
    }

    /// Mutably iterate every `(slot, name, attachment)` this skin defines.
    pub(crate) fn iter_mut(&mut self) -> impl Iterator<Item = (usize, &str, &mut Attachment)> {
        self.attachments.iter_mut().flat_map(|(slot, m)| {
            m.iter_mut()
                .map(move |(name, att)| (*slot, name.as_str(), att))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attach::RegionAttachment;

    fn region(name: &str) -> Attachment {
        Attachment::Region(RegionAttachment::new(name, name))
    }

    #[test]
    fn set_and_resolve_attachment() {
        let mut skin = Skin::new("default");
        assert!(skin.is_empty());
        skin.set(2, "head", region("head"));
        skin.set(2, "head-angry", region("head-angry"));
        skin.set(0, "torso", region("torso"));

        assert!(!skin.is_empty());
        assert!(skin.attachment(2, "head").is_some());
        assert!(skin.attachment(2, "head-angry").is_some());
        assert!(skin.attachment(0, "torso").is_some());
        // wrong slot or name -> None.
        assert!(skin.attachment(0, "head").is_none());
        assert!(skin.attachment(2, "missing").is_none());
    }
}
