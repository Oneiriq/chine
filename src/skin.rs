//! Skins: named sets of attachments.
//!
//! A [`Skin`] maps `(slot index, attachment name)` to an [`Attachment`]. A
//! skeleton's *default* skin holds its base attachments; *named* skins (e.g.
//! character variants) override or add to it. Resolving an attachment checks
//! the active skin first, then the default.

use std::collections::HashMap;

use crate::attach::Attachment;

/// A named set of attachments keyed by `(slot index, attachment name)`.
#[derive(Debug, Clone, Default)]
pub struct Skin {
    /// Skin name (`"default"` for the base skin).
    pub name: String,
    // slot index -> (attachment name -> attachment)
    attachments: HashMap<usize, HashMap<String, Attachment>>,
}

impl Skin {
    /// A new, empty skin with the given name.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
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

    /// `true` if the skin defines no attachments.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.attachments.is_empty()
    }

    /// Mutable iterator over every attachment, for binding to an atlas.
    pub(crate) fn attachments_mut(&mut self) -> impl Iterator<Item = &mut Attachment> {
        self.attachments.values_mut().flat_map(|m| m.values_mut())
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
