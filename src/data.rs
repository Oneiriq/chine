//! Setup-pose skeleton data: the immutable rig loaded from a Spine export.
//!
//! [`SkeletonData`] is the shared, read-only description of a rig: its bones,
//! slots, and (in later milestones) skins, attachments, animations, and
//! constraints. A posable `Skeleton` instance is created from it; many
//! skeletons can share one `SkeletonData`.

use std::sync::Arc;

use glam::Vec2;

use crate::anim::Animation;
use crate::attach::Attachment;
use crate::constraint::ik::IkConstraintData;
use crate::constraint::path::PathConstraintData;
use crate::constraint::transform::TransformConstraintData;
use crate::skin::Skin;

/// An RGBA color with components in `[0, 1]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Color {
    /// Red channel.
    pub r: f32,
    /// Green channel.
    pub g: f32,
    /// Blue channel.
    pub b: f32,
    /// Alpha channel.
    pub a: f32,
}

impl Color {
    /// Opaque white (`1, 1, 1, 1`), the default tint.
    pub const WHITE: Self = Self {
        r: 1.0,
        g: 1.0,
        b: 1.0,
        a: 1.0,
    };

    /// A color from its four channels (each expected in `[0, 1]`).
    #[must_use]
    pub const fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }
}

impl Default for Color {
    fn default() -> Self {
        Self::WHITE
    }
}

/// How a bone inherits its parent's world transform (Spine 4.3 `Inherit`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Inherit {
    /// Inherit translation, rotation, scale, and reflection.
    #[default]
    Normal,
    /// Inherit only the parent's translation.
    OnlyTranslation,
    /// Inherit everything except the parent's rotation and reflection.
    NoRotationOrReflection,
    /// Inherit everything except the parent's scale.
    NoScale,
    /// Inherit everything except the parent's scale and reflection.
    NoScaleOrReflection,
}

/// How a slot's attachment is blended into the frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BlendMode {
    /// Standard source-over alpha blending.
    #[default]
    Normal,
    /// Additive blending.
    Additive,
    /// Multiply blending.
    Multiply,
    /// Screen blending.
    Screen,
}

/// Setup-pose data for one bone in the rig.
///
/// Stored in hierarchy order in [`SkeletonData::bones`] (a parent always
/// precedes its children), so a single forward pass computes world transforms.
#[derive(Debug, Clone)]
pub struct BoneData {
    /// Index of this bone in [`SkeletonData::bones`].
    pub index: usize,
    /// Bone name, unique within the skeleton.
    pub name: String,
    /// Parent bone index, or `None` for the root bone.
    pub parent: Option<usize>,
    /// Bone length, used by IK and rendering helpers.
    pub length: f32,
    /// Setup-pose local position relative to the parent.
    pub position: Vec2,
    /// Setup-pose local rotation, in degrees.
    pub rotation: f32,
    /// Setup-pose local scale (`1, 1` is unscaled).
    pub scale: Vec2,
    /// Setup-pose local shear, in degrees.
    pub shear: Vec2,
    /// How this bone inherits the parent's world transform.
    pub inherit: Inherit,
}

impl Default for BoneData {
    fn default() -> Self {
        Self {
            index: 0,
            name: String::new(),
            parent: None,
            length: 0.0,
            position: Vec2::ZERO,
            rotation: 0.0,
            scale: Vec2::ONE,
            shear: Vec2::ZERO,
            inherit: Inherit::Normal,
        }
    }
}

/// Setup-pose data for one slot: a draw-order entry that displays the
/// attachment of a given name on a given bone.
#[derive(Debug, Clone)]
pub struct SlotData {
    /// Index of this slot in [`SkeletonData::slots`].
    pub index: usize,
    /// Slot name, unique within the skeleton.
    pub name: String,
    /// Index (into [`SkeletonData::bones`]) of the bone this slot follows.
    pub bone: usize,
    /// Setup-pose tint color applied to the attachment.
    pub color: Color,
    /// Optional dark color for two-color (tint-black) rendering.
    pub dark_color: Option<Color>,
    /// Name of the attachment shown in the setup pose, if any.
    pub attachment: Option<String>,
    /// Blend mode used when rendering this slot's attachment.
    pub blend: BlendMode,
}

/// An immutable Spine rig: the shared data that a `Skeleton` poses and animates.
///
/// Later milestones extend this with skins, attachments, animations, events,
/// and constraints; M1 establishes the bone/slot spine of the model.
#[derive(Debug, Clone, Default)]
pub struct SkeletonData {
    /// Skeleton name from the export, if present.
    pub name: Option<String>,
    /// Spine editor version string that produced the export.
    pub spine_version: Option<String>,
    /// Setup-pose bounds offset (the export's `x`/`y`).
    pub position: Vec2,
    /// Setup-pose bounds size (the export's `width`/`height`).
    pub size: Vec2,
    /// Bones in hierarchy order: the root first, every parent before its
    /// children.
    pub bones: Vec<BoneData>,
    /// Slots in setup-pose draw order (back to front).
    pub slots: Vec<SlotData>,
    /// The base skin holding the skeleton's default attachments.
    pub default_skin: Skin,
    /// Named skins (character variants) that override or extend the default.
    pub skins: Vec<Skin>,
    /// Animations from the export, each shareable for playback on a track.
    pub animations: Vec<Arc<Animation>>,
    /// IK constraints, applied after FK by [`crate::skel::Skeleton`].
    pub ik_constraints: Vec<IkConstraintData>,
    /// Transform constraints, applied after FK by [`crate::skel::Skeleton`].
    pub transform_constraints: Vec<TransformConstraintData>,
    /// Path constraints, applied after FK by [`crate::skel::Skeleton`].
    pub path_constraints: Vec<PathConstraintData>,
}

impl SkeletonData {
    /// Find a bone's index by name.
    #[must_use]
    pub fn find_bone(&self, name: &str) -> Option<usize> {
        self.bones.iter().position(|b| b.name == name)
    }

    /// Find a slot's index by name.
    #[must_use]
    pub fn find_slot(&self, name: &str) -> Option<usize> {
        self.slots.iter().position(|s| s.name == name)
    }

    /// Find a named skin.
    #[must_use]
    pub fn find_skin(&self, name: &str) -> Option<&Skin> {
        self.skins.iter().find(|s| s.name == name)
    }

    /// Resolve the attachment for `(slot, name)`: the active `skin` if it
    /// defines it, otherwise the default skin.
    #[must_use]
    pub fn attachment<'a>(
        &'a self,
        slot: usize,
        name: &str,
        skin: Option<&'a Skin>,
    ) -> Option<&'a Attachment> {
        skin.and_then(|s| s.attachment(slot, name))
            .or_else(|| self.default_skin.attachment(slot, name))
    }

    /// Find an animation by name.
    #[must_use]
    pub fn find_animation(&self, name: &str) -> Option<&Arc<Animation>> {
        self.animations.iter().find(|a| a.name() == name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn color_defaults_to_opaque_white() {
        assert_eq!(Color::default(), Color::WHITE);
    }

    #[test]
    fn bone_data_defaults_to_unit_scale() {
        assert_eq!(BoneData::default().scale, Vec2::ONE);
        assert_eq!(BoneData::default().inherit, Inherit::Normal);
    }

    #[test]
    fn find_bone_and_slot_by_name() {
        let data = SkeletonData {
            bones: vec![
                BoneData {
                    index: 0,
                    name: "root".into(),
                    ..Default::default()
                },
                BoneData {
                    index: 1,
                    name: "torso".into(),
                    parent: Some(0),
                    ..Default::default()
                },
            ],
            slots: vec![SlotData {
                index: 0,
                name: "head".into(),
                bone: 1,
                color: Color::WHITE,
                dark_color: None,
                attachment: None,
                blend: BlendMode::Normal,
            }],
            ..Default::default()
        };
        assert_eq!(data.find_bone("torso"), Some(1));
        assert_eq!(data.find_bone("missing"), None);
        assert_eq!(data.find_slot("head"), Some(0));
    }
}
