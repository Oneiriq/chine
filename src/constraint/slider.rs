//! Slider constraints (Spine 4.3).
//!
//! A slider exposes a single animatable value (its `time`) that other timelines
//! sample, and can map that value onto one local bone property. chine currently
//! loads slider setup data from an export; runtime application is not yet wired
//! into [`crate::skel::Skeleton`].

/// The single local bone property a slider can drive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SliderProperty {
    /// Local rotation.
    Rotate,
    /// Local X position.
    X,
    /// Local Y position.
    Y,
    /// Local X scale.
    ScaleX,
    /// Local Y scale.
    ScaleY,
    /// Local Y shear.
    ShearY,
}

/// Setup data for a slider constraint.
#[derive(Debug, Clone)]
pub struct SliderData {
    /// Constraint name, unique within the skeleton.
    pub name: String,
    /// Global constraint order (lower applies first).
    pub order: usize,
    /// Whether the active skin must list this constraint for it to apply.
    pub skin_required: bool,
    /// Whether the slider value wraps from its end back to the start.
    pub looping: bool,
    /// Whether the driven property adds to the pose instead of replacing it.
    pub additive: bool,
    /// Setup-pose slider time (the value other timelines sample).
    pub time: f32,
    /// Editor slider maximum (nonessential).
    pub max: f32,
    /// Setup mix (`0` disables, `1` full).
    pub mix: f32,
    /// Whether the driven bone property is read in local (vs world) space.
    pub local: bool,
    /// Index of the bone whose property the slider drives, if any.
    pub bone: Option<usize>,
    /// Which bone property the slider drives, if any.
    pub property: Option<SliderProperty>,
    /// Offset added to the driven property's value.
    pub property_offset: f32,
    /// Slider value offset.
    pub offset: f32,
    /// Slider value scale.
    pub scale: f32,
}

impl Default for SliderData {
    fn default() -> Self {
        Self {
            name: String::new(),
            order: 0,
            skin_required: false,
            looping: false,
            additive: false,
            time: 0.0,
            max: 1.0,
            mix: 1.0,
            local: false,
            bone: None,
            property: None,
            property_offset: 0.0,
            offset: 0.0,
            scale: 1.0,
        }
    }
}
