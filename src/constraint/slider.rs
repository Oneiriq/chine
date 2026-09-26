//! Slider constraints (Spine 4.3).
//!
//! A slider reads one bone property (local pose or world transform), maps it to
//! a scrub time, and applies the animation it drives at that time, so the bone
//! scrubs that animation. A bone-less slider's scrub time instead comes from a
//! SLIDER_TIME timeline.

use crate::anim::MixFrom;
use crate::skel::{Bone, Skeleton};

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
    /// Index (into [`crate::data::SkeletonData::animations`]) of the animation
    /// this slider scrubs, set after the animations are loaded.
    pub animation_index: Option<usize>,
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
            animation_index: None,
        }
    }
}

/// A runtime slider pose: its current scrub time and mix, both animatable (by
/// the SLIDER_TIME and SLIDER_MIX timelines).
#[derive(Debug, Clone, Copy)]
pub(crate) struct SliderPose {
    /// Scrub time, used directly when the slider has no driving bone.
    pub time: f32,
    /// Mix weight in `[0, 1]`.
    pub mix: f32,
}

impl SliderPose {
    /// Build a runtime pose from setup data.
    #[must_use]
    pub(crate) fn from_data(data: &SliderData) -> Self {
        Self {
            time: data.time,
            mix: data.mix,
        }
    }
}

/// Read the property value a slider maps to its scrub time, in local pose space
/// or, when `local` is false, decomposed from the bone's world transform.
fn read_bone_property(bone: &Bone, property: SliderProperty, local: bool) -> f32 {
    if local {
        return match property {
            SliderProperty::Rotate => bone.rotation,
            SliderProperty::X => bone.x,
            SliderProperty::Y => bone.y,
            SliderProperty::ScaleX => bone.scale_x,
            SliderProperty::ScaleY => bone.scale_y,
            SliderProperty::ShearY => bone.shear_y,
        };
    }
    match property {
        SliderProperty::Rotate => bone.c().atan2(bone.a()).to_degrees(),
        SliderProperty::X => bone.world_x(),
        SliderProperty::Y => bone.world_y(),
        SliderProperty::ScaleX => bone.a().hypot(bone.c()),
        SliderProperty::ScaleY => bone.b().hypot(bone.d()),
        // The y-axis's deviation from perpendicular to the x-axis.
        SliderProperty::ShearY => {
            let x_angle = bone.c().atan2(bone.a());
            let y_angle = bone.d().atan2(bone.b());
            (y_angle - x_angle).to_degrees() - 90.0
        }
    }
}

/// Apply the slider at index `c`: read its bone's local property, map it to a
/// scrub time (`offset + (value - property offset) * scale`, looped or clamped
/// against the driven animation's duration), and apply that animation at the
/// scrub time. Sliders with no bone, property, or driven animation do nothing,
/// and so do sliders whose slider, bone, or animation index is out of range.
pub(crate) fn solve(skel: &mut Skeleton, c: usize) {
    let data = skel.data_arc();
    let Some(slider) = data.sliders.get(c) else {
        return;
    };
    // The skeleton sizes its slider poses from the same data, so `c` is in
    // range for `slider_pose` once it is in range for `data.sliders`.
    let pose = skel.slider_pose(c);
    if pose.mix == 0.0 {
        return;
    }
    let Some(animation_index) = slider.animation_index else {
        return;
    };
    let Some(animation) = data.animations.get(animation_index) else {
        return;
    };
    let duration = animation.duration();
    // A driving bone computes the scrub time from its property. Otherwise the
    // (possibly animated) pose time is used directly.
    let mut time = if let (Some(bone_index), Some(property)) = (slider.bone, slider.property) {
        let value = match skel.bone(bone_index) {
            Some(bone) => read_bone_property(bone, property, slider.local),
            None => return,
        };
        slider.offset + (value - slider.property_offset) * slider.scale
    } else {
        pose.time
    };
    if slider.looping && duration > 0.0 {
        // Spine maps the time into `[0, 2 * duration)`, then applies the
        // animation looped, which wraps it into `[0, duration)`.
        time = (duration + time % duration) % duration;
    } else {
        time = time.max(0.0);
    }
    animation.apply(
        skel,
        time,
        time,
        pose.mix,
        MixFrom::Current,
        slider.additive,
    );
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::anim::{Animation, BoneTimeline, Timeline};
    use crate::data::{BoneData, SkeletonData};
    use crate::skel::Skeleton;

    #[test]
    fn slider_scrubs_its_animation_from_a_bone() {
        // The scrubbed animation rotates bone 1 from 0 to 90 over time 0 to 1.
        let mut rot = BoneTimeline::one_value(1, 2, 0);
        rot.set_frame1(0, 0.0, 0.0);
        rot.set_frame1(1, 1.0, 90.0);
        let scrub = Animation::new("scrub", 1.0, vec![Timeline::Rotate(rot)]);

        // The slider maps bone 0's rotation directly onto the scrub time.
        let slider = SliderData {
            name: "s".into(),
            bone: Some(0),
            property: Some(SliderProperty::Rotate),
            animation_index: Some(0),
            mix: 1.0,
            scale: 1.0,
            ..Default::default()
        };
        let data = SkeletonData {
            bones: vec![
                BoneData {
                    index: 0,
                    name: "driver".into(),
                    ..Default::default()
                },
                BoneData {
                    index: 1,
                    name: "target".into(),
                    parent: Some(0),
                    ..Default::default()
                },
            ],
            sliders: vec![slider],
            animations: vec![Arc::new(scrub)],
            ..Default::default()
        };

        let mut sk = Skeleton::new(Arc::new(data));
        // Driving bone 0 to rotation 0.5 scrubs the animation to time 0.5.
        sk.bone_mut(0).unwrap().rotation = 0.5;
        sk.update_world_transform();
        // The scrubbed animation rotated bone 1 to ~45 degrees.
        let target = sk.bone(1).unwrap().rotation;
        assert!((target - 45.0).abs() < 1.0, "target rotation = {target}");
    }

    #[test]
    fn slider_time_timeline_scrubs_a_bone_less_slider() {
        use crate::anim::{AnimationState, ConstraintTimeline};

        // Scrubbed animation rotates bone 1 from 0 to 90 over time 0..1.
        let mut rot = BoneTimeline::one_value(1, 2, 0);
        rot.set_frame1(0, 0.0, 0.0);
        rot.set_frame1(1, 1.0, 90.0);
        let scrub = Animation::new("scrub", 1.0, vec![Timeline::Rotate(rot)]);

        // A bone-less slider scrubbing that animation. Its time comes from a
        // SLIDER_TIME timeline rather than a bone.
        let slider = SliderData {
            name: "s".into(),
            animation_index: Some(0),
            mix: 1.0,
            scale: 1.0,
            ..Default::default()
        };
        let data = SkeletonData {
            bones: vec![
                BoneData {
                    index: 0,
                    name: "root".into(),
                    ..Default::default()
                },
                BoneData {
                    index: 1,
                    name: "target".into(),
                    parent: Some(0),
                    ..Default::default()
                },
            ],
            sliders: vec![slider],
            animations: vec![Arc::new(scrub)],
            ..Default::default()
        };

        // A SLIDER_TIME timeline sets the slider's scrub time to 0.5.
        let mut st = ConstraintTimeline::new(0, 1, 0, 2);
        st.set_frame(0, 0.0, &[0.5]);
        let control = Animation::new("control", 0.0, vec![Timeline::SliderTime(st)]);

        let mut sk = Skeleton::new(Arc::new(data));
        let mut state = AnimationState::new();
        state.set_animation(Arc::new(control), false);
        state.update(0.0);
        state.apply(&mut sk);
        sk.update_world_transform();
        // Scrub time 0.5 rotated bone 1 to ~45 degrees.
        let target = sk.bone(1).unwrap().rotation;
        assert!((target - 45.0).abs() < 1.0, "target rotation = {target}");
    }

    // A looping slider mapped its time into `[0, 2 * duration)` and applied
    // the animation there, so every time past the first loop held the last
    // frame. Spine applies the animation looped, which wraps the time.
    #[test]
    fn looping_slider_wraps_past_the_animation_end() {
        let mut rot = BoneTimeline::one_value(1, 2, 0);
        rot.set_frame1(0, 0.0, 0.0);
        rot.set_frame1(1, 1.0, 90.0);
        let scrub = Animation::new("scrub", 1.0, vec![Timeline::Rotate(rot)]);
        let slider = SliderData {
            name: "s".into(),
            bone: Some(0),
            property: Some(SliderProperty::Rotate),
            animation_index: Some(0),
            looping: true,
            ..Default::default()
        };
        let data = SkeletonData {
            bones: vec![
                BoneData {
                    index: 0,
                    name: "driver".into(),
                    ..Default::default()
                },
                BoneData {
                    index: 1,
                    name: "target".into(),
                    parent: Some(0),
                    ..Default::default()
                },
            ],
            sliders: vec![slider],
            animations: vec![Arc::new(scrub)],
            ..Default::default()
        };
        let mut sk = Skeleton::new(Arc::new(data));
        for (driver, expected) in [(0.25, 22.5), (1.5, 45.0), (2.75, 67.5), (-0.25, 67.5)] {
            sk.set_bones_to_setup_pose();
            sk.bone_mut(0).unwrap().rotation = driver;
            sk.update_world_transform();
            let target = sk.bone(1).unwrap().rotation;
            assert!(
                (target - expected).abs() < 0.5,
                "driver {driver}: target rotation {target}, expected {expected}"
            );
        }
    }

    // Spine computes the bones a slider's animation changes again after the
    // slider, with their descendants. The update cache computed a bone once,
    // so a bone sorted before the slider kept a world transform that ignored
    // the slider.
    #[test]
    fn bones_a_slider_animates_are_computed_after_it() {
        // The scrubbed animation rotates the root to 90 degrees at time 1.
        let mut rot = BoneTimeline::one_value(0, 1, 0);
        rot.set_frame1(0, 1.0, 90.0);
        let scrub = Animation::new("scrub", 1.0, vec![Timeline::Rotate(rot)]);
        // A slider on the root's child, read in world space, pinned to time 1.
        let slider = SliderData {
            name: "s".into(),
            bone: Some(1),
            property: Some(SliderProperty::Rotate),
            animation_index: Some(0),
            offset: 1.0,
            scale: 0.0,
            ..Default::default()
        };
        let data = SkeletonData {
            bones: vec![
                BoneData {
                    index: 0,
                    name: "root".into(),
                    ..Default::default()
                },
                BoneData {
                    index: 1,
                    name: "child".into(),
                    parent: Some(0),
                    ..Default::default()
                },
            ],
            sliders: vec![slider],
            animations: vec![Arc::new(scrub)],
            ..Default::default()
        };
        let mut sk = Skeleton::new(Arc::new(data));
        sk.update_world_transform();
        let root = sk.bone(0).unwrap();
        assert!(
            root.a().abs() < 1e-5 && (root.c() - 1.0).abs() < 1e-5,
            "{root:?}"
        );
        let child = sk.bone(1).unwrap();
        assert!((child.c() - 1.0).abs() < 1e-5, "{child:?}");
    }

    #[test]
    fn out_of_range_slider_or_animation_index_is_ignored() {
        let slider = SliderData {
            name: "s".into(),
            bone: Some(0),
            property: Some(SliderProperty::Rotate),
            animation_index: Some(5),
            ..Default::default()
        };
        let data = SkeletonData {
            bones: vec![BoneData {
                index: 0,
                name: "root".into(),
                ..Default::default()
            }],
            sliders: vec![slider],
            ..Default::default()
        };
        let mut sk = Skeleton::new(Arc::new(data));
        // The slider's animation index is past the (empty) animation list.
        sk.update_world_transform();
        // A slider index past the slider list does nothing.
        solve(&mut sk, 1);
        assert_eq!(sk.bone(0).unwrap().rotation, 0.0);
    }
}
