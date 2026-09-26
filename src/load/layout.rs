//! Tests for Spine 4.3 JSON layout details: the keys and nesting current
//! exports use, which older exports spelled differently.

use std::sync::Arc;

use super::*;
use crate::anim::MixFrom;
use crate::skel::Skeleton;

/// Load `json` and apply its animation `name` at `time`, from the setup pose.
fn posed(json: &str, name: &str, time: f32) -> Skeleton {
    let data = from_json(json).unwrap();
    let anim = Arc::clone(data.find_animation(name).unwrap());
    let mut sk = Skeleton::new(Arc::new(data));
    anim.apply(&mut sk, -1.0, time, 1.0, MixFrom::Setup, false);
    sk
}

// Path position and spacing keys hold their value under "value". The loader
// read "position" and "spacing", so both timelines keyed 0.
#[test]
fn path_position_and_spacing_timelines_read_value() {
    let json = r#"{
        "bones": [ { "name": "root" }, { "name": "b", "parent": "root" } ],
        "slots": [ { "name": "s", "bone": "root" } ],
        "constraints": [
            { "type": "path", "name": "p", "slot": "s", "bones": [ "b" ] }
        ],
        "animations": {
            "a": { "path": { "p": {
                "position": [ { "value": 0.5 } ],
                "spacing": [ { "value": 3 } ]
            } } }
        }
    }"#;
    let mut sk = posed(json, "a", 0.0);
    let (pose, _) = sk.path_pose_and_setup(0).unwrap();
    assert_eq!(pose.position, 0.5);
    assert_eq!(pose.spacing, 3.0);
}

// A transform or path mix key without "mixY" uses its "mixX", as Spine's
// reader does. The loader used 1.
#[test]
fn omitted_mix_y_keys_take_mix_x() {
    let json = r#"{
        "bones": [ { "name": "root" }, { "name": "b", "parent": "root" } ],
        "slots": [ { "name": "s", "bone": "root" } ],
        "constraints": [
            { "type": "transform", "name": "t", "source": "root", "bones": [ "b" ] },
            { "type": "path", "name": "p", "slot": "s", "bones": [ "b" ] }
        ],
        "animations": {
            "a": {
                "transform": { "t": [ { "mixX": 0.25 } ] },
                "path": { "p": { "mix": [ { "mixX": 0.75 } ] } }
            }
        }
    }"#;
    let mut sk = posed(json, "a", 0.0);
    let (transform, _) = sk.transform_pose_and_setup(0).unwrap();
    assert_eq!((transform.mix_x, transform.mix_y), (0.25, 0.25));
    let (path, _) = sk.path_pose_and_setup(0).unwrap();
    assert_eq!((path.mix_x, path.mix_y), (0.75, 0.75));
}

// Bone "inherit" timelines were skipped as an unknown bone channel.
#[test]
fn bone_inherit_timelines_apply() {
    let json = r#"{
        "bones": [ { "name": "root" }, { "name": "b", "parent": "root" } ],
        "animations": {
            "a": { "bones": { "b": { "inherit": [
                { "inherit": "onlyTranslation" },
                { "time": 1, "inherit": "noScaleOrReflection" }
            ] } } }
        }
    }"#;
    let data = from_json(json).unwrap();
    assert_eq!(data.find_animation("a").unwrap().duration(), 1.0);
    let sk = posed(json, "a", 0.5);
    assert_eq!(sk.bone(1).unwrap().inherit(), Inherit::OnlyTranslation);
    let sk = posed(json, "a", 1.0);
    assert_eq!(sk.bone(1).unwrap().inherit(), Inherit::NoScaleOrReflection);
}

// Since Spine 4.1, JSON exports nest deform timelines under "attachments",
// beside sequence timelines. The loader read only the Spine 4.0 "deform" map,
// so mesh deforms from current exports did not play.
#[test]
fn deform_timelines_under_attachments_apply() {
    let json = r#"{
        "bones": [ { "name": "root" } ],
        "slots": [ { "name": "s", "bone": "root", "attachment": "m" } ],
        "skins": [ { "name": "default", "attachments": { "s": { "m": {
            "type": "mesh", "uvs": [0,0, 1,0, 0,1], "triangles": [0,1,2], "vertices": [0,0, 10,0, 0,10]
        } } } } ],
        "animations": {
            "wobble": { "attachments": { "default": { "s": { "m": { "deform": [
                { "vertices": [0,0, 0,0, 0,0], "curve": "stepped" },
                { "time": 1, "offset": 2, "vertices": [5, 0] }
            ] } } } } }
        }
    }"#;
    let data = from_json(json).unwrap();
    assert_eq!(data.find_animation("wobble").unwrap().duration(), 1.0);
    let sk = posed(json, "wobble", 1.0);
    // Vertex 1's x (index 2) gets +5: setup 10 -> 15.
    assert_eq!(sk.slot(0).unwrap().deform, [0.0, 0.0, 15.0, 0.0, 0.0, 10.0]);
    // The first key is stepped, so it holds until the second.
    let sk = posed(json, "wobble", 0.5);
    assert_eq!(sk.slot(0).unwrap().deform, [0.0, 0.0, 10.0, 0.0, 0.0, 10.0]);
}

// Slider constraints in JSON were dropped: the loader parsed only the other
// constraint types, and skipped slider timelines for lack of a constraint.
#[test]
fn slider_constraints_load_and_scrub_their_animation() {
    let json = r#"{
        "bones": [ { "name": "root" }, { "name": "b", "parent": "root" } ],
        "constraints": [
            { "type": "slider", "name": "clock", "time": 0.5, "animation": "spin" },
            { "type": "slider", "name": "dial", "bone": "root", "property": "x",
              "from": 1, "to": 2, "scale": 3, "max": 4, "local": true, "additive": true,
              "mix": 0.5, "animation": "spin" }
        ],
        "animations": {
            "spin": { "bones": { "b": { "rotate": [ {}, { "time": 1, "value": 90 } ] } } },
            "still": { "slider": { "clock": { "mix": [ { "value": 0.25 } ] } } }
        }
    }"#;
    let data = from_json(json).unwrap();
    let spin = data.animations.iter().position(|a| a.name() == "spin");
    let clock = &data.sliders[0];
    assert_eq!(clock.time, 0.5);
    assert!(!clock.looping && clock.bone.is_none());
    assert_eq!(clock.animation_index, spin);
    let dial = &data.sliders[1];
    assert_eq!(dial.bone, Some(0));
    assert_eq!(dial.property, Some(SliderProperty::X));
    let values = [
        dial.property_offset,
        dial.offset,
        dial.scale,
        dial.max,
        dial.mix,
    ];
    assert_eq!(values, [1.0, 2.0, 3.0, 4.0, 0.5]);
    assert!(dial.local && dial.additive);

    // The bone-less slider scrubs "spin" to its setup time: 45 degrees.
    let mut sk = Skeleton::new(Arc::new(data));
    sk.update_world_transform();
    assert!((sk.bone(1).unwrap().rotation - 45.0).abs() < 1e-4);

    // Slider timelines now find their constraint.
    let mut sk = posed(json, "still", 0.0);
    let (pose, _) = sk.slider_pose_and_setup(0).unwrap();
    assert_eq!(pose.mix, 0.25);
}

// A slider must name an animation the export defines.
#[test]
fn slider_with_an_unknown_animation_is_rejected() {
    let json = r#"{
        "bones": [ { "name": "root" } ],
        "constraints": [ { "type": "slider", "name": "s", "animation": "missing" } ]
    }"#;
    assert!(matches!(from_json(json), Err(LoadError::BadReference(name)) if name == "missing"));
}
