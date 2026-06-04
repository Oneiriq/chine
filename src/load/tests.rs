use super::*;
use crate::attach::Attachment;

const JSON: &str = r#"{
    "skeleton": { "spine": "4.3.11", "x": -10, "y": 0, "width": 200, "height": 300 },
    "bones": [
        { "name": "root" },
        { "name": "torso", "parent": "root", "length": 100, "y": 20, "rotation": 90, "inherit": "noScale" }
    ],
    "slots": [
        { "name": "head", "bone": "torso", "attachment": "head", "color": "ff8800ff", "blend": "additive" }
    ],
    "skins": [
        {
            "name": "default",
            "attachments": {
                "head": {
                    "head": { "x": 5, "y": 6, "width": 40, "height": 50, "rotation": 15 },
                    "head-mesh": {
                        "type": "mesh",
                        "uvs": [0,0, 1,0, 0,1],
                        "triangles": [0,1,2],
                        "vertices": [10,10, 20,10, 10,20],
                        "hull": 3
                    }
                }
            }
        }
    ]
}"#;

#[test]
fn parses_header_bones_slots() {
    let data = from_json(JSON).unwrap();
    assert_eq!(data.spine_version.as_deref(), Some("4.3.11"));
    assert_eq!(data.size, Vec2::new(200.0, 300.0));

    assert_eq!(data.bones.len(), 2);
    let torso = &data.bones[data.find_bone("torso").unwrap()];
    assert_eq!(torso.parent, Some(0));
    assert!((torso.position.y - 20.0).abs() < 1e-6);
    assert!((torso.rotation - 90.0).abs() < 1e-6);
    assert_eq!(torso.inherit, Inherit::NoScale);
    // defaults filled where omitted.
    assert!((torso.scale.x - 1.0).abs() < 1e-6);

    let head = &data.slots[data.find_slot("head").unwrap()];
    assert_eq!(head.bone, 1);
    assert_eq!(head.blend, BlendMode::Additive);
    assert!((head.color.r - 1.0).abs() < 1e-3 && (head.color.g - 0.533).abs() < 1e-2);
    assert_eq!(head.attachment.as_deref(), Some("head"));
}

#[test]
fn parses_skin_region_and_mesh_attachments() {
    let data = from_json(JSON).unwrap();
    let slot = data.find_slot("head").unwrap();

    match data.attachment(slot, "head", None) {
        Some(Attachment::Region(r)) => {
            assert!((r.width - 40.0).abs() < 1e-6 && (r.rotation - 15.0).abs() < 1e-6);
            assert!((r.x - 5.0).abs() < 1e-6);
        }
        other => panic!("expected region, got {other:?}"),
    }

    match data.attachment(slot, "head-mesh", None) {
        Some(Attachment::Mesh(m)) => {
            assert_eq!(m.vertex_count(), 3);
            assert_eq!(m.triangles, vec![0, 1, 2]);
            assert_eq!(m.hull_length, 3);
        }
        other => panic!("expected mesh, got {other:?}"),
    }
}

#[test]
fn unknown_bone_reference_errors() {
    let bad = r#"{ "bones": [ { "name": "a", "parent": "ghost" } ] }"#;
    assert!(matches!(from_json(bad), Err(LoadError::BadReference(_))));
}

#[test]
fn parses_and_plays_a_rotate_animation() {
    let json = r#"{
        "bones": [ { "name": "root" }, { "name": "arm", "parent": "root" } ],
        "animations": {
            "wave": {
                "bones": {
                    "arm": {
                        "rotate": [ { "time": 0, "value": 0 }, { "time": 1, "value": 90 } ]
                    }
                }
            }
        }
    }"#;
    let data = from_json(json).unwrap();
    assert_eq!(data.animations.len(), 1);
    let anim = data.find_animation("wave").unwrap().clone();
    assert!((anim.duration() - 1.0).abs() < 1e-6);

    let arm = data.find_bone("arm").unwrap();
    let mut sk = crate::skel::Skeleton::new(std::sync::Arc::new(data));
    let mut state = crate::anim::AnimationState::new();
    state.set_animation(anim, false);
    state.update(0.5);
    state.apply(&mut sk);
    // setup rotation 0 + interpolated 45 at the halfway point.
    assert!((sk.bone(arm).unwrap().rotation - 45.0).abs() < 1e-4);
}

#[test]
fn parses_and_plays_a_slot_alpha_timeline() {
    let json = r#"{
        "bones": [ { "name": "root" } ],
        "slots": [ { "name": "s", "bone": "root" } ],
        "animations": {
            "fade": {
                "slots": {
                    "s": {
                        "alpha": [ { "time": 0, "value": 1 }, { "time": 1, "value": 0 } ]
                    }
                }
            }
        }
    }"#;
    let data = from_json(json).unwrap();
    let anim = data.find_animation("fade").unwrap().clone();
    let mut sk = crate::skel::Skeleton::new(std::sync::Arc::new(data));
    let mut state = crate::anim::AnimationState::new();
    state.set_animation(anim, false);
    state.update(0.5);
    state.apply(&mut sk);
    // setup alpha 1 fading to 0; ~0.5 at the halfway point.
    assert!((sk.slot(0).unwrap().color.a - 0.5).abs() < 1e-4);
}

#[test]
fn parses_ik_constraint() {
    let json = r#"{
        "bones": [
            { "name": "root" }, { "name": "thigh", "parent": "root" },
            { "name": "shin", "parent": "thigh" }, { "name": "goal", "parent": "root" }
        ],
        "constraints": [
            { "type": "ik", "name": "leg", "bones": ["thigh", "shin"],
              "target": "goal", "bendPositive": false, "mix": 0.8 }
        ]
    }"#;
    let data = from_json(json).unwrap();
    assert_eq!(data.ik_constraints.len(), 1);
    let c = &data.ik_constraints[0];
    assert_eq!(c.bones, vec![1, 2]);
    assert_eq!(c.target, 3);
    assert_eq!(c.bend_direction, -1);
    assert!((c.mix - 0.8).abs() < 1e-6);
}

#[test]
fn parses_transform_constraint() {
    let json = r#"{
        "bones": [
            { "name": "root" }, { "name": "src", "parent": "root" },
            { "name": "dst", "parent": "root" }
        ],
        "constraints": [
            { "type": "transform", "name": "follow", "bones": ["dst"], "source": "src",
              "mixRotate": 0.5, "properties": { "rotate": { "to": { "rotate": {} } } } }
        ]
    }"#;
    let data = from_json(json).unwrap();
    assert_eq!(data.transform_constraints.len(), 1);
    let t = &data.transform_constraints[0];
    assert_eq!(t.bones, vec![2]);
    assert_eq!(t.source, 1);
    assert_eq!(t.order, 0);
    assert!((t.mix_rotate - 0.5).abs() < 1e-6);
    assert_eq!(t.properties.len(), 1);
    assert_eq!(t.properties[0].to.len(), 1);
}

#[test]
fn parses_physics_constraint() {
    let json = r#"{
        "skeleton": { "referenceScale": 120 },
        "bones": [ { "name": "root" }, { "name": "tail", "parent": "root", "length": 30 } ],
        "constraints": [
            { "type": "physics", "name": "jiggle", "bone": "tail",
              "y": 1, "rotate": 1, "gravity": 2, "strength": 80, "damping": 0.9,
              "mass": 2, "fps": 120, "mix": 1 }
        ]
    }"#;
    let data = from_json(json).unwrap();
    assert!((data.reference_scale - 120.0).abs() < 1e-6);
    assert_eq!(data.physics_constraints.len(), 1);
    let p = &data.physics_constraints[0];
    assert_eq!(p.bone, 1);
    assert_eq!(p.order, 0);
    assert!((p.gravity - 2.0).abs() < 1e-6);
    assert!((p.strength - 80.0).abs() < 1e-6);
    // fps 120 -> step 1/120.
    assert!((p.step - 1.0 / 120.0).abs() < 1e-6);
    // mass 2 -> mass_inverse 0.5.
    assert!((p.mass_inverse - 0.5).abs() < 1e-6);
    // omitted inertia defaults to 0.5.
    assert!((p.inertia - 0.5).abs() < 1e-6);
}

#[test]
fn physics_strength_timeline_animates_the_constraint() {
    let json = r#"{
        "bones": [ { "name": "root" }, { "name": "tail", "parent": "root", "length": 20 } ],
        "constraints": [
            { "type": "physics", "name": "jiggle", "bone": "tail", "y": 1, "strength": 100 }
        ],
        "animations": {
            "soften": {
                "physics": {
                    "jiggle": {
                        "strength": [ { "time": 0, "value": 100 }, { "time": 1, "value": 20 } ]
                    }
                }
            }
        }
    }"#;
    let data = from_json(json).unwrap();
    let anim = data.find_animation("soften").unwrap().clone();
    let mut sk = crate::skel::Skeleton::new(std::sync::Arc::new(data));
    let mut state = crate::anim::AnimationState::new();
    state.set_animation(anim, false);
    // Halfway through, strength interpolates 100 -> 20, i.e. 60.
    state.update(0.5);
    state.apply(&mut sk);
    let s = sk.physics_constraint(0).unwrap().strength;
    assert!((s - 60.0).abs() < 1e-3, "strength={s}");
}

#[test]
fn global_physics_timeline_drives_flagged_constraints() {
    let json = r#"{
        "bones": [
            { "name": "root" },
            { "name": "a", "parent": "root", "length": 10 },
            { "name": "b", "parent": "root", "length": 10 }
        ],
        "constraints": [
            { "type": "physics", "name": "pa", "bone": "a", "y": 1, "strength": 100, "strengthGlobal": true },
            { "type": "physics", "name": "pb", "bone": "b", "y": 1, "strength": 100 }
        ],
        "animations": {
            "soften": {
                "physics": {
                    "": { "strength": [ { "time": 0, "value": 40 }, { "time": 1, "value": 40 } ] }
                }
            }
        }
    }"#;
    let data = from_json(json).unwrap();
    assert_eq!(data.physics_constraints.len(), 2);
    assert!(data.physics_constraints[0].strength_global);
    assert!(!data.physics_constraints[1].strength_global);
    let anim = data.find_animation("soften").unwrap().clone();
    let mut sk = crate::skel::Skeleton::new(std::sync::Arc::new(data));
    let mut state = crate::anim::AnimationState::new();
    state.set_animation(anim, false);
    state.update(0.5);
    state.apply(&mut sk);
    // The global strength timeline (value 40) drives only "pa"
    // (strengthGlobal); "pb" keeps its setup strength (100).
    assert!((sk.physics_constraint(0).unwrap().strength - 40.0).abs() < 1e-3);
    assert!((sk.physics_constraint(1).unwrap().strength - 100.0).abs() < 1e-3);
}

#[test]
fn physics_reset_timeline_zeroes_the_offset() {
    let json = r#"{
        "skeleton": { "referenceScale": 100 },
        "bones": [ { "name": "root" }, { "name": "tail", "parent": "root", "length": 20 } ],
        "constraints": [
            { "type": "physics", "name": "p", "bone": "tail",
              "y": 1, "gravity": 1, "strength": 50, "damping": 0.9, "mass": 1, "fps": 60 }
        ],
        "animations": {
            "blink": { "physics": { "p": { "reset": [ { "time": 0.5 } ] } } }
        }
    }"#;
    let data = from_json(json).unwrap();
    let anim = data.find_animation("blink").unwrap().clone();
    let mut sk = crate::skel::Skeleton::new(std::sync::Arc::new(data));
    let mut state = crate::anim::AnimationState::new();
    state.set_animation(anim, false);
    let dt = 1.0 / 60.0;
    let step = |state: &mut crate::anim::AnimationState, sk: &mut crate::skel::Skeleton| {
        state.update(dt);
        sk.update(dt);
        sk.set_bones_to_setup_pose();
        state.apply(sk);
        sk.update_world_transform();
    };
    // Droop for ~0.48s, before the reset keyframe at t=0.5.
    for _ in 0..29 {
        step(&mut state, &mut sk);
    }
    let drooped = sk.bone(1).unwrap().world_y();
    assert!(drooped < -0.2, "expected droop before reset, got {drooped}");
    // Two more frames cross t=0.5 and fire the reset, undoing the droop.
    step(&mut state, &mut sk);
    step(&mut state, &mut sk);
    let after = sk.bone(1).unwrap().world_y();
    assert!(
        after.abs() < 0.1,
        "expected reset to undo the droop, got {after}"
    );
}

#[test]
fn slot_color_timeline_animates_the_tint() {
    let json = r#"{
        "bones": [ { "name": "root" } ],
        "slots": [ { "name": "s", "bone": "root", "attachment": "a", "color": "ffffffff" } ],
        "animations": {
            "fade": {
                "slots": {
                    "s": { "rgba": [ { "time": 0, "color": "ffffffff" }, { "time": 1, "color": "ff000000" } ] }
                }
            }
        }
    }"#;
    let data = from_json(json).unwrap();
    let anim = data.find_animation("fade").unwrap().clone();
    let mut sk = crate::skel::Skeleton::new(std::sync::Arc::new(data));
    let mut state = crate::anim::AnimationState::new();
    state.set_animation(anim, false);
    // Halfway, white (1,1,1,1) -> opaque-red-faded (1,0,0,0): (1, 0.5, 0.5, 0.5).
    state.update(0.5);
    sk.set_slots_to_setup_pose();
    state.apply(&mut sk);
    let c = sk.slot(0).unwrap().color;
    assert!((c.r - 1.0).abs() < 1e-2, "r={}", c.r);
    assert!((c.g - 0.5).abs() < 1e-2, "g={}", c.g);
    assert!((c.a - 0.5).abs() < 1e-2, "a={}", c.a);
}

#[test]
fn slot_rgb_timeline_preserves_alpha() {
    let json = r#"{
        "bones": [ { "name": "root" } ],
        "slots": [ { "name": "s", "bone": "root", "attachment": "a", "color": "ffffff80" } ],
        "animations": {
            "tint": {
                "slots": { "s": { "rgb": [ { "time": 0, "color": "ffffff" }, { "time": 1, "color": "ff0000" } ] } }
            }
        }
    }"#;
    let data = from_json(json).unwrap();
    let anim = data.find_animation("tint").unwrap().clone();
    let mut sk = crate::skel::Skeleton::new(std::sync::Arc::new(data));
    let mut state = crate::anim::AnimationState::new();
    state.set_animation(anim, false);
    state.update(0.5);
    sk.set_slots_to_setup_pose();
    state.apply(&mut sk);
    let c = sk.slot(0).unwrap().color;
    // rgb interpolates white -> red (g 0.5); alpha stays the setup 0x80 (~0.502).
    assert!((c.g - 0.5).abs() < 1e-2, "g={}", c.g);
    assert!((c.a - 0.502).abs() < 1e-2, "a={}", c.a);
}

#[test]
fn event_timeline_fires_events() {
    let json = r#"{
        "bones": [ { "name": "root" } ],
        "events": { "footstep": { "int": 5 } },
        "animations": {
            "walk": { "events": [ { "time": 0.5, "name": "footstep", "int": 7 } ] }
        }
    }"#;
    let data = from_json(json).unwrap();
    assert_eq!(data.events.len(), 1);
    let anim = data.find_animation("walk").unwrap().clone();
    let mut sk = crate::skel::Skeleton::new(std::sync::Arc::new(data));
    let mut state = crate::anim::AnimationState::new();
    state.set_animation(anim, false);
    // Before the keyframe at 0.5, nothing fires.
    state.update(0.3);
    state.apply(&mut sk);
    assert!(sk.events().is_empty());
    // Crossing 0.5 fires the footstep with the keyframe int override (7).
    state.update(0.3);
    state.apply(&mut sk);
    assert_eq!(sk.events().len(), 1);
    assert_eq!(sk.events()[0].name, "footstep");
    assert_eq!(sk.events()[0].int_value, 7);
    // The next apply clears it.
    state.update(0.3);
    state.apply(&mut sk);
    assert!(sk.events().is_empty());
}

#[test]
fn single_axis_bone_timelines() {
    let json = r#"{
        "bones": [ { "name": "b" } ],
        "animations": {
            "anim": {
                "bones": {
                    "b": {
                        "translatex": [ { "time": 0, "value": 0 }, { "time": 1, "value": 10 } ],
                        "scaley": [ { "time": 0, "value": 1 }, { "time": 1, "value": 3 } ],
                        "shearx": [ { "time": 0, "value": 0 }, { "time": 1, "value": 20 } ]
                    }
                }
            }
        }
    }"#;
    let data = from_json(json).unwrap();
    let anim = data.find_animation("anim").unwrap().clone();
    let mut sk = crate::skel::Skeleton::new(std::sync::Arc::new(data));
    let mut state = crate::anim::AnimationState::new();
    state.set_animation(anim, false);
    state.update(0.5);
    sk.set_bones_to_setup_pose();
    state.apply(&mut sk);
    let b = sk.bone(0).unwrap();
    // translatex 0->10 at half = 5; scaley 1->3 = 2; shearx 0->20 = 10.
    assert!((b.x - 5.0).abs() < 1e-3, "x={}", b.x);
    assert!((b.scale_y - 2.0).abs() < 1e-3, "scale_y={}", b.scale_y);
    assert!((b.shear_x - 10.0).abs() < 1e-3, "shear_x={}", b.shear_x);
}

#[test]
fn shear_bone_timeline() {
    let json = r#"{
        "bones": [ { "name": "b" } ],
        "animations": {
            "anim": {
                "bones": {
                    "b": { "shear": [ { "time": 0, "x": 0, "y": 0 }, { "time": 1, "x": 30, "y": 40 } ] }
                }
            }
        }
    }"#;
    let data = from_json(json).unwrap();
    let anim = data.find_animation("anim").unwrap().clone();
    let mut sk = crate::skel::Skeleton::new(std::sync::Arc::new(data));
    let mut state = crate::anim::AnimationState::new();
    state.set_animation(anim, false);
    state.update(0.5);
    sk.set_bones_to_setup_pose();
    state.apply(&mut sk);
    let b = sk.bone(0).unwrap();
    assert!((b.shear_x - 15.0).abs() < 1e-3, "shear_x={}", b.shear_x);
    assert!((b.shear_y - 20.0).abs() < 1e-3, "shear_y={}", b.shear_y);
}

#[test]
fn deform_timeline_moves_mesh_vertices() {
    let json = r#"{
        "bones": [ { "name": "root" } ],
        "slots": [ { "name": "s", "bone": "root", "attachment": "m" } ],
        "skins": [ { "name": "default", "attachments": { "s": { "m": {
            "type": "mesh", "uvs": [0,0, 1,0, 0,1], "triangles": [0,1,2], "vertices": [0,0, 10,0, 0,10]
        } } } } ],
        "animations": {
            "wobble": {
                "deform": { "default": { "s": { "m": [
                    { "time": 0, "vertices": [0,0, 0,0, 0,0] },
                    { "time": 1, "offset": 2, "vertices": [5, 0] }
                ] } } }
            }
        }
    }"#;
    let data = from_json(json).unwrap();
    let anim = data.find_animation("wobble").unwrap().clone();
    let mut sk = crate::skel::Skeleton::new(std::sync::Arc::new(data));
    let mut state = crate::anim::AnimationState::new();
    state.set_animation(anim, false);
    state.update(1.0);
    sk.set_slots_to_setup_pose();
    state.apply(&mut sk);
    let d = &sk.slot(0).unwrap().deform;
    // vertex 1's x (index 2) gets +5: setup 10 -> 15; others unchanged.
    assert_eq!(d.len(), 6);
    assert!((d[2] - 15.0).abs() < 1e-3, "d[2]={}", d[2]);
    assert!((d[0] - 0.0).abs() < 1e-3, "d[0]={}", d[0]);
    assert!((d[5] - 10.0).abs() < 1e-3, "d[5]={}", d[5]);
}

#[test]
fn stepped_deform_snaps_to_the_earlier_frame() {
    let json = r#"{
        "bones": [ { "name": "root" } ],
        "slots": [ { "name": "s", "bone": "root", "attachment": "m" } ],
        "skins": [ { "name": "default", "attachments": { "s": { "m": {
            "type": "mesh", "uvs": [0,0], "triangles": [], "vertices": [2, 3]
        } } } } ],
        "animations": {
            "snap": {
                "deform": { "default": { "s": { "m": [
                    { "time": 0, "vertices": [0, 0], "curve": "stepped" },
                    { "time": 1, "vertices": [10, 0] }
                ] } } }
            }
        }
    }"#;
    let data = from_json(json).unwrap();
    let anim = data.find_animation("snap").unwrap().clone();
    let mut sk = crate::skel::Skeleton::new(std::sync::Arc::new(data));
    let mut state = crate::anim::AnimationState::new();
    state.set_animation(anim, false);
    state.update(0.5);
    sk.set_slots_to_setup_pose();
    state.apply(&mut sk);
    let d = &sk.slot(0).unwrap().deform;
    // Stepped: at t=0.5 the deform holds frame 0 (offset 0), so it equals the
    // setup [2, 3] rather than the linear midpoint [7, 3].
    assert_eq!(d.len(), 2);
    assert!((d[0] - 2.0).abs() < 1e-3, "d[0]={}", d[0]);
    assert!((d[1] - 3.0).abs() < 1e-3, "d[1]={}", d[1]);
}

#[test]
fn weighted_deform_offsets_vertices() {
    // A weighted mesh: 1 vertex, 1 influence (bone 0, bind at origin).
    let json = r#"{
        "bones": [ { "name": "root" } ],
        "slots": [ { "name": "s", "bone": "root", "attachment": "m" } ],
        "skins": [ { "name": "default", "attachments": { "s": { "m": {
            "type": "mesh", "uvs": [0,0], "triangles": [], "vertices": [1, 0, 0, 0, 1]
        } } } } ],
        "animations": {
            "wob": {
                "deform": { "default": { "s": { "m": [
                    { "time": 0, "vertices": [0, 0] },
                    { "time": 1, "vertices": [3, 4] }
                ] } } }
            }
        }
    }"#;
    let data = from_json(json).unwrap();
    let anim = data.find_animation("wob").unwrap().clone();
    let mut sk = crate::skel::Skeleton::new(std::sync::Arc::new(data));
    let mut state = crate::anim::AnimationState::new();
    state.set_animation(anim, false);
    state.update(1.0);
    sk.set_slots_to_setup_pose();
    state.apply(&mut sk);
    sk.update_world_transform();
    // Weighted deform stores the per-influence offsets (not setup + offset).
    let d = sk.slot(0).unwrap().deform.clone();
    assert_eq!(d, vec![3.0, 4.0]);
    // The world vertex = bind (0,0) + offset (3,4) at the identity root.
    let crate::attach::Attachment::Mesh(m) = sk.data().attachment(0, "m", None).unwrap() else {
        panic!("expected mesh");
    };
    let w = m.compute_world_vertices(&sk, 0, &d);
    assert!(
        (w[0].x - 3.0).abs() < 1e-3 && (w[0].y - 4.0).abs() < 1e-3,
        "w0={:?}",
        w[0]
    );
}

#[test]
fn two_color_timeline_animates_light_and_dark() {
    let json = r#"{
        "bones": [ { "name": "root" } ],
        "slots": [ { "name": "s", "bone": "root", "attachment": "a", "color": "ffffffff", "dark": "000000" } ],
        "animations": {
            "tc": {
                "slots": { "s": { "rgba2": [
                    { "time": 0, "light": "ffffffff", "dark": "000000" },
                    { "time": 1, "light": "ff0000ff", "dark": "00ff00" }
                ] } }
            }
        }
    }"#;
    let data = from_json(json).unwrap();
    let anim = data.find_animation("tc").unwrap().clone();
    let mut sk = crate::skel::Skeleton::new(std::sync::Arc::new(data));
    let mut state = crate::anim::AnimationState::new();
    state.set_animation(anim, false);
    state.update(0.5);
    sk.set_slots_to_setup_pose();
    state.apply(&mut sk);
    let slot = sk.slot(0).unwrap();
    // light white -> red (g 1->0.5); dark black -> green (g 0->0.5).
    assert!(
        (slot.color.g - 0.5).abs() < 1e-2,
        "light g={}",
        slot.color.g
    );
    let dark = slot.dark_color.unwrap();
    assert!((dark.g - 0.5).abs() < 1e-2, "dark g={}", dark.g);
    assert!((dark.r - 0.0).abs() < 1e-2, "dark r={}", dark.r);
}

#[test]
fn attachment_timeline_swaps_the_attachment() {
    let json = r#"{
        "bones": [ { "name": "root" } ],
        "slots": [ { "name": "s", "bone": "root", "attachment": "a" } ],
        "animations": {
            "swap": {
                "slots": {
                    "s": { "attachment": [ { "time": 0, "name": "a" }, { "time": 1, "name": "b" } ] }
                }
            }
        }
    }"#;
    let data = from_json(json).unwrap();
    let anim = data.find_animation("swap").unwrap().clone();
    let mut sk = crate::skel::Skeleton::new(std::sync::Arc::new(data));
    let mut state = crate::anim::AnimationState::new();
    state.set_animation(anim, false);
    // t=0.5 is before the second key, so the attachment is still "a" (stepped).
    state.update(0.5);
    sk.set_slots_to_setup_pose();
    state.apply(&mut sk);
    assert_eq!(sk.slot(0).unwrap().attachment.as_deref(), Some("a"));
    // Past t=1.0 the attachment switches to "b".
    state.update(0.6);
    sk.set_slots_to_setup_pose();
    state.apply(&mut sk);
    assert_eq!(sk.slot(0).unwrap().attachment.as_deref(), Some("b"));
}

#[test]
fn draw_order_timeline_reorders_slots() {
    let json = r#"{
        "bones": [ { "name": "root" } ],
        "slots": [
            { "name": "s0", "bone": "root" },
            { "name": "s1", "bone": "root" },
            { "name": "s2", "bone": "root" }
        ],
        "animations": {
            "reorder": {
                "drawOrder": [
                    { "time": 0 },
                    { "time": 1, "offsets": [ { "slot": "s0", "offset": 2 } ] }
                ]
            }
        }
    }"#;
    let data = from_json(json).unwrap();
    let anim = data.find_animation("reorder").unwrap().clone();
    let mut sk = crate::skel::Skeleton::new(std::sync::Arc::new(data));
    let mut state = crate::anim::AnimationState::new();
    state.set_animation(anim, false);
    // t=0: identity draw order.
    state.update(0.0);
    sk.set_slots_to_setup_pose();
    state.apply(&mut sk);
    assert_eq!(sk.draw_order(), &[0, 1, 2]);
    // t>=1: slot 0 moves two places back, others shift forward.
    state.update(1.0);
    sk.set_slots_to_setup_pose();
    state.apply(&mut sk);
    assert_eq!(sk.draw_order(), &[1, 2, 0]);
}

#[test]
fn ik_mix_timeline_fades_the_constraint() {
    let json = r#"{
        "bones": [
            { "name": "root" }, { "name": "aim", "parent": "root", "length": 10 },
            { "name": "target", "parent": "root", "y": 10 }
        ],
        "constraints": [
            { "type": "ik", "name": "aim-ik", "bones": ["aim"], "target": "target", "mix": 1 }
        ],
        "animations": {
            "fade": { "ik": { "aim-ik": [ { "time": 0, "mix": 0 }, { "time": 1, "mix": 1 } ] } }
        }
    }"#;
    let data = from_json(json).unwrap();
    let aim = data.find_bone("aim").unwrap();
    let anim = data.find_animation("fade").unwrap().clone();
    let mut sk = crate::skel::Skeleton::new(std::sync::Arc::new(data));
    let mut state = crate::anim::AnimationState::new();
    state.set_animation(anim, false);

    // time 0: IK mix 0 -> the constraint does nothing, aim keeps +x (a~1).
    state.apply(&mut sk);
    sk.update_world_transform();
    assert!(
        (sk.bone(aim).unwrap().a() - 1.0).abs() < 1e-2,
        "a={}",
        sk.bone(aim).unwrap().a()
    );

    // time 1: IK mix 1 -> aim is driven to point at the target (+y): a~0, c~1.
    state.update(1.0);
    sk.set_bones_to_setup_pose();
    state.apply(&mut sk);
    sk.update_world_transform();
    let aim_bone = sk.bone(aim).unwrap();
    assert!(aim_bone.a().abs() < 1e-2, "a={}", aim_bone.a());
    assert!((aim_bone.c() - 1.0).abs() < 1e-2, "c={}", aim_bone.c());
}

#[test]
fn parses_boundingbox_and_point_attachments() {
    let bb: Value = serde_json::from_str(
        r#"{"type":"boundingbox","vertexCount":3,"vertices":[0,0,10,0,10,10]}"#,
    )
    .unwrap();
    match parse_attachment("hit", &bb) {
        Some(Attachment::BoundingBox(b)) => assert_eq!(b.vertex_count(), 3),
        other => panic!("expected boundingbox, got {other:?}"),
    }
    let pt: Value =
        serde_json::from_str(r#"{"type":"point","x":5,"y":6,"rotation":30}"#).unwrap();
    match parse_attachment("muzzle", &pt) {
        Some(Attachment::Point(p)) => {
            assert!((p.x - 5.0).abs() < 1e-4);
            assert!((p.y - 6.0).abs() < 1e-4);
            assert!((p.rotation - 30.0).abs() < 1e-4);
        }
        other => panic!("expected point, got {other:?}"),
    }
}

#[test]
fn parses_clipping_attachment() {
    let v: Value = serde_json::from_str(
        r#"{"type":"clipping","end":"mouth","vertexCount":3,"vertices":[0,0,10,0,10,10]}"#,
    )
    .unwrap();
    match parse_attachment("mask", &v) {
        Some(Attachment::Clipping(c)) => {
            assert_eq!(c.vertex_count(), 3);
            assert_eq!(c.end_slot, "mouth");
        }
        other => panic!("expected clipping, got {other:?}"),
    }
}

#[test]
fn resolves_linked_mesh_to_parent_geometry() {
    let json = r#"{
        "skeleton": { "spine": "4.3" },
        "bones": [ { "name": "root" } ],
        "slots": [ { "name": "wing", "bone": "root" } ],
        "skins": [
            { "name": "default", "attachments": { "wing": { "wing": {
                "type": "mesh", "uvs": [0,0, 1,0, 0,1], "triangles": [0,1,2],
                "vertices": [0,0, 10,0, 0,10], "hull": 3
            } } } },
            { "name": "blue", "attachments": { "wing": { "wing-blue": {
                "type": "linkedmesh", "skin": "default", "parent": "wing"
            } } } }
        ]
    }"#;
    let data = from_json(json).unwrap();
    let blue = data.skins.iter().find(|s| s.name == "blue").unwrap();
    match blue.attachment(0, "wing-blue") {
        // The link resolved to a concrete mesh borrowing the parent's
        // geometry, while keeping its own name.
        Some(Attachment::Mesh(m)) => {
            assert_eq!(m.triangles, vec![0, 1, 2]);
            assert_eq!(m.vertex_count(), 3);
            assert_eq!(m.hull_length, 3);
            assert_eq!(m.name, "wing-blue");
        }
        other => panic!("expected resolved mesh, got {other:?}"),
    }
}

#[test]
fn linked_mesh_inherits_parent_deform_under_active_skin() {
    // The slot's "m" placeholder is a mesh in the default skin and a linked
    // mesh (own red tint) in "alt"; a deform animation targets "m". With the
    // alt skin active, render shows the link and the parent's deform drives
    // it - purely through name matching, no deform-source indirection.
    let json = r#"{
        "bones": [ { "name": "root" } ],
        "slots": [ { "name": "s", "bone": "root", "attachment": "m" } ],
        "skins": [
            { "name": "default", "attachments": { "s": { "m": {
                "type": "mesh", "uvs": [0,0, 1,0, 0,1], "triangles": [0,1,2],
                "vertices": [0,0, 10,0, 0,10]
            } } } },
            { "name": "alt", "attachments": { "s": { "m": {
                "type": "linkedmesh", "skin": "default", "parent": "m", "color": "ff0000ff"
            } } } }
        ],
        "animations": {
            "flap": {
                "deform": { "default": { "s": { "m": [
                    { "time": 0, "vertices": [0,0, 0,0, 0,0] },
                    { "time": 1, "offset": 2, "vertices": [5, 0] }
                ] } } }
            }
        }
    }"#;
    let data = from_json(json).unwrap();
    let anim = data.find_animation("flap").unwrap().clone();
    let mut sk = crate::skel::Skeleton::new(std::sync::Arc::new(data));
    sk.set_skin("alt");
    let mut state = crate::anim::AnimationState::new();
    state.set_animation(anim, false);
    state.update(1.0);
    sk.set_slots_to_setup_pose();
    state.apply(&mut sk);
    sk.update_world_transform();
    let cmds = crate::render::render(&sk);
    assert_eq!(cmds.len(), 1);
    let c = &cmds[0];
    // The alt skin's linked mesh is the one rendered (its own red tint)...
    assert!(c.color.r > 0.9 && c.color.g < 0.1, "color={:?}", c.color);
    // ...and the parent's deform drives it: vertex 1's x goes 10 -> 15.
    assert!(
        (c.positions[1].x - 15.0).abs() < 1e-3,
        "x={}",
        c.positions[1].x
    );
}

#[test]
fn deform_is_skin_aware() {
    // Slot "s" placeholder "m" is a different mesh in each skin, each with
    // its own deform under that skin. The active skin selects which applies.
    let json = r#"{
        "bones": [ { "name": "root" } ],
        "slots": [ { "name": "s", "bone": "root", "attachment": "m" } ],
        "skins": [
            { "name": "default", "attachments": { "s": { "m": {
                "type": "mesh", "uvs": [0,0, 1,0, 0,1], "triangles": [0,1,2],
                "vertices": [0,0, 10,0, 0,10]
            } } } },
            { "name": "alt", "attachments": { "s": { "m": {
                "type": "mesh", "uvs": [0,0, 1,0, 0,1], "triangles": [0,1,2],
                "vertices": [0,0, 20,0, 0,20]
            } } } }
        ],
        "animations": {
            "flap": {
                "deform": {
                    "default": { "s": { "m": [
                        { "time": 0, "vertices": [0,0, 0,0, 0,0] },
                        { "time": 1, "offset": 2, "vertices": [5, 0] }
                    ] } },
                    "alt": { "s": { "m": [
                        { "time": 0, "vertices": [0,0, 0,0, 0,0] },
                        { "time": 1, "offset": 2, "vertices": [100, 0] }
                    ] } }
                }
            }
        }
    }"#;
    let data = from_json(json).unwrap();
    let anim = data.find_animation("flap").unwrap().clone();
    let mut sk = crate::skel::Skeleton::new(std::sync::Arc::new(data));
    let mut state = crate::anim::AnimationState::new();
    state.set_animation(anim, false);
    state.update(1.0);

    // Default skin: only the default deform applies (vtx1.x 10 -> 15).
    sk.set_slots_to_setup_pose();
    state.apply(&mut sk);
    sk.update_world_transform();
    let c = crate::render::render(&sk);
    assert!(
        (c[0].positions[1].x - 15.0).abs() < 1e-3,
        "default x={}",
        c[0].positions[1].x
    );

    // Alt skin: only the alt deform applies (vtx1.x 20 -> 120), not 15.
    sk.set_skin("alt");
    sk.set_slots_to_setup_pose();
    state.apply(&mut sk);
    sk.update_world_transform();
    let c = crate::render::render(&sk);
    assert!(
        (c[0].positions[1].x - 120.0).abs() < 1e-3,
        "alt x={}",
        c[0].positions[1].x
    );
}

#[test]
fn non_inheriting_link_uses_its_own_deform() {
    // alt's "m" is a link with "deform": false, so it does NOT inherit the
    // parent's deform; the alt-authored deform drives it (its own load no
    // longer skipped, since resolution runs before animation parsing).
    let json = r#"{
        "bones": [ { "name": "root" } ],
        "slots": [ { "name": "s", "bone": "root", "attachment": "m" } ],
        "skins": [
            { "name": "default", "attachments": { "s": { "m": {
                "type": "mesh", "uvs": [0,0, 1,0, 0,1], "triangles": [0,1,2],
                "vertices": [0,0, 10,0, 0,10]
            } } } },
            { "name": "alt", "attachments": { "s": { "m": {
                "type": "linkedmesh", "skin": "default", "parent": "m", "deform": false
            } } } }
        ],
        "animations": {
            "flap": {
                "deform": {
                    "default": { "s": { "m": [
                        { "time": 0, "vertices": [0,0, 0,0, 0,0] },
                        { "time": 1, "offset": 2, "vertices": [5, 0] }
                    ] } },
                    "alt": { "s": { "m": [
                        { "time": 0, "vertices": [0,0, 0,0, 0,0] },
                        { "time": 1, "offset": 2, "vertices": [50, 0] }
                    ] } }
                }
            }
        }
    }"#;
    let data = from_json(json).unwrap();
    let anim = data.find_animation("flap").unwrap().clone();
    let mut sk = crate::skel::Skeleton::new(std::sync::Arc::new(data));
    sk.set_skin("alt");
    let mut state = crate::anim::AnimationState::new();
    state.set_animation(anim, false);
    state.update(1.0);
    sk.set_slots_to_setup_pose();
    state.apply(&mut sk);
    sk.update_world_transform();
    let c = crate::render::render(&sk);
    // The link shares the parent geometry (base x 10) but uses the alt deform
    // (+50 -> 60), not the parent's (+5 -> 15).
    assert!(
        (c[0].positions[1].x - 60.0).abs() < 1e-3,
        "x={}",
        c[0].positions[1].x
    );
}
