use super::timelines::*;
use super::*;

#[test]
fn reads_var_uint_multi_byte() {
    // 300 = 0b1_0010_1100 -> low 7 bits 0x2C with continuation, then 0x02.
    let mut r = BinaryReader::new(&[0xAC, 0x02]);
    assert_eq!(r.var_uint(), 300);
    assert!(!r.overran());
}

#[test]
fn reads_float_big_endian() {
    let bytes = 1.0_f32.to_be_bytes();
    let mut r = BinaryReader::new(&bytes);
    assert!((r.float() - 1.0).abs() < 1e-6);
}

#[test]
fn reads_length_prefixed_strings() {
    // 0 -> None, 1 -> "", else n-1 bytes.
    let mut r = BinaryReader::new(&[0x00, 0x01, 0x03, b'h', b'i']);
    assert_eq!(r.string(), None);
    assert_eq!(r.string(), Some(String::new()));
    assert_eq!(r.string(), Some("hi".to_string()));
}

#[test]
fn flags_truncation() {
    let mut r = BinaryReader::new(&[0x05, b'h', b'i']); // claims 4 bytes, has 2
    let _ = r.string();
    assert!(r.overran());
}

fn enc_str(out: &mut Vec<u8>, s: &str) {
    out.push((s.len() + 1) as u8); // short names only
    out.extend_from_slice(s.as_bytes());
}

#[test]
fn parses_header_and_bones() {
    let mut b = Vec::new();
    b.extend_from_slice(&[0; 8]); // hash
    enc_str(&mut b, "4.3.00"); // version
    for v in [0.0_f32, 0.0, 200.0, 300.0, 1.0] {
        b.extend_from_slice(&v.to_be_bytes()); // x, y, width, height, referenceScale
    }
    b.push(0); // nonessential = false
    b.push(0); // string table count = 0
    b.push(1); // bone count = 1
    enc_str(&mut b, "root");
    // rotation, x, y, scaleX, scaleY, shearX, shearY, length
    for v in [0.0_f32, 10.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0] {
        b.extend_from_slice(&v.to_be_bytes());
    }
    b.push(3); // inherit = NoScale
    b.push(0); // skinRequired = false
    b.push(0); // slot count = 0
    b.push(0); // constraint count = 0
    b.push(0); // default skin slot count = 0
    b.push(0); // named skin count = 0
    b.push(0); // event count = 0
    b.push(0); // animation count = 0

    let data = from_binary(&b).unwrap();
    assert_eq!(data.spine_version.as_deref(), Some("4.3.00"));
    assert!((data.reference_scale - 1.0).abs() < 1e-6);
    assert_eq!(data.size, Vec2::new(200.0, 300.0));
    assert_eq!(data.bones.len(), 1);
    let root = &data.bones[0];
    assert_eq!(root.name, "root");
    assert_eq!(root.parent, None);
    assert!((root.position.x - 10.0).abs() < 1e-6);
    assert!((root.scale.x - 1.0).abs() < 1e-6);
    assert_eq!(root.inherit, Inherit::NoScale);
}

// Validates the parser against a real Spine 4.3 `.skel` when the local
// fixture is present (it is not committed), and skips cleanly otherwise.
#[cfg_attr(
    not(skel_fixtures),
    ignore = "requires the gitignored Spine fixtures in data/"
)]
#[test]
fn parses_real_skel_header_and_bones() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/data/Spine.skel");
    let Ok(bytes) = std::fs::read(path) else {
        return;
    };
    let data = from_binary(&bytes).unwrap();
    assert_eq!(data.spine_version.as_deref(), Some("4.3.13"));
    assert_eq!(data.bones.len(), 40);
    assert_eq!(data.bones[0].name, "root");
    assert_eq!(data.bones[0].parent, None);
    assert_eq!(data.bones[1].name, "skeleton-control");
    assert_eq!(data.bones[1].parent, Some(0));
    // Every bone has a non-empty name and a valid parent index.
    for (i, b) in data.bones.iter().enumerate() {
        assert!(!b.name.is_empty(), "bone {i} has empty name");
        if let Some(p) = b.parent {
            assert!(p < data.bones.len(), "bone {i} bad parent {p}");
        }
    }

    // Slots: draw order back-to-front, each on a valid bone.
    assert_eq!(data.slots.len(), 32);
    assert_eq!(data.slots[0].name, "foot-back");
    for s in &data.slots {
        assert!(!s.name.is_empty());
        assert!(
            s.bone < data.bones.len(),
            "slot {} bad bone {}",
            s.name,
            s.bone
        );
    }
    // Setup attachments resolve through the string table.
    assert!(data.slots.iter().any(|s| s.attachment.is_some()));

    // IK constraints (the rig's three foot/leg IK chains).
    assert_eq!(data.ik_constraints.len(), 3);
    let ik = &data.ik_constraints[0];
    assert_eq!(ik.name, "leg-front-IK");
    assert_eq!(ik.bones, vec![3, 4]);
    assert_eq!(ik.target, 37);
    for c in &data.ik_constraints {
        assert!((c.mix - 1.0).abs() < 1e-6, "{} mix={}", c.name, c.mix);
        assert!(c.target < data.bones.len());
        assert!(c.bones.iter().all(|&b| b < data.bones.len()));
    }

    // Default skin: one attachment per slot, meshes with valid triangles.
    assert_eq!(data.default_skin.iter().count(), 32);
    assert!(data.skins.is_empty());
    let mesh = data
        .default_skin
        .iter()
        .find_map(|(_, _, a)| match a {
            Attachment::Mesh(m) => Some(m),
            _ => None,
        })
        .expect("a mesh attachment");
    assert!(mesh.vertex_count() > 0);
    assert!(mesh
        .triangles
        .iter()
        .all(|&t| (t as usize) < mesh.vertex_count()));

    // One animation (empty in this WIP rig) named "animation".
    assert_eq!(data.animations.len(), 1);
    assert_eq!(data.animations[0].name(), "animation");
}

// Validates the binary loader against a complete Spine 4.3 project (the
// diamond rig) when the local fixture is present, and skips otherwise. Its
// non-bone animation timelines are still being added, so this asserts the
// structural pieces that are wired up: bones, slots, the new 4.3 slider
// constraint, and the first (bone-only) animation parsing in full.
#[cfg_attr(
    not(skel_fixtures),
    ignore = "requires the gitignored Spine fixtures in data/"
)]
#[test]
fn parses_diamond_rig() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/data/diamond-pro.skel");
    let Ok(bytes) = std::fs::read(path) else {
        return;
    };
    let data = from_binary(&bytes).expect("diamond parses without overrun");
    assert_eq!(data.bones.len(), 8);
    assert_eq!(data.slots.len(), 30);

    // The new 4.3 slider constraint ("rotation") drives a bone property.
    assert_eq!(data.sliders.len(), 1);
    let slider = &data.sliders[0];
    assert_eq!(slider.name, "rotation");
    assert!(slider.bone.is_some());
    assert!(slider.property.is_some());
    // The slider scrubs the like-named "rotation" animation.
    let scrubbed = slider.animation_index.expect("slider animation index");
    assert_eq!(data.animations[scrubbed].name(), "rotation");

    // All eight animations parse end to end. A misaligned parse would
    // surface as a garbage or truncated name. "appear" is bone-only.
    // "disappear" exercises slot color, attachment, slider, deform, and
    // sequence timelines.
    let names: Vec<&str> = data.animations.iter().map(|a| a.name()).collect();
    assert_eq!(
        names,
        [
            "appear",
            "disappear",
            "idle-rotating",
            "idle-rotating-alt-shape",
            "idle-still",
            "rotation",
            "size-changing-rotation",
            "size-changing-rotation-perspective",
        ]
    );
    assert!(data.animations[0].duration() > 0.0);
    assert!(data.animations[1].duration() > 0.0);
}

// The "disappear" animation deforms the diamond mesh. Loading it from binary
// must build a real deform timeline (not silently fall back to consuming the
// bytes). Playing it populates a slot's deform buffer.
#[cfg_attr(
    not(skel_fixtures),
    ignore = "requires the gitignored Spine fixtures in data/"
)]
#[test]
fn diamond_deform_timeline_applies() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/data/diamond-pro.skel");
    let Ok(bytes) = std::fs::read(path) else {
        return;
    };
    let data = from_binary(&bytes).unwrap();
    let slot_count = data.slots.len();
    let anims: Vec<_> = data.animations.clone();
    let arc = std::sync::Arc::new(data);
    // A deform applies only while its slot shows the keyed attachment and
    // time is past the first keyframe. Sample every animation across time
    // and confirm at least one populates a deform buffer.
    let deformed = anims.iter().any(|anim| {
        let dur = anim.duration();
        (0..=10).any(|i| {
            let t = dur * (i as f32 / 10.0);
            let mut sk = crate::skel::Skeleton::new(arc.clone());
            let mut state = crate::anim::AnimationState::new();
            state.set_animation(anim.clone(), false);
            state.update(t);
            sk.set_slots_to_setup_pose();
            state.apply(&mut sk);
            (0..slot_count).any(|s| sk.slot(s).is_some_and(|sl| !sl.deform.is_empty()))
        })
    });
    assert!(deformed, "expected a diamond animation to deform a mesh");
}

// Exercises the slider constraint on the real rig: playing an animation and
// updating world transforms runs the slider, which scrubs the "rotation"
// animation from its bone. This must pose the rig without panicking.
#[cfg_attr(
    not(skel_fixtures),
    ignore = "requires the gitignored Spine fixtures in data/"
)]
#[test]
fn diamond_slider_runs() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/data/diamond-pro.skel");
    let Ok(bytes) = std::fs::read(path) else {
        return;
    };
    let data = from_binary(&bytes).unwrap();
    let anim = data.find_animation("idle-rotating").unwrap().clone();
    let mut sk = crate::skel::Skeleton::new(std::sync::Arc::new(data));
    let mut state = crate::anim::AnimationState::new();
    state.set_animation(anim, false);
    state.update(0.5);
    sk.set_slots_to_setup_pose();
    state.apply(&mut sk);
    sk.update_world_transform();
    assert!(sk.bone(0).is_some());
}

#[test]
fn reads_a_one_value_bone_timeline() {
    // Bezier count 0, then two rotate frames (0,0) and (1,90) with a linear
    // curve between them.
    let mut b = Vec::new();
    b.push(0); // bezier-segment count = 0
    for v in [0.0_f32, 0.0, 1.0, 90.0] {
        b.extend_from_slice(&v.to_be_bytes());
    }
    b.push(0); // curve = linear
    let mut r = BinaryReader::new(&b);
    let (_tl, duration) = read_bone_timeline1(&mut r, 0, 2);
    assert!((duration - 1.0).abs() < 1e-6);
    assert!(!r.overran());
}

#[test]
fn reads_an_event_timeline() {
    // A skeleton with a root bone and one event named "footstep".
    let data = SkeletonData {
        bones: vec![BoneData {
            index: 0,
            name: "root".into(),
            ..Default::default()
        }],
        events: vec![EventData {
            name: "footstep".into(),
            int_value: 5,
            float_value: 0.0,
            string_value: String::new(),
            audio_path: None,
            volume: 1.0,
            balance: 0.0,
        }],
        ..Default::default()
    };
    // One keyframe at time 0.5: event index 0, int override 7, float 1.5,
    // no string override, no audio (so no volume / balance bytes).
    let mut b = Vec::new();
    b.extend_from_slice(&0.5_f32.to_be_bytes());
    b.push(0); // event index
    b.push(14); // int value 7 as a zig-zag var-int
    b.extend_from_slice(&1.5_f32.to_be_bytes());
    b.push(0); // string value = None
    let mut r = BinaryReader::new(&b);
    let (tl, dur) = read_event_timeline(&mut r, &data, 1);
    assert!((dur - 0.5).abs() < 1e-6);
    assert!(!r.overran());

    // Playing past the keyframe fires "footstep" with the keyframe int (7).
    let anim = Animation::new("walk", dur, vec![Timeline::Event(tl)]);
    let mut sk = crate::skel::Skeleton::new(std::sync::Arc::new(data));
    let mut state = crate::anim::AnimationState::new();
    state.set_animation(std::sync::Arc::new(anim), false);
    state.update(1.0);
    state.apply(&mut sk);
    assert_eq!(sk.events().len(), 1);
    assert_eq!(sk.events()[0].name, "footstep");
    assert_eq!(sk.events()[0].int_value, 7);
}

#[test]
fn reads_a_draw_order_timeline() {
    fn slot(index: usize, name: &str) -> SlotData {
        SlotData {
            index,
            name: name.into(),
            bone: 0,
            color: Color::WHITE,
            dark_color: None,
            attachment: None,
            blend: BlendMode::Normal,
        }
    }
    // Three slots. One keyframe at 0.5 moves slot 0 back by 2 (to the end).
    let data = SkeletonData {
        bones: vec![BoneData {
            index: 0,
            name: "root".into(),
            ..Default::default()
        }],
        slots: vec![slot(0, "a"), slot(1, "b"), slot(2, "c")],
        ..Default::default()
    };
    let mut b = Vec::new();
    b.extend_from_slice(&0.5_f32.to_be_bytes()); // time
    b.push(1); // change count
    b.push(0); // slot index
    b.push(2); // offset
    let mut r = BinaryReader::new(&b);
    let (tl, dur) = read_draw_order_timeline(&mut r, 1, 3);
    assert!((dur - 0.5).abs() < 1e-6);
    assert!(!r.overran());

    let anim = Animation::new("reorder", dur, vec![Timeline::DrawOrder(tl)]);
    let mut sk = crate::skel::Skeleton::new(std::sync::Arc::new(data));
    let mut state = crate::anim::AnimationState::new();
    state.set_animation(std::sync::Arc::new(anim), false);
    state.update(1.0);
    state.apply(&mut sk);
    // Slot 0 moved to the end: the order becomes b, c, a.
    assert_eq!(sk.draw_order(), &[1, 2, 0]);
}

#[test]
fn reads_a_slot_alpha_timeline() {
    let data = SkeletonData {
        bones: vec![BoneData {
            index: 0,
            name: "root".into(),
            ..Default::default()
        }],
        slots: vec![SlotData {
            index: 0,
            name: "s".into(),
            bone: 0,
            color: Color::WHITE,
            dark_color: None,
            attachment: None,
            blend: BlendMode::Normal,
        }],
        ..Default::default()
    };
    // Alpha 1.0 -> 0.0 over time 0 -> 1, linear (one byte per alpha channel).
    let mut b = Vec::new();
    b.push(0); // bezier-segment count
    b.extend_from_slice(&0.0_f32.to_be_bytes());
    b.push(255); // alpha 1.0
    b.extend_from_slice(&1.0_f32.to_be_bytes());
    b.push(0); // alpha 0.0
    b.push(0); // linear curve
    let mut r = BinaryReader::new(&b);
    let (tl, dur) = read_slot_color_timeline(&mut r, 0, 2, 1);
    assert!((dur - 1.0).abs() < 1e-6);
    assert!(!r.overran());

    let anim = Animation::new("fade", dur, vec![Timeline::SlotAlpha(tl)]);
    let mut sk = crate::skel::Skeleton::new(std::sync::Arc::new(data));
    let mut state = crate::anim::AnimationState::new();
    state.set_animation(std::sync::Arc::new(anim), false);
    state.update(0.5);
    sk.set_slots_to_setup_pose();
    state.apply(&mut sk);
    // Halfway the alpha is ~0.5, and red stays at the setup white.
    let c = sk.slot(0).unwrap().color;
    assert!((c.a - 0.5).abs() < 0.05, "alpha={}", c.a);
    assert!((c.r - 1.0).abs() < 1e-3, "r={}", c.r);
}

#[test]
fn reads_an_ik_constraint_timeline() {
    use crate::constraint::ik::IkConstraintData;
    use crate::constraint::ScaleYMode;

    // Mix 0 at time 0, mix 1 at time 1 (flags-packed, linear).
    let mut b = Vec::new();
    b.push(0); // bezier-segment count
    b.push(0); // flags: mix absent (0)
    b.extend_from_slice(&0.0_f32.to_be_bytes()); // time 0
    b.push(1); // flags: mix present, default 1, linear
    b.extend_from_slice(&1.0_f32.to_be_bytes()); // time 1
    let mut r = BinaryReader::new(&b);
    let (tl, dur) = read_ik_constraint_timeline(&mut r, 0, 2);
    assert!((dur - 1.0).abs() < 1e-6);
    assert!(!r.overran());

    // A one-bone IK aiming at a target above it. The timeline drives its mix.
    let data = SkeletonData {
        bones: vec![
            BoneData {
                index: 0,
                name: "root".into(),
                ..Default::default()
            },
            BoneData {
                index: 1,
                name: "aim".into(),
                parent: Some(0),
                length: 10.0,
                ..Default::default()
            },
            BoneData {
                index: 2,
                name: "target".into(),
                parent: Some(0),
                position: Vec2::new(0.0, 10.0),
                ..Default::default()
            },
        ],
        ik_constraints: vec![IkConstraintData {
            name: "aim-ik".into(),
            order: 0,
            bones: vec![1],
            target: 2,
            scale_y_mode: ScaleYMode::None,
            mix: 1.0,
            softness: 0.0,
            bend_direction: 1,
            compress: false,
            stretch: false,
        }],
        ..Default::default()
    };
    let anim = Animation::new("ik", dur, vec![Timeline::Ik(tl)]);
    let mut sk = crate::skel::Skeleton::new(std::sync::Arc::new(data));
    let mut state = crate::anim::AnimationState::new();
    state.set_animation(std::sync::Arc::new(anim), false);
    // At time 0 the keyed mix is 0, so the aim keeps its setup +x heading
    // rather than rotating up toward the target (which mix 1 would do).
    state.update(0.0);
    state.apply(&mut sk);
    sk.update_world_transform();
    let aim = sk.bone(1).unwrap();
    assert!((aim.a() - 1.0).abs() < 0.1, "a={}", aim.a());
    assert!(aim.c().abs() < 0.1, "c={}", aim.c());
}

#[test]
fn reads_a_transform_mix_timeline() {
    // Two frames, six float mix channels, linear.
    let mut b = Vec::new();
    b.push(0); // bezier-segment count
    b.extend_from_slice(&0.0_f32.to_be_bytes());
    for _ in 0..6 {
        b.extend_from_slice(&1.0_f32.to_be_bytes());
    }
    b.extend_from_slice(&1.0_f32.to_be_bytes());
    for _ in 0..6 {
        b.extend_from_slice(&0.5_f32.to_be_bytes());
    }
    b.push(0); // linear curve
    let mut r = BinaryReader::new(&b);
    let (_tl, dur) = read_curve_timeline_n(&mut r, 0, 2, 6);
    assert!((dur - 1.0).abs() < 1e-6);
    assert!(!r.overran());
}

#[test]
fn parses_a_path_constraint() {
    // bone count 1, bone 0, slot 0, flags (position/spacing Percent, rotate
    // Chain).
    let mut b = vec![1_u8, 0, 0, 26];
    for v in [0.5_f32, 10.0, 1.0, 1.0, 1.0] {
        b.extend_from_slice(&v.to_be_bytes());
    }
    let mut r = BinaryReader::new(&b);
    let pc = parse_path(&mut r, "path".into(), 2);
    assert!(!r.overran());
    assert_eq!(pc.slot, 0);
    assert_eq!(pc.position_mode, PositionMode::Percent);
    assert_eq!(pc.spacing_mode, SpacingMode::Percent);
    assert_eq!(pc.rotate_mode, RotateMode::Chain);
    assert!((pc.position - 0.5).abs() < 1e-6);
    assert!((pc.spacing - 10.0).abs() < 1e-6);
}

#[test]
fn parses_a_physics_constraint() {
    // bone 0, flags (none), step divisor 30 -> 1/30.
    let mut b = vec![0_u8, 0, 30];
    for v in [0.5_f32, 1.0, 0.9, 0.0, -10.0] {
        b.extend_from_slice(&v.to_be_bytes()); // inertia, strength, damping, wind, gravity
    }
    b.push(0); // global flags (mix defaults to 1)
    let mut r = BinaryReader::new(&b);
    let pc = parse_physics(&mut r, "phys".into(), 3);
    assert!(!r.overran());
    assert_eq!(pc.bone, 0);
    assert!((pc.step - 1.0 / 30.0).abs() < 1e-6);
    assert!((pc.inertia - 0.5).abs() < 1e-6);
    assert!((pc.gravity + 10.0).abs() < 1e-6);
    assert!((pc.mix - 1.0).abs() < 1e-6);
    assert!((pc.limit - 5000.0).abs() < 1e-3);
    assert!((pc.mass_inverse - 1.0).abs() < 1e-6);
}

#[test]
fn parses_a_transform_constraint() {
    // 1 bone, source 0, no property mappings, no offsets, default mixes.
    let b = vec![1_u8, 0, 0, 0, 0, 0];
    let mut r = BinaryReader::new(&b);
    let tc = parse_transform(&mut r, "tf".into(), 2);
    assert!(!r.overran());
    assert_eq!(tc.bones, vec![0]);
    assert_eq!(tc.source, 0);
    assert!(tc.properties.is_empty());
    assert!((tc.mix_rotate - 1.0).abs() < 1e-6);
    assert!(tc.offsets[0].abs() < 1e-6);
}

#[test]
fn parses_a_transform_constraint_with_a_property() {
    // flags 32 -> one source property: rotation -> rotation, scale 2.
    let mut b = vec![1_u8, 0, 0, 32];
    b.push(0); // from property: Rotate
    b.extend_from_slice(&0.5_f32.to_be_bytes()); // from offset
    b.push(1); // to count
    b.push(0); // to property: Rotate
    b.extend_from_slice(&0.0_f32.to_be_bytes()); // to offset
    b.extend_from_slice(&1.0_f32.to_be_bytes()); // to max
    b.extend_from_slice(&2.0_f32.to_be_bytes()); // to scale
    b.push(0); // offset flags
    b.push(0); // mix flags
    let mut r = BinaryReader::new(&b);
    let tc = parse_transform(&mut r, "tf".into(), 2);
    assert!(!r.overran());
    assert_eq!(tc.properties.len(), 1);
    assert_eq!(tc.properties[0].property, FromProp::Rotate);
    assert!((tc.properties[0].offset - 0.5).abs() < 1e-6);
    assert_eq!(tc.properties[0].to.len(), 1);
    assert_eq!(tc.properties[0].to[0].property, ToProp::Rotate);
    assert!((tc.properties[0].to[0].scale - 2.0).abs() < 1e-6);
}

// Binds the diamond rig against its atlas and confirms a sequenced
// attachment resolved its frames to distinct UVs (the per-frame regions
// were found, i.e. the frame path names match the atlas). Skips if the
// local fixtures are absent.
#[cfg_attr(
    not(skel_fixtures),
    ignore = "requires the gitignored Spine fixtures in data/"
)]
#[test]
fn diamond_sequence_binds_distinct_frames() {
    let skel = concat!(env!("CARGO_MANIFEST_DIR"), "/data/diamond-pro.skel");
    let atlas = concat!(env!("CARGO_MANIFEST_DIR"), "/data/diamond-pro.atlas");
    let (Ok(bytes), Ok(atlas_text)) = (std::fs::read(skel), std::fs::read_to_string(atlas)) else {
        return;
    };
    let mut data = from_binary(&bytes).unwrap();
    let atlas = crate::atlas::Atlas::parse(&atlas_text);
    crate::render::bind_atlas(&mut data, &atlas);

    let distinct = data.default_skin.iter().any(|(_, _, att)| {
        let seq = match att {
            Attachment::Region(r) => r.sequence.as_ref(),
            Attachment::Mesh(m) => m.sequence.as_ref(),
            _ => None,
        };
        seq.and_then(|s| Some((s.frame(0)?.0.to_vec(), s.frame(1)?.0.to_vec())))
            .is_some_and(|(a, b)| a != b)
    });
    assert!(
        distinct,
        "a sequenced attachment should bind distinct frame UVs"
    );
}

// Drives the diamond rig through the whole pipeline (load, bind, pose,
// render) and confirms it produces well-formed draw commands: each carries
// geometry with one UV pair per vertex, in-range triangle indices, and a
// real atlas page. Skips if the local fixtures are absent.
#[cfg_attr(
    not(skel_fixtures),
    ignore = "requires the gitignored Spine fixtures in data/"
)]
#[test]
fn diamond_renders_end_to_end() {
    let skel = concat!(env!("CARGO_MANIFEST_DIR"), "/data/diamond-pro.skel");
    let atlas = concat!(env!("CARGO_MANIFEST_DIR"), "/data/diamond-pro.atlas");
    let (Ok(bytes), Ok(atlas_text)) = (std::fs::read(skel), std::fs::read_to_string(atlas)) else {
        return;
    };
    let mut data = from_binary(&bytes).unwrap();
    let atlas = crate::atlas::Atlas::parse(&atlas_text);
    crate::render::bind_atlas(&mut data, &atlas);

    let anim = data.find_animation("idle-rotating").unwrap().clone();
    let mut sk = crate::skel::Skeleton::new(std::sync::Arc::new(data));
    let mut state = crate::anim::AnimationState::new();
    state.set_animation(anim, false);
    state.update(0.25);
    sk.set_slots_to_setup_pose();
    state.apply(&mut sk);
    sk.update_world_transform();

    let commands = crate::render::render(&sk);
    assert!(!commands.is_empty(), "the rig should draw something");
    for cmd in &commands {
        assert!(!cmd.positions.is_empty(), "a command needs vertices");
        assert_eq!(
            cmd.uvs.len(),
            cmd.positions.len() * 2,
            "one UV pair per vertex"
        );
        assert!(!cmd.triangles.is_empty(), "a command needs triangles");
        let vertex_count = cmd.positions.len() as u16;
        assert!(
            cmd.triangles.iter().all(|&i| i < vertex_count),
            "triangle indices stay in range"
        );
        assert!(cmd.page < atlas.pages.len(), "a real atlas page");
    }
}

// Builds a complete, minimal essential `.skel` in memory (one root bone, one
// slot, no constraints / skins / events / animations) and round-trips it
// through `from_binary`. Covers the whole header plus the bone and slot
// sections plus the empty trailing sections without depending on the
// gitignored diamond fixture, so this path stays covered in a clean checkout.
#[test]
fn from_binary_round_trips_a_minimal_skeleton() {
    fn put_string(out: &mut Vec<u8>, s: &str) {
        // var_uint length where 0 is None and n encodes n - 1 bytes.
        out.push(u8::try_from(s.len() + 1).unwrap());
        out.extend_from_slice(s.as_bytes());
    }
    fn put_f32(out: &mut Vec<u8>, v: f32) {
        out.extend_from_slice(&v.to_be_bytes());
    }

    let mut b = Vec::new();
    // Header: 64-bit hash, version, setup bounds, reference scale, flag.
    b.extend_from_slice(&[0; 8]);
    put_string(&mut b, "4.3.00");
    for v in [0.0_f32, 0.0, 100.0, 200.0] {
        put_f32(&mut b, v);
    }
    put_f32(&mut b, 1.0); // reference scale
    b.push(0); // essential (nonessential = false)
    b.push(0); // string table: 0 entries

    // One root bone.
    b.push(1); // bone count
    put_string(&mut b, "root");
    // rotation, x, y, scaleX, scaleY, shearX, shearY, length.
    for v in [0.0_f32, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0] {
        put_f32(&mut b, v);
    }
    b.push(0); // inherit = Normal
    b.push(0); // skin required = false

    // One slot on the root bone: white, no dark tint, no setup attachment.
    b.push(1); // slot count
    put_string(&mut b, "slot");
    b.push(0); // bone index 0
    b.extend_from_slice(&0xFFFF_FFFF_u32.to_be_bytes()); // color
    b.extend_from_slice(&0xFFFF_FFFF_u32.to_be_bytes()); // dark (none)
    b.push(0); // attachment = None (string ref 0)
    b.push(0); // blend = normal

    // No constraints, an empty default skin, no named skins / events / anims.
    b.push(0); // constraint count
    b.push(0); // default skin: 0 slots
    b.push(0); // named skin count
    b.push(0); // event count
    b.push(0); // animation count

    let data = from_binary(&b).expect("minimal skeleton parses");
    assert_eq!(data.spine_version.as_deref(), Some("4.3.00"));
    assert_eq!(data.bones.len(), 1);
    assert_eq!(data.bones[0].name, "root");
    assert_eq!(data.slots.len(), 1);
    assert_eq!(data.slots[0].name, "slot");
    assert_eq!(data.slots[0].bone, 0);
}

/// A complete, minimal essential `.skel` (one root bone, no slots /
/// constraints / skins / events / animations) whose header declares
/// `version`, so version-validation tests doctor only the version field.
fn minimal_skeleton_bytes(version: &str) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(&[0; 8]); // hash
    enc_str(&mut b, version);
    for v in [0.0_f32, 0.0, 100.0, 200.0, 1.0] {
        b.extend_from_slice(&v.to_be_bytes()); // bounds + reference scale
    }
    b.push(0); // essential (nonessential = false)
    b.push(0); // string table: 0 entries
    b.push(1); // bone count
    enc_str(&mut b, "root");
    for v in [0.0_f32, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0] {
        b.extend_from_slice(&v.to_be_bytes()); // bone transform
    }
    b.push(0); // inherit = Normal
    b.push(0); // skin required = false
               // No slots / constraints, an empty default skin, no skins / events / anims.
    b.extend_from_slice(&[0; 6]);
    b
}

// A doctored header from another Spine release must be rejected up front:
// the section layouts differ, so reading on would misparse silently. This
// includes "4.30", which a naive prefix test would accept as "4.3".
#[test]
fn rejects_an_export_from_another_spine_version() {
    for wrong in ["4.4.00", "4.2.43", "3.8.99", "4.30.1", "5.0.00"] {
        match from_binary(&minimal_skeleton_bytes(wrong)) {
            Err(BinaryError::UnsupportedVersion { found, expected }) => {
                assert_eq!(found, wrong);
                assert_eq!(expected, "4.3");
            }
            other => panic!("version {wrong} must be rejected, got {other:?}"),
        }
    }
}

// Any 4.3 patch release uses the 4.3 layout, so all of them load. A missing
// or empty version string (hand-built data, as in the tests above) is also
// accepted as-is.
#[test]
fn accepts_any_4_3_patch_release() {
    for ok in ["4.3", "4.3.00", "4.3.99", ""] {
        let data = from_binary(&minimal_skeleton_bytes(ok))
            .unwrap_or_else(|e| panic!("version {ok:?} must load: {e}"));
        assert_eq!(data.bones.len(), 1);
    }
}

// A header that declares 100 bones but provides no bone data: each bone
// needs many bytes, so the count exceeds the remaining data and must be
// rejected as corrupt rather than driving the read loop. (A real attack
// uses a multi-byte varint for billions, and 100 with no data is the same bug.)
#[test]
fn corrupt_length_is_rejected_not_looped() {
    let mut b = Vec::new();
    b.extend_from_slice(&[0; 8]); // hash
    b.push(0); // version = None
    for v in [0.0_f32, 0.0, 0.0, 0.0, 1.0] {
        b.extend_from_slice(&v.to_be_bytes()); // bounds + reference scale
    }
    b.push(0); // essential
    b.push(0); // 0 strings
    b.push(100); // bone count = 100, but no bone bytes follow
    assert!(matches!(from_binary(&b), Err(BinaryError::CorruptLength)));
}

// An otherwise valid header whose single constraint carries an unknown type
// tag must surface as an error, not silently stop reading the constraints.
#[test]
fn unknown_constraint_type_is_an_error() {
    let mut b = Vec::new();
    b.extend_from_slice(&[0; 8]); // hash
    b.push(0); // version = None
    for v in [0.0_f32, 0.0, 0.0, 0.0, 1.0] {
        b.extend_from_slice(&v.to_be_bytes());
    }
    b.push(0); // essential
    b.push(0); // 0 strings
    b.push(1); // 1 bone (root)
    b.push(0); // root name = None
    for _ in 0..8 {
        b.extend_from_slice(&0.0_f32.to_be_bytes()); // bone transform
    }
    b.push(0); // inherit = Normal
    b.push(0); // skin required = false
    b.push(0); // 0 slots
    b.push(1); // 1 constraint
    b.push(0); // constraint name = None
    b.push(99); // unknown type tag
    assert!(matches!(
        from_binary(&b),
        Err(BinaryError::UnknownConstraintType(99))
    ));
}

// Loads the same tiny rig (two bones, one rotate animation) from JSON and
// from binary and confirms both loaders pose it identically. This is the
// guard against the two loaders drifting apart: they share the channel
// definitions and must agree on curve reading. Both inputs are built in
// memory, so the test needs no external fixture.
#[cfg(feature = "json")]
#[test]
fn json_and_binary_loaders_agree() {
    const JSON: &str = r#"{
        "bones": [
            { "name": "root" },
            { "name": "child", "parent": "root", "x": 10 }
        ],
        "animations": {
            "spin": {
                "bones": {
                    "child": {
                        "rotate": [ { "time": 0, "value": 0 }, { "time": 1, "value": 90 } ]
                    }
                }
            }
        }
    }"#;

    fn put_string(out: &mut Vec<u8>, s: &str) {
        out.push(u8::try_from(s.len() + 1).unwrap());
        out.extend_from_slice(s.as_bytes());
    }
    fn put_f32(out: &mut Vec<u8>, v: f32) {
        out.extend_from_slice(&v.to_be_bytes());
    }

    let mut b = Vec::new();
    b.extend_from_slice(&[0; 8]); // hash
    b.push(0); // version = None
    for v in [0.0_f32, 0.0, 0.0, 0.0, 1.0] {
        put_f32(&mut b, v); // bounds + reference scale
    }
    b.push(0); // essential
    b.push(0); // 0 strings
    b.push(2); // 2 bones
    put_string(&mut b, "root");
    for v in [0.0_f32, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0] {
        put_f32(&mut b, v);
    }
    b.push(0); // inherit
    b.push(0); // skin required
    put_string(&mut b, "child");
    b.push(0); // parent index 0
    for v in [0.0_f32, 10.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0] {
        put_f32(&mut b, v); // rotation, x = 10, y, scaleX, scaleY, shearX, shearY, length
    }
    b.push(0); // inherit
    b.push(0); // skin required
    b.push(0); // 0 slots
    b.push(0); // 0 constraints
    b.push(0); // empty default skin
    b.push(0); // 0 named skins
    b.push(0); // 0 events
    b.push(1); // 1 animation
    put_string(&mut b, "spin");
    b.push(0); // ignored timeline count
    b.push(0); // 0 slot groups
    b.push(1); // 1 bone group
    b.push(1); // bone index 1 (child)
    b.push(1); // 1 timeline for this bone
    b.push(0); // kind 0 = rotate
    b.push(2); // 2 frames
    b.push(0); // 0 bezier curves
    put_f32(&mut b, 0.0); // frame 0 time
    put_f32(&mut b, 0.0); // frame 0 value
    put_f32(&mut b, 1.0); // frame 1 time
    put_f32(&mut b, 90.0); // frame 1 value
    b.push(0); // linear curve
               // ik, transform, path, physics, slider, deform, draw order, folder, event
    b.extend_from_slice(&[0; 9]);

    let json_data = crate::load::from_json(JSON).unwrap();
    let bin_data = from_binary(&b).unwrap();
    assert_eq!(json_data.bones.len(), 2);
    assert_eq!(json_data.bones.len(), bin_data.bones.len());

    // Pose each at the same time and read the child's world transform.
    let pose = |data: SkeletonData| {
        let anim = data.find_animation("spin").unwrap().clone();
        let mut sk = crate::skel::Skeleton::new(std::sync::Arc::new(data));
        let mut state = crate::anim::AnimationState::new();
        state.set_animation(anim, false);
        state.update(0.5);
        sk.set_bones_to_setup_pose();
        state.apply(&mut sk);
        sk.update_world_transform();
        let bone = sk.bone(1).unwrap();
        [
            bone.a(),
            bone.b(),
            bone.c(),
            bone.d(),
            bone.world_x(),
            bone.world_y(),
        ]
    };
    let from_json_pose = pose(json_data);
    let from_binary_pose = pose(bin_data);
    for (j, b) in from_json_pose.iter().zip(from_binary_pose.iter()) {
        assert!(
            (j - b).abs() < 1e-5,
            "loaders disagree: json {from_json_pose:?} vs binary {from_binary_pose:?}"
        );
    }
}

#[test]
fn applies_a_sequence_timeline() {
    use crate::anim::SequenceTimeline;

    // A slot showing a four-region sequence "seq". One looping keyframe at
    // time 0 (mode loop = 2, index 0) with a 0.1s delay: at 0.25s the index
    // advances by floor(0.25 / 0.1) = 2.
    let data = SkeletonData {
        bones: vec![BoneData {
            index: 0,
            name: "root".into(),
            ..Default::default()
        }],
        slots: vec![SlotData {
            index: 0,
            name: "s".into(),
            bone: 0,
            color: Color::WHITE,
            dark_color: None,
            attachment: Some("seq".into()),
            blend: BlendMode::Normal,
        }],
        ..Default::default()
    };
    let tl = SequenceTimeline::new(0, "seq".into(), 4, vec![0.0], vec![2_u32], vec![0.1]);
    let anim = Animation::new("flip", 1.0, vec![Timeline::Sequence(tl)]);
    let mut sk = crate::skel::Skeleton::new(std::sync::Arc::new(data));
    let mut state = crate::anim::AnimationState::new();
    state.set_animation(std::sync::Arc::new(anim), false);
    state.update(0.25);
    state.apply(&mut sk);
    assert_eq!(sk.slot(0).unwrap().sequence_index, 2);
}
