//! Parity tests against the official Spine 4.3 example rigs.
//!
//! Each rig is loaded from its `.json` and its `.skel` export, and the two
//! loads must agree. The exports are not committed. build.rs sets the
//! `spine_examples` cfg when they are present under `data/examples/`.

use std::sync::Arc;

use super::from_binary;
use crate::attach::Attachment;
use crate::data::SkeletonData;
use crate::load::from_json;
use crate::render::render;
use crate::skel::Skeleton;
use crate::skin::Skin;

/// The official example rigs, each exported as `.json` and `.skel`.
const EXAMPLES: [&str; 8] = [
    "coin-pro",
    "diamond-pro",
    "mix-and-match-pro",
    "raptor-pro",
    "spineboy-pro",
    "stretchyman-pro",
    "tank-pro",
    "vine-pro",
];

/// The JSON and binary loads of example `name`.
fn load(name: &str) -> (SkeletonData, SkeletonData) {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/data/examples");
    let json = std::fs::read_to_string(format!("{dir}/{name}.json")).expect("the json export");
    let skel = std::fs::read(format!("{dir}/{name}.skel")).expect("the skel export");
    let json = from_json(&json).unwrap_or_else(|e| panic!("{name}.json: {e}"));
    let skel = from_binary(&skel).unwrap_or_else(|e| panic!("{name}.skel: {e}"));
    (json, skel)
}

// The binary loader read a bone's length before its inherit mode, where Spine
// writes the inherit byte first. The examples then loaded 275 wrong lengths
// and 21 wrong inherit modes from their `.skel` exports.
#[cfg_attr(
    not(spine_examples),
    ignore = "requires the official Spine examples in data/examples/"
)]
#[test]
fn bone_lengths_and_inherit_modes_match_json() {
    for name in EXAMPLES {
        let (json, skel) = load(name);
        assert_eq!(json.bones.len(), skel.bones.len(), "{name}");
        for (j, b) in json.bones.iter().zip(&skel.bones) {
            assert_eq!(j.name, b.name, "{name}");
            assert_eq!(j.parent, b.parent, "{name} {}", j.name);
            assert!(
                (j.length - b.length).abs() < 1e-3,
                "{name} {}: length {} in json, {} in skel",
                j.name,
                j.length,
                b.length
            );
            assert_eq!(j.inherit, b.inherit, "{name} {}", j.name);
        }
    }
}

/// A short description of an attachment: its kind, and its atlas path for
/// the kinds that draw.
fn describe(attachment: &Attachment) -> String {
    match attachment {
        Attachment::Region(region) => format!("region {}", region.path),
        Attachment::Mesh(mesh) => format!("mesh {} {}", mesh.path, mesh.vertex_count()),
        Attachment::LinkedMesh(link) => format!("unresolved link {}", link.path),
        Attachment::Path(_) => "path".into(),
        Attachment::BoundingBox(_) => "bounding box".into(),
        Attachment::Point(_) => "point".into(),
        Attachment::Clipping(_) => "clipping".into(),
    }
}

// The JSON loader took an attachment's path from its key in the skin rather
// than its "name", so it drew other atlas regions than the .skel export.
// Every attachment now loads the same from both exports, and every linked
// mesh resolves.
#[cfg_attr(
    not(spine_examples),
    ignore = "requires the official Spine examples in data/examples/"
)]
#[test]
fn attachments_match_json() {
    for name in EXAMPLES {
        let (json, skel) = load(name);
        let skins = |data: &SkeletonData| -> Vec<Skin> {
            std::iter::once(&data.default_skin)
                .chain(&data.skins)
                .cloned()
                .collect()
        };
        for (j, b) in skins(&json).iter().zip(&skins(&skel)) {
            assert_eq!(j.name, b.name, "{name}");
            let mut attachments = 0;
            for (slot, key, attachment) in j.iter() {
                let other = b.attachment(slot, key).map(describe);
                let described = describe(attachment);
                assert!(!described.starts_with("unresolved"), "{name}: {described}");
                assert_eq!(Some(described), other, "{name} {} {slot} {key}", j.name);
                attachments += 1;
            }
            assert_eq!(attachments, b.iter().count(), "{name} {}", j.name);
        }
    }
}

// The binary loader dropped a linked mesh's source skin and slot, so a
// mix-and-match skin drew fewer meshes from its .skel export. With every
// skin, the setup pose now draws the same commands from both exports.
#[cfg_attr(
    not(spine_examples),
    ignore = "requires the official Spine examples in data/examples/"
)]
#[test]
fn every_skin_renders_the_same_from_both_exports() {
    for name in EXAMPLES {
        let (json, skel) = load(name);
        let skin_names: Vec<String> = json.skins.iter().map(|s| s.name.clone()).collect();
        let (json, skel) = (Arc::new(json), Arc::new(skel));
        let mut commands = 0;
        for skin in std::iter::once(None).chain(skin_names.iter().map(Some)) {
            let draw = |data: &Arc<SkeletonData>| {
                let mut sk = Skeleton::new(Arc::clone(data));
                if let Some(skin) = skin {
                    sk.set_skin(skin);
                }
                sk.update_world_transform();
                render(&sk)
                    .iter()
                    .map(|c| (c.positions.len(), c.triangles.len()))
                    .collect::<Vec<_>>()
            };
            let drawn = draw(&json);
            commands += drawn.len();
            assert_eq!(drawn, draw(&skel), "{name} {skin:?}");
        }
        assert!(commands > 0, "{name} draws nothing");
    }
}

// Skin-required bones and constraints, and the bones and constraints each
// skin lists, load the same from both exports.
#[cfg_attr(
    not(spine_examples),
    ignore = "requires the official Spine examples in data/examples/"
)]
#[test]
fn skin_requirements_match_json() {
    use std::collections::HashSet;

    let mut listed = 0;
    for name in EXAMPLES {
        let (json, skel) = load(name);
        let required = |data: &SkeletonData| {
            let bones: Vec<bool> = data.bones.iter().map(|b| b.skin_required).collect();
            let constraints: Vec<bool> = (data.ik_constraints.iter().map(|c| c.skin_required))
                .chain(data.transform_constraints.iter().map(|c| c.skin_required))
                .chain(data.path_constraints.iter().map(|c| c.skin_required))
                .chain(data.physics_constraints.iter().map(|c| c.skin_required))
                .chain(data.sliders.iter().map(|c| c.skin_required))
                .collect();
            (bones, constraints)
        };
        assert_eq!(required(&json), required(&skel), "{name}");
        for (j, b) in json.skins.iter().zip(&skel.skins) {
            let bones = |skin: &Skin| skin.bones.iter().copied().collect::<HashSet<_>>();
            let constraints =
                |skin: &Skin| skin.constraints.iter().copied().collect::<HashSet<_>>();
            assert_eq!(bones(j), bones(b), "{name} {}", j.name);
            assert_eq!(constraints(j), constraints(b), "{name} {}", j.name);
            listed += j.bones.len() + j.constraints.len();
        }
    }
    // mix-and-match-pro's skins list skin-required bones and constraints.
    assert!(listed > 0);
}

// Every animation shows the same sequence frames from both exports.
#[cfg_attr(
    not(spine_examples),
    ignore = "requires the official Spine examples in data/examples/"
)]
#[test]
fn sequence_frames_match_json() {
    use crate::anim::MixFrom;

    for name in EXAMPLES {
        let (json, skel) = load(name);
        let (json, skel) = (Arc::new(json), Arc::new(skel));
        for anim in &json.animations {
            let other = skel.find_animation(anim.name()).expect("the animation");
            // Times off the keys, which the two exports round differently.
            for step in 0..20 {
                let time = anim.duration() * (step as f32 + 0.37) / 20.0;
                let frames = |data: &Arc<SkeletonData>, anim: &crate::anim::Animation| {
                    let mut sk = Skeleton::new(Arc::clone(data));
                    anim.apply(&mut sk, -1.0, time, 1.0, MixFrom::Setup, false);
                    sk.slots()
                        .iter()
                        .map(|s| s.sequence_index)
                        .collect::<Vec<_>>()
                };
                assert_eq!(
                    frames(&json, anim),
                    frames(&skel, other),
                    "{name} {} at {time}",
                    anim.name()
                );
            }
        }
    }
}

/// Each bone's world transform and every drawn vertex, `time` into `anim`
/// played from the setup pose with `skin` active.
fn posed(
    data: &Arc<SkeletonData>,
    skin: Option<&str>,
    anim: &crate::anim::Animation,
    time: f32,
) -> (Vec<[f32; 6]>, Vec<glam::Vec2>) {
    let mut sk = Skeleton::new(Arc::clone(data));
    if let Some(skin) = skin {
        sk.set_skin(skin);
    }
    anim.apply(&mut sk, -1.0, time, 1.0, crate::anim::MixFrom::Setup, false);
    sk.update_world_transform();
    let bones = sk
        .bones()
        .iter()
        .map(|b| [b.a(), b.b(), b.c(), b.d(), b.world_x(), b.world_y()])
        .collect();
    let vertices = render(&sk)
        .iter()
        .flat_map(|c| c.positions.iter().copied())
        .collect();
    (bones, vertices)
}

// Every animation poses the bones and draws the vertices of each skin within
// float precision of each other from both exports. JSON stores decimals and
// the binary stores floats, so the large rigs differ by up to about 0.3.
#[cfg_attr(
    not(spine_examples),
    ignore = "requires the official Spine examples in data/examples/"
)]
#[test]
fn animations_pose_the_same_from_both_exports() {
    for name in EXAMPLES {
        let (json, skel) = load(name);
        let skins: Vec<String> = json.skins.iter().map(|s| s.name.clone()).collect();
        let (json, skel) = (Arc::new(json), Arc::new(skel));
        for skin in std::iter::once(None).chain(skins.iter().map(|s| Some(s.as_str()))) {
            for anim in &json.animations {
                let other = skel.find_animation(anim.name()).expect("the animation");
                // Times off the keys: JSON and binary round a key time
                // differently, so a sample right on a key can land on
                // either side of it.
                for step in 0..5 {
                    let time = anim.duration() * (step as f32 + 0.37) / 5.0;
                    let (bones, vertices) = posed(&json, skin, anim, time);
                    let (other_bones, other_vertices) = posed(&skel, skin, other, time);
                    let at = format!("{name} {skin:?} {} at {time}", anim.name());
                    assert_eq!(vertices.len(), other_vertices.len(), "{at}");
                    for (a, b) in bones.iter().zip(&other_bones) {
                        let worst = (0..6).map(|k| (a[k] - b[k]).abs()).fold(0.0, f32::max);
                        assert!(worst < 0.05, "{at}: bone off by {worst}");
                    }
                    for (a, b) in vertices.iter().zip(&other_vertices) {
                        assert!((*a - *b).length() < 0.5, "{at}: {a} vs {b}");
                    }
                }
            }
        }
    }
}
