//! Parity tests against the official Spine 4.3 example rigs.
//!
//! Each rig is loaded from its `.json` and its `.skel` export, and the two
//! loads must agree. The exports are not committed. build.rs sets the
//! `spine_examples` cfg when they are present under `data/examples/`.

use super::from_binary;
use crate::data::SkeletonData;
use crate::load::from_json;

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
