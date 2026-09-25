//! Atlas text parsing, region lookup, and binding to a fixed skeleton.
//!
//! The skeleton has one attachment of each kind that reads the atlas: a region
//! (`r`), a mesh (`m`), a region sequence (`s0` to `s2`), and a mesh sequence
//! (`t0`, `t1`). Seeds that name those regions reach the bind and render paths.

#![no_main]

use std::sync::Arc;

use chine::atlas::Atlas;
use chine::render::{bind_atlas, render};
use chine::skel::Skeleton;
use libfuzzer_sys::fuzz_target;

const SKELETON: &str = r#"{
    "bones": [ { "name": "root" } ],
    "slots": [
        { "name": "a", "bone": "root", "attachment": "r" },
        { "name": "b", "bone": "root", "attachment": "m" },
        { "name": "c", "bone": "root", "attachment": "s" },
        { "name": "d", "bone": "root", "attachment": "t" }
    ],
    "skins": [ { "name": "default", "attachments": {
        "a": { "r": { "width": 40, "height": 50, "rotation": 15 } },
        "b": { "m": {
            "type": "mesh", "uvs": [0,0, 1,0, 0,1], "triangles": [0,1,2],
            "vertices": [10,10, 20,10, 10,20], "hull": 3
        } },
        "c": { "s": { "width": 8, "height": 8,
            "sequence": { "count": 3, "start": 0, "digits": 1 } } },
        "d": { "t": {
            "type": "mesh", "uvs": [0,0, 1,0, 0,1], "triangles": [0,1,2],
            "vertices": [0,0, 5,0, 0,5], "hull": 3,
            "sequence": { "count": 2, "start": 0, "digits": 1 }
        } }
    } } ]
}"#;

fuzz_target!(|bytes: &[u8]| {
    let text = String::from_utf8_lossy(bytes);
    let atlas = Atlas::parse(&text);
    for region in &atlas.regions {
        let _ = atlas.find_region(&region.name);
    }
    let Ok(mut data) = chine::load::from_json(SKELETON) else {
        return;
    };
    bind_atlas(&mut data, &atlas);
    let mut skeleton = Skeleton::new(Arc::new(data));
    skeleton.update_world_transform();
    let _ = render(&skeleton);
});
