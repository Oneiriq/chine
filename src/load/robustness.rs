//! Regression tests for corrupt and hostile JSON exports: each input used to
//! panic, hang, or allocate without bound, and now loads or fails cleanly.

use super::*;

/// A one-bone, one-slot rig whose default skin holds `attachment` (a JSON
/// object) as `a` on slot `s`, followed by the top-level members in `extra`.
fn rig(attachment: &str, extra: &str) -> String {
    format!(
        r#"{{
        "bones": [ {{ "name": "root" }} ],
        "slots": [ {{ "name": "s", "bone": "root", "attachment": "a" }} ],
        "skins": [ {{ "name": "default", "attachments": {{ "s": {{ "a": {attachment} }} }} }} ]
        {extra}
    }}"#
    )
}

/// The message of the schema error `json` must fail with.
fn schema_error(json: &str) -> String {
    match from_json(json) {
        Err(LoadError::Schema(msg)) => msg,
        other => panic!("expected a schema error, got {:?}", other.map(|_| ())),
    }
}

/// `n` comma-separated zeros, for large vertex arrays.
fn zeros(n: usize) -> String {
    vec!["0"; n].join(",")
}

/// An unweighted mesh with `vertex_count` vertices at the origin.
fn big_mesh(vertex_count: usize) -> String {
    let coords = zeros(vertex_count * 2);
    format!(r#"{{ "type": "mesh", "uvs": [{coords}], "triangles": [], "vertices": [{coords}] }}"#)
}

#[test]
fn fuzzed_meshes_with_mismatched_vertices_are_rejected() {
    // Fuzzed inputs that loaded and then indexed past the weighted vertex
    // arrays at render time: a misspelled `vertices` key, 5 UV values for 6
    // vertex values, and 7 UV values.
    let inputs = [
        r#"{"bones":[{"name":"root"}],"slots":[{"name":"s","bone":"root","attachment":"m"}],
            "skins":[{"name":"default","attachments":{"s":{"m":{"type":"mesh","uvs":[0,0,1,0,0,1],
            "triangles":[0,1,2],"vurtices":[0,0,10,0,0,10]}}}},{"name":"alt","attachments":{"s":{"m":{}}}}],
            "animations":{"flap":{"deform":{"default":{"s":{"m":[{"time":0,"typvertices":[0,0,0,0,0,0]},
            {"time":1,"offset":2,"vers":[0,0,0,0,0,0]},{"time":1,"offset":2,"vertices":[50,0]}]}}}}}}"#,
        r#"{"bones":[{"name":"root"}],"slots":[{"name":"s","bone":"root","attachment":"m"}],
            "skins":[{"name":"default","attachments":{"s":{"m":{"type":"mesh","uvs":[0,0,1,0,0.1],
            "triangles":[0,1,2],"vertices":[1,0,10,0,0,10]}}}}]}"#,
        r#"{"bones":[{"name":"root"}],"slots":[{"name":"s","bone":"root","attachment":"m"}],
            "skins":[{"name":"default","attachments":{"s":{"m":{"type":"mesh","uvs":[1,0,1,0,0,1,2],
            "vertices":[0,0,10,0,0,10]}}}},{"name":"alt","attachments":{"s":{"m":{"type":"linkedmesh",
            "skin":"defa~lt","parent":"m","color":"ff0000ff"}}}}]}"#,
    ];
    for input in inputs {
        schema_error(input);
    }
}

#[test]
fn mesh_with_an_odd_uv_count_is_rejected() {
    let json = rig(
        r#"{ "type": "mesh", "uvs": [0,0, 1], "triangles": [], "vertices": [0,0, 1] }"#,
        "",
    );
    assert!(schema_error(&json).contains("odd number of UV"));
}

#[test]
fn weighted_vertex_running_past_the_array_is_rejected() {
    // The vertex claims two bones but holds one and a stray value.
    let json = rig(
        r#"{ "type": "mesh", "uvs": [0,0], "vertices": [2, 0,1,1,0.5, 0] }"#,
        "",
    );
    assert!(schema_error(&json).contains("truncated"));
}

#[test]
fn weighted_vertex_with_a_huge_bone_count_is_rejected() {
    // 1e30 saturates to usize::MAX bones for a single vertex.
    let json = rig(
        r#"{ "type": "mesh", "uvs": [0,0], "vertices": [1e30] }"#,
        "",
    );
    assert!(schema_error(&json).contains("truncated"));
}

#[test]
fn weighted_vertex_naming_a_missing_bone_is_rejected() {
    for bone in ["1", "7", "-1", "1e30"] {
        let json = rig(
            &format!(r#"{{ "type": "mesh", "uvs": [0,0], "vertices": [1, {bone},0,0,1] }}"#),
            "",
        );
        assert!(
            matches!(from_json(&json), Err(LoadError::BadReference(_))),
            "bone {bone}"
        );
    }
}

#[test]
fn weighted_vertex_count_must_match_the_uvs() {
    // Two vertices of UVs, but weighted data for one.
    let json = rig(
        r#"{ "type": "mesh", "uvs": [0,0, 1,1], "vertices": [1, 0,0,0,1] }"#,
        "",
    );
    assert!(schema_error(&json).contains("vertex count"));
}

#[test]
fn well_formed_weighted_mesh_still_loads() {
    // Two vertices: one on two bones (with -0.5 truncating to bone 0 as in
    // Spine), one on no bones.
    let json = r#"{
        "bones": [ { "name": "root" }, { "name": "b", "parent": "root" } ],
        "slots": [ { "name": "s", "bone": "root" } ],
        "skins": [ { "name": "default", "attachments": { "s": { "a": {
            "type": "mesh", "uvs": [0,0, 1,1], "triangles": [0,1,1], "hull": 2,
            "vertices": [2, -0.5,1,2,0.5, 1,3,4,0.5, 0]
        } } } } ]
    }"#;
    let data = from_json(json).unwrap();
    let Some(Attachment::Mesh(m)) = data.default_skin.attachment(0, "a") else {
        panic!("expected a mesh");
    };
    assert_eq!(m.vertex_count(), 2);
    assert_eq!(m.deform_len(), 4);
    assert_eq!(m.hull_length, 2);
    assert!(m.setup_vertices().is_none());
}

#[test]
fn triangle_index_past_the_vertex_list_is_rejected() {
    for index in ["3", "65536", "18446744073709551615"] {
        let json = rig(
            &format!(
                r#"{{ "type": "mesh", "uvs": [0,0, 1,0, 0,1], "triangles": [0,1,{index}], "vertices": [0,0, 1,0, 0,1] }}"#
            ),
            "",
        );
        assert!(schema_error(&json).contains("triangle"), "index {index}");
    }
}

#[test]
fn hull_longer_than_the_vertex_list_is_rejected() {
    let json = rig(
        r#"{ "type": "mesh", "uvs": [0,0, 1,0, 0,1], "triangles": [0,1,2], "vertices": [0,0, 1,0, 0,1], "hull": 4 }"#,
        "",
    );
    assert!(schema_error(&json).contains("hull"));
}

#[test]
fn polygon_vertex_counts_must_match_their_vertices() {
    // `vertexCount * 2` used to overflow for a count near u64::MAX, and a
    // count with no vertex data loaded as an empty weighted polygon.
    for kind in ["path", "boundingbox", "clipping"] {
        for (count, vertices) in [("18446744073709551615", ""), ("3", ""), ("1", "2, 0,0,0,1")] {
            let json = rig(
                &format!(
                    r#"{{ "type": "{kind}", "vertexCount": {count}, "vertices": [{vertices}] }}"#
                ),
                "",
            );
            schema_error(&json);
        }
    }
}

#[test]
fn deform_offset_near_u64_max_is_ignored() {
    // `offset + index` used to overflow.
    let json = rig(
        r#"{ "type": "mesh", "uvs": [0,0], "triangles": [], "vertices": [5,6] }"#,
        r#", "animations": { "d": { "deform": { "default": { "s": { "a": [
            { "time": 0, "offset": 18446744073709551615, "vertices": [1, 2] },
            { "time": 1, "offset": 1, "vertices": [3, 4] }
        ] } } } } }"#,
    );
    let data = from_json(&json).unwrap();
    let anim = data.find_animation("d").unwrap().clone();
    let mut sk = crate::skel::Skeleton::new(Arc::new(data));
    anim.apply(&mut sk, 0.0, 1.0, 1.0, crate::anim::MixFrom::Setup, false);
    // The second key writes 3 at index 1 and drops the 4 past the end.
    assert_eq!(sk.slot(0).unwrap().deform, vec![5.0, 9.0]);
}

/// A rig with slots `a`, `b`, and `c` and one draw order key with `offsets`.
fn draw_order_rig(offsets: &str) -> String {
    format!(
        r#"{{
        "bones": [ {{ "name": "root" }} ],
        "slots": [
            {{ "name": "a", "bone": "root" }},
            {{ "name": "b", "bone": "root" }},
            {{ "name": "c", "bone": "root" }}
        ],
        "animations": {{ "o": {{ "drawOrder": [ {{ "time": 0, "offsets": [{offsets}] }} ] }} }}
    }}"#
    )
}

#[test]
fn draw_order_listing_a_slot_twice_is_rejected() {
    // This used to index past the unchanged-slot list.
    let json = draw_order_rig(r#"{ "slot": "a", "offset": 0 }, { "slot": "a", "offset": 0 }"#);
    assert!(schema_error(&json).contains("draw order"));
}

#[test]
fn draw_order_offset_outside_the_slot_list_is_rejected() {
    // i32::MAX used to overflow `slot + offset`, 2^32 wrapped to 0, and -5
    // left a hole in the order.
    for offset in [
        "2147483647",
        "4294967296",
        "-5",
        "3",
        "-9223372036854775808",
    ] {
        let json = draw_order_rig(&format!(r#"{{ "slot": "b", "offset": {offset} }}"#));
        assert!(
            schema_error(&json).contains("draw order"),
            "offset {offset}"
        );
    }
}

#[test]
fn draw_order_moving_two_slots_to_one_position_is_rejected() {
    // Both land on position 1, which used to leave a hole marked usize::MAX.
    let json = draw_order_rig(r#"{ "slot": "a", "offset": 1 }, { "slot": "b", "offset": 0 }"#);
    assert!(schema_error(&json).contains("draw order"));
}

#[test]
fn valid_draw_order_with_unsorted_offsets_still_loads() {
    // Move `c` to the front and `a` to the back, listed out of slot order.
    let json = draw_order_rig(r#"{ "slot": "c", "offset": -2 }, { "slot": "a", "offset": 2 }"#);
    let data = from_json(&json).unwrap();
    let anim = data.find_animation("o").unwrap().clone();
    let mut sk = crate::skel::Skeleton::new(Arc::new(data));
    anim.apply(&mut sk, 0.0, 0.0, 1.0, crate::anim::MixFrom::Setup, false);
    assert_eq!(sk.draw_order(), [2, 1, 0]);
}

#[test]
fn sequence_fields_past_i32_are_rejected() {
    // A start near usize::MAX used to overflow `start + index` at bind time.
    for (key, value) in [
        ("start", "9223372036854775807"),
        ("count", "2147483648"),
        ("digits", "4294967296"),
        ("setup", "18446744073709551615"),
    ] {
        let json = rig(
            &format!(r#"{{ "type": "region", "sequence": {{ "count": 1, "{key}": {value} }} }}"#),
            "",
        );
        assert!(schema_error(&json).contains("sequence"), "{key}");
    }
}

#[test]
fn huge_sequence_count_or_digits_is_rejected() {
    // Binding an atlas loops over `count` and pads names to `digits`, so these
    // hung or allocated gigabytes.
    for sequence in [
        r#"{ "count": 2147483647 }"#,
        r#"{ "count": 1, "digits": 2147483647 }"#,
    ] {
        let json = rig(
            &format!(r#"{{ "type": "region", "sequence": {sequence} }}"#),
            "",
        );
        assert!(schema_error(&json).contains("load limit"), "{sequence}");
    }
    // A realistic flipbook still loads.
    let json = rig(
        r#"{ "type": "region", "sequence": { "count": 120, "start": 1, "digits": 4 } }"#,
        "",
    );
    assert!(from_json(&json).is_ok());
}

#[test]
fn deform_keys_past_the_load_limit_are_rejected() {
    // A thousand empty keys, each expanded to a 40,000-float frame (160 MB).
    let keys = vec![r#"{ "time": 0 }"#; 1000].join(",");
    let json = rig(
        &big_mesh(20_000),
        &format!(
            r#", "animations": {{ "d": {{ "deform": {{ "default": {{ "s": {{ "a": [{keys}] }} }} }} }} }}"#
        ),
    );
    assert!(schema_error(&json).contains("deform"));
}

#[test]
fn draw_order_keys_past_the_load_limit_are_rejected() {
    // Four thousand empty keys, each expanded to a 10,000-slot order (320 MB).
    let slots = (0..10_000)
        .map(|i| format!(r#"{{ "name": "s{i}", "bone": "root" }}"#))
        .collect::<Vec<_>>()
        .join(",");
    let keys = vec!["{}"; 4000].join(",");
    let json = format!(
        r#"{{ "bones": [ {{ "name": "root" }} ], "slots": [{slots}],
        "animations": {{ "o": {{ "drawOrder": [{keys}] }} }} }}"#
    );
    assert!(schema_error(&json).contains("draw order"));
}

#[test]
fn event_string_copies_past_the_load_limit_are_rejected() {
    // A thousand keys, each copying a 1 MB setup string (1 GB).
    let string = "x".repeat(1 << 20);
    let keys = vec![r#"{ "name": "e" }"#; 1000].join(",");
    let json = format!(
        r#"{{ "events": {{ "e": {{ "string": "{string}" }} }},
        "animations": {{ "fire": {{ "events": [{keys}] }} }} }}"#
    );
    assert!(schema_error(&json).contains("event"));
}

#[test]
fn linked_mesh_copies_past_the_load_limit_are_rejected() {
    // Five hundred links, each copying a 20,000-vertex parent (240 MB).
    let links: String = (0..500)
        .map(|i| format!(r#", "l{i}": {{ "type": "linkedmesh", "parent": "a" }}"#))
        .collect();
    let json = format!(
        r#"{{
        "bones": [ {{ "name": "root" }} ],
        "slots": [ {{ "name": "s", "bone": "root" }} ],
        "skins": [ {{ "name": "default", "attachments": {{ "s": {{ "a": {}{links} }} }} }} ]
    }}"#,
        big_mesh(20_000)
    );
    assert!(schema_error(&json).contains("linked mesh"));
}

#[test]
fn huge_physics_fps_keeps_a_usable_step() {
    // A step of 1/inf = 0 made the fixed-step physics loop endless.
    let json = r#"{
        "bones": [ { "name": "root" } ],
        "constraints": [ { "type": "physics", "name": "p", "bone": "root", "x": 1, "fps": 1e300 } ]
    }"#;
    let data = from_json(json).unwrap();
    let step = data.physics_constraints[0].step;
    assert!((step - 1.0 / 255.0).abs() < 1e-9, "step {step}");
}

#[test]
fn sequence_timeline_index_saturates() {
    // 2^32 used to wrap to index 0.
    let json = rig(
        r#"{ "type": "region", "sequence": { "count": 4 } }"#,
        r#", "animations": { "p": { "attachments": { "default": { "s": { "a": {
            "sequence": [ { "time": 0, "index": 4294967296 } ]
        } } } } } }"#,
    );
    let data = from_json(&json).unwrap();
    let anim = data.find_animation("p").unwrap().clone();
    let mut sk = crate::skel::Skeleton::new(Arc::new(data));
    sk.set_slots_to_setup_pose();
    anim.apply(&mut sk, 0.0, 0.0, 1.0, crate::anim::MixFrom::Setup, false);
    assert_eq!(sk.slot(0).unwrap().sequence_index, (u32::MAX >> 4) as i32);
}

#[test]
fn duplicate_names_resolve_to_the_first_entry() {
    // The name maps keep the first bone, slot, and constraint with a name,
    // as the linear searches did.
    let json = r#"{
        "bones": [ { "name": "a" }, { "name": "a", "parent": "a" }, { "name": "b", "parent": "a" } ],
        "slots": [ { "name": "s", "bone": "a" }, { "name": "s", "bone": "b" } ],
        "constraints": [
            { "type": "ik", "name": "k", "bones": [ "b" ], "target": "a" },
            { "type": "ik", "name": "k", "bones": [ "a" ], "target": "b" }
        ],
        "animations": { "x": {
            "bones": { "a": { "rotate": [ { "value": 10 } ] } },
            "slots": { "s": { "alpha": [ { "value": 0.5 } ] } },
            "ik": { "k": [ { "mix": 0.25 } ] }
        } }
    }"#;
    let data = from_json(json).unwrap();
    assert_eq!(data.bones[1].parent, Some(0));
    assert_eq!(data.bones[2].parent, Some(0));
    assert_eq!((data.slots[0].bone, data.slots[1].bone), (0, 2));
    assert_eq!(data.ik_constraints[0].target, 0);
    let anim = data.find_animation("x").unwrap().clone();
    let mut sk = crate::skel::Skeleton::new(Arc::new(data));
    sk.set_slots_to_setup_pose();
    anim.apply(&mut sk, 0.0, 0.0, 1.0, crate::anim::MixFrom::Setup, false);
    assert!((sk.bone(0).unwrap().rotation - 10.0).abs() < 1e-6);
    assert!((sk.bone(1).unwrap().rotation).abs() < 1e-6);
    assert!((sk.slot(0).unwrap().color.a - 0.5).abs() < 1e-6);
    assert!((sk.slot(1).unwrap().color.a - 1.0).abs() < 1e-6);
}
