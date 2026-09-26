//! Regression tests for corrupt `.skel` data: every loaded index is checked
//! against its table, curve storage is sized from what the data can back, and
//! mutated input never panics the loader.

use std::sync::Arc;

use super::*;

/// A byte writer for hand-built `.skel` data.
#[derive(Default)]
struct Out(Vec<u8>);

impl Out {
    fn byte(&mut self, b: u8) {
        self.0.push(b);
    }

    /// A var_uint (7 bits per byte, high bit continues).
    fn var(&mut self, mut v: u32) {
        while v >= 0x80 {
            self.0.push((v & 0x7f) as u8 | 0x80);
            v >>= 7;
        }
        self.0.push(v as u8);
    }

    fn u32(&mut self, v: u32) {
        self.0.extend_from_slice(&v.to_be_bytes());
    }

    fn f32(&mut self, v: f32) {
        self.0.extend_from_slice(&v.to_be_bytes());
    }

    fn floats(&mut self, values: &[f32]) {
        for &v in values {
            self.f32(v);
        }
    }

    /// A length-prefixed string (`None` is length 0).
    fn str(&mut self, s: Option<&str>) {
        match s {
            None => self.var(0),
            Some(s) => {
                self.var(u32::try_from(s.len() + 1).unwrap());
                self.0.extend_from_slice(s.as_bytes());
            }
        }
    }
}

/// A small essential rig that exercises every section the loader reads:
/// three bones, two slots, one constraint of each type, a default skin with a
/// region, a weighted mesh, and a clipping attachment, a named skin, an event,
/// and one animation with every timeline group. Each field is a value a test
/// can corrupt. The defaults describe valid data.
struct Rig {
    /// Parents of bones 1 and 2.
    parents: [u32; 2],
    /// Bones of slots 0 and 1.
    slot_bones: [u32; 2],
    ik_target: u32,
    transform_source: u32,
    path_slot: u32,
    physics_bone: u32,
    slider_bone: u32,
    /// The string reference of slot 0's setup attachment.
    setup_attachment: u32,
    /// The flags byte of the default skin's region (the type is its low bits).
    region_flags: u8,
    /// The slot index of the default skin's first entry.
    skin_slot: u32,
    /// The bone of the weighted mesh's first influence.
    weighted_bone: u32,
    /// The length of the weighted mesh's `[influences, bone...]` layout.
    weighted_total: u32,
    /// The mesh's third triangle index.
    triangle: u32,
    clip_end: u32,
    slot_timeline_slot: u32,
    bone_timeline_bone: u32,
    ik_timeline_index: u32,
    /// Frame count of the rotate timeline.
    rotate_frames: u32,
    /// Declared Bezier count of the rotate timeline.
    rotate_beziers: u32,
    /// Curve type between the rotate timeline's two frames.
    rotate_curve: u8,
    deform_skin: u32,
    /// Where the second deform key's two-value run starts (the mesh has six
    /// deform values).
    deform_start: u32,
    /// The physics constraint's step divisor (frames per second).
    physics_fps: u8,
    /// The draw order key's `(slot, offset)` moves.
    draw_order: Vec<(u32, u32)>,
    event_index: u32,
    slider_animation: u32,
}

impl Default for Rig {
    fn default() -> Self {
        Self {
            parents: [0, 1],
            slot_bones: [1, 2],
            ik_target: 2,
            transform_source: 1,
            path_slot: 1,
            physics_bone: 1,
            slider_bone: 1,
            setup_attachment: 1,
            region_flags: 0,
            skin_slot: 0,
            weighted_bone: 1,
            weighted_total: 6,
            triangle: 2,
            clip_end: 1,
            slot_timeline_slot: 0,
            bone_timeline_bone: 1,
            ik_timeline_index: 0,
            rotate_frames: 2,
            rotate_beziers: 1,
            rotate_curve: 2,
            deform_skin: 0,
            deform_start: 0,
            physics_fps: 30,
            draw_order: vec![(0, 1)],
            event_index: 0,
            slider_animation: 0,
        }
    }
}

impl Rig {
    fn bytes(&self) -> Vec<u8> {
        let mut o = Out::default();
        // Header: hash, version, bounds, reference scale, essential.
        o.0.extend_from_slice(&[0; 8]);
        o.str(Some("4.3.00"));
        o.floats(&[0.0, 0.0, 100.0, 100.0, 1.0]);
        o.byte(0);
        // String table.
        o.var(3);
        for s in ["att", "mesh", "clip"] {
            o.str(Some(s));
        }

        // Bones: root, then "a" and "b".
        o.var(3);
        for (i, name) in ["root", "a", "b"].into_iter().enumerate() {
            o.str(Some(name));
            if i > 0 {
                o.var(self.parents[i - 1]);
            }
            // rotation, x, y, scaleX, scaleY, shearX, shearY, length.
            o.floats(&[0.0, 10.0, 0.0, 1.0, 1.0, 0.0, 0.0, 5.0]);
            o.byte(0); // inherit
            o.byte(0); // skin required
        }

        // Slots.
        o.var(2);
        for (i, name) in ["s0", "s1"].into_iter().enumerate() {
            o.str(Some(name));
            o.var(self.slot_bones[i]);
            o.u32(0xFFFF_FFFF); // color
            o.u32(0xFFFF_FFFF); // no dark color
            o.var(if i == 0 { self.setup_attachment } else { 0 });
            o.var(0); // blend
        }

        // Constraints, one of each type (unified indices 0 to 4).
        o.var(5);
        o.str(Some("ik"));
        o.byte(0);
        o.var(1);
        o.var(1);
        o.var(self.ik_target);
        o.byte(0); // flags
        o.str(Some("path"));
        o.byte(1);
        o.var(1);
        o.var(2);
        o.var(self.path_slot);
        o.byte(0); // flags
        o.floats(&[0.0, 0.0, 1.0, 1.0, 1.0]);
        o.str(Some("tc"));
        o.byte(2);
        o.var(1);
        o.var(2);
        o.var(self.transform_source);
        o.byte(0); // flags: no property mappings
        o.byte(0); // offset flags
        o.byte(0); // mix flags
        o.str(Some("phys"));
        o.byte(3);
        o.var(self.physics_bone);
        o.byte(0); // flags
        o.byte(self.physics_fps); // step divisor
        o.floats(&[0.5, 100.0, 0.9, 0.0, 0.0]); // inertia, strength, damping, wind, gravity
        o.byte(0); // global flags
        o.str(Some("slider"));
        o.byte(4);
        o.byte(64); // flags: bone property
        o.var(self.slider_bone);
        o.f32(0.0); // property offset
        o.byte(0); // property: rotate
        o.f32(0.0); // offset
        o.f32(1.0); // scale

        // Default skin: slot 0 holds a region and a weighted mesh, slot 1 a
        // clipping attachment.
        o.var(2);
        o.var(self.skin_slot);
        o.var(2);
        o.var(1); // placeholder "att"
        o.byte(self.region_flags); // region
        o.floats(&[0.0, 0.0, 1.0, 1.0, 10.0, 10.0]);
        o.var(2); // placeholder "mesh"
        o.byte(2 | 128); // weighted mesh
        o.var(3); // hull
        o.var(3); // vertex count
        o.var(self.weighted_total);
        for bone in [self.weighted_bone, 1, 2] {
            o.var(1); // one influence
            o.var(bone);
            o.floats(&[1.0, 2.0, 1.0]);
        }
        o.floats(&[0.0, 0.0, 1.0, 0.0, 0.0, 1.0]); // uvs
        o.var(0);
        o.var(1);
        o.var(self.triangle);
        o.var(0); // timeline slots
        o.var(1);
        o.var(1);
        o.var(3); // placeholder "clip"
        o.byte(6); // clipping
        o.var(self.clip_end);
        o.var(3); // vertex count
        o.floats(&[0.0, 0.0, 1.0, 0.0, 0.0, 1.0]);

        // One named skin with a bone, a constraint, and a region.
        o.var(1);
        o.str(Some("extra"));
        o.var(1);
        o.var(1);
        o.var(1);
        o.var(0);
        o.var(1);
        o.var(1);
        o.var(1);
        o.var(1);
        o.byte(0);
        o.floats(&[0.0, 0.0, 1.0, 1.0, 10.0, 10.0]);

        // Events.
        o.var(1);
        o.str(Some("ev"));
        o.var(0); // int
        o.f32(0.0);
        o.str(None); // string
        o.str(None); // audio path

        // One animation.
        o.var(1);
        o.str(Some("anim"));
        o.var(0); // timeline count (unused)

        // Slot timelines: attachment, then RGBA with a Bezier curve.
        o.var(1);
        o.var(self.slot_timeline_slot);
        o.var(2);
        o.byte(0);
        o.var(2);
        o.f32(0.0);
        o.var(1);
        o.f32(1.0);
        o.var(0);
        o.byte(1);
        o.var(2);
        o.var(4);
        o.f32(0.0);
        o.0.extend_from_slice(&[255, 255, 255, 255]);
        o.f32(1.0);
        o.0.extend_from_slice(&[0, 0, 0, 0]);
        o.byte(2);
        for _ in 0..4 {
            o.floats(&[0.25, 0.0, 0.75, 1.0]);
        }

        // Bone timelines: rotate, then translate with Bezier curves.
        o.var(1);
        o.var(self.bone_timeline_bone);
        o.var(2);
        o.byte(0);
        o.var(self.rotate_frames);
        o.var(self.rotate_beziers);
        o.floats(&[0.0, 0.0]);
        if self.rotate_frames > 1 {
            o.floats(&[1.0, 90.0]);
            o.byte(self.rotate_curve);
            if self.rotate_curve == 2 {
                o.floats(&[0.25, 0.0, 0.75, 90.0]);
            }
        }
        o.byte(1);
        o.var(2);
        o.var(2);
        o.floats(&[0.0, 0.0, 0.0, 1.0, 5.0, 5.0]);
        o.byte(2);
        o.floats(&[0.25, 0.0, 0.75, 5.0, 0.25, 0.0, 0.75, 5.0]);

        // IK timeline with a Bezier curve (mix and softness).
        o.var(1);
        o.var(self.ik_timeline_index);
        o.var(2);
        o.var(2);
        o.byte(0);
        o.f32(0.0);
        o.byte(1 | 128);
        o.f32(1.0);
        o.floats(&[0.25, 0.0, 0.75, 1.0, 0.25, 0.0, 0.75, 0.0]);

        // Transform timeline (constraint 2), keying a rotate mix of 0.5.
        o.var(1);
        o.var(2);
        o.var(1);
        o.var(0);
        o.floats(&[0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0]);

        // Path position timeline (constraint 1).
        o.var(1);
        o.var(1);
        o.var(1);
        o.byte(0);
        o.var(1);
        o.var(0);
        o.floats(&[0.0, 0.5]);

        // Physics timelines: a global reset, then an inertia of 0.25 on
        // constraint 3.
        o.var(2);
        o.var(0);
        o.var(1);
        o.byte(8);
        o.var(1);
        o.f32(0.5);
        o.var(4);
        o.var(1);
        o.byte(0);
        o.var(1);
        o.var(0);
        o.floats(&[0.0, 0.25]);

        // Slider time timeline (constraint 4).
        o.var(1);
        o.var(4);
        o.var(1);
        o.byte(0);
        o.var(1);
        o.var(0);
        o.floats(&[0.0, 0.5]);

        // Deform timeline on the weighted mesh, with a Bezier curve.
        o.var(1);
        o.var(self.deform_skin);
        o.var(1);
        o.var(0);
        o.var(1);
        o.var(2); // "mesh"
        o.byte(0);
        o.var(2);
        o.var(1);
        o.f32(0.0);
        o.var(0); // setup vertices
        o.f32(1.0);
        o.byte(2);
        o.floats(&[0.25, 0.0, 0.75, 1.0]);
        o.var(2);
        o.var(self.deform_start);
        o.floats(&[3.0, 4.0]);

        // Draw order: one key.
        o.var(1);
        o.f32(0.5);
        o.var(u32::try_from(self.draw_order.len()).unwrap());
        for &(slot, offset) in &self.draw_order {
            o.var(slot);
            o.var(offset);
        }
        o.var(0); // draw order folders

        // Event timeline.
        o.var(1);
        o.f32(0.25);
        o.var(self.event_index);
        o.var(0);
        o.f32(0.0);
        o.str(None);

        // Slider animation index.
        o.var(self.slider_animation);
        o.0
    }

    fn load(&self) -> Result<SkeletonData, BinaryError> {
        from_binary(&self.bytes())
    }
}

/// Load `rig` and require it to be rejected as corrupt.
#[track_caller]
fn assert_corrupt(rig: &Rig) {
    match rig.load() {
        Err(BinaryError::CorruptLength) => {}
        Err(other) => panic!("expected CorruptLength, got {other:?}"),
        Ok(_) => panic!("corrupt data loaded"),
    }
}

#[test]
fn valid_rig_loads_and_plays() {
    let data = Rig::default().load().expect("the valid rig loads");
    assert_eq!(data.bones.len(), 3);
    assert_eq!(data.bones[2].parent, Some(1));
    assert_eq!(data.slots[0].attachment.as_deref(), Some("att"));
    assert_eq!(data.ik_constraints[0].target, 2);
    assert_eq!(data.transform_constraints[0].source, 1);
    assert_eq!(data.path_constraints[0].slot, 1);
    assert_eq!(data.physics_constraints[0].bone, 1);
    assert_eq!(data.sliders[0].bone, Some(1));
    assert_eq!(data.sliders[0].animation_index, Some(0));
    assert_eq!(data.default_skin.iter().count(), 3);
    assert_eq!(data.skins.len(), 1);
    assert_eq!(data.events.len(), 1);
    assert_eq!(data.animations.len(), 1);
    assert!((data.animations[0].duration() - 1.0).abs() < 1e-6);
    match data.default_skin.attachment(0, "mesh") {
        Some(Attachment::Mesh(m)) => assert_eq!(m.triangles, vec![0, 1, 2]),
        other => panic!("expected the weighted mesh, got {other:?}"),
    }

    // Past the draw order key the two slots swap, and the event fires.
    let anim = Arc::clone(&data.animations[0]);
    let mut sk = crate::skel::Skeleton::new(Arc::new(data));
    let mut state = crate::anim::AnimationState::new();
    state.set_animation(anim, false);
    state.update(0.75);
    sk.set_slots_to_setup_pose();
    state.apply(&mut sk);
    sk.update_world_transform();
    assert_eq!(sk.draw_order(), &[1, 0]);
    assert_eq!(sk.events().len(), 1);
    assert_eq!(sk.events()[0].name, "ev");
}

// Spine 4.3 constraint timelines index the one list that holds every
// constraint type. The rig's path (1), transform (2), physics (3), and slider
// (4) timelines were looked up by that index in their own type's list, where
// each constraint is at index 0, so they drove nothing.
#[test]
fn constraint_timelines_index_the_single_constraint_list() {
    let data = Rig::default().load().expect("the valid rig loads");
    let anim = Arc::clone(&data.animations[0]);
    let mut sk = crate::skel::Skeleton::new(Arc::new(data));
    anim.apply(&mut sk, -1.0, 0.5, 1.0, crate::anim::MixFrom::Setup, false);
    let (path, _) = sk.path_pose_and_setup(0).unwrap();
    assert_eq!(path.position, 0.5);
    let (transform, _) = sk.transform_pose_and_setup(0).unwrap();
    assert_eq!(transform.mix_rotate, 0.5);
    let (physics, _) = sk.physics_pose_and_setup(0).unwrap();
    assert_eq!(physics.inertia, 0.25);
    let (slider, _) = sk.slider_pose_and_setup(0).unwrap();
    assert_eq!(slider.time, 0.5);
}

// A constraint timeline whose index names a constraint of another type
// cannot be applied. Spine's reader rejects it too.
#[test]
fn constraint_timeline_of_the_wrong_type_is_rejected() {
    // Index 1 is the path constraint.
    assert_corrupt(&Rig {
        ik_timeline_index: 1,
        ..Rig::default()
    });
}

// Bone 2's parent 0xFFFF_FFFF was stored raw and indexed out of bounds when
// the skeleton built its update cache.
#[test]
fn bone_parent_out_of_range_is_rejected() {
    assert_corrupt(&Rig {
        parents: [0, u32::MAX],
        ..Rig::default()
    });
}

// Bones 1 and 2 naming each other as parents made the update cache recurse
// until the stack overflowed.
#[test]
fn bone_parent_cycle_is_rejected() {
    assert_corrupt(&Rig {
        parents: [2, 1],
        ..Rig::default()
    });
}

// A parent at the bone's own index is also a forward reference.
#[test]
fn bone_parent_self_reference_is_rejected() {
    assert_corrupt(&Rig {
        parents: [1, 1],
        ..Rig::default()
    });
}

// Out-of-range bone and slot references in slots and constraints were stored
// raw, then indexed out of bounds by the update cache and constraint solvers.
#[test]
fn slot_and_constraint_references_out_of_range_are_rejected() {
    let rigs = [
        Rig {
            slot_bones: [1, 3],
            ..Rig::default()
        },
        Rig {
            ik_target: 3,
            ..Rig::default()
        },
        Rig {
            transform_source: 99,
            ..Rig::default()
        },
        Rig {
            path_slot: 2,
            ..Rig::default()
        },
        Rig {
            physics_bone: 3,
            ..Rig::default()
        },
        Rig {
            slider_bone: 3,
            ..Rig::default()
        },
    ];
    for rig in &rigs {
        assert_corrupt(rig);
    }
}

// A curve timeline with zero frames made Curve::new subtract with overflow.
#[test]
fn zero_frame_curve_timeline_is_rejected() {
    assert_corrupt(&Rig {
        rotate_frames: 0,
        rotate_beziers: 0,
        ..Rig::default()
    });
}

// A Bezier curve beyond the declared Bezier count wrote past the curve
// storage.
#[test]
fn bezier_past_the_declared_count_is_rejected() {
    assert_corrupt(&Rig {
        rotate_beziers: 0,
        ..Rig::default()
    });
}

// Declaring more Bezier curves than a timeline uses only oversizes the
// storage, so it still loads.
#[test]
fn surplus_bezier_count_still_loads() {
    let rig = Rig {
        rotate_beziers: 3,
        ..Rig::default()
    };
    assert!(rig.load().is_ok());
}

// Every curve timeline allocated storage for its declared Bezier count up
// front. Thousands of one-frame timelines that each declared tens of thousands
// of curves allocated gigabytes from a few dozen kilobytes.
#[test]
fn declared_bezier_counts_do_not_drive_allocation() {
    const TIMELINES: u32 = 3000;
    const BEZIERS: u32 = 60_000;
    let mut o = Out::default();
    o.0.extend_from_slice(&[0; 8]);
    o.str(None);
    o.floats(&[0.0, 0.0, 0.0, 0.0, 1.0]);
    o.byte(0);
    o.var(0); // strings
    o.var(1); // root bone
    o.str(None);
    o.floats(&[0.0; 8]);
    o.byte(0);
    o.byte(0);
    o.var(0); // slots
    o.var(0); // constraints
    o.var(0); // default skin
    o.var(0); // skins
    o.var(0); // events
    o.var(1); // animations
    o.str(None);
    o.var(0);
    o.var(0); // slot timelines
    o.var(1); // bone timelines on the root
    o.var(0);
    o.var(TIMELINES);
    for _ in 0..TIMELINES {
        o.byte(0); // rotate
        o.var(1); // one frame
        o.var(BEZIERS);
        o.floats(&[0.0, 0.0]);
    }
    o.0.extend_from_slice(&[0; 9]); // the remaining groups are empty
                                    // Padding so every declared count passes the remaining-bytes check.
    o.0.resize(o.0.len() + usize::try_from(BEZIERS).unwrap(), 0);
    assert!(from_binary(&o.0).is_ok());
}

// A draw order key that moved slot 0 twice overran the unchanged-slot buffer.
#[test]
fn draw_order_with_a_repeated_slot_is_rejected() {
    assert_corrupt(&Rig {
        draw_order: vec![(0, 1), (0, 0)],
        ..Rig::default()
    });
}

// An offset near i32::MAX overflowed the position arithmetic.
#[test]
fn draw_order_offset_overflow_is_rejected() {
    assert_corrupt(&Rig {
        draw_order: vec![(1, 0x7FFF_FFFF)],
        ..Rig::default()
    });
}

// Moves that leave the slot list, name a missing slot, or land two slots on
// one position left `usize::MAX` holes in the draw order.
#[test]
fn draw_order_that_is_not_a_permutation_is_rejected() {
    for moves in [
        vec![(0, 2)],
        vec![(0, u32::MAX - 1)],
        vec![(2, 0)],
        vec![(0, 1), (1, 0)],
    ] {
        assert_corrupt(&Rig {
            draw_order: moves,
            ..Rig::default()
        });
    }
}

// A negative in-range offset is valid.
#[test]
fn draw_order_with_a_negative_offset_loads() {
    let rig = Rig {
        draw_order: vec![(1, u32::MAX)],
        ..Rig::default()
    };
    assert!(rig.load().is_ok());
}

// Timeline targets past their tables were kept, and the runtime skipped them
// silently. Spine's reader cannot load them either.
#[test]
fn timeline_targets_out_of_range_are_rejected() {
    let rigs = [
        Rig {
            slot_timeline_slot: 2,
            ..Rig::default()
        },
        Rig {
            bone_timeline_bone: 3,
            ..Rig::default()
        },
        Rig {
            ik_timeline_index: 5,
            ..Rig::default()
        },
        Rig {
            deform_skin: 2,
            ..Rig::default()
        },
        Rig {
            event_index: 1,
            ..Rig::default()
        },
        Rig {
            slider_animation: 1,
            ..Rig::default()
        },
    ];
    for rig in &rigs {
        assert_corrupt(rig);
    }
}

// A weighted vertex naming a missing bone, or a weighted layout describing
// fewer vertices than the mesh declares, made vertex skinning read past the
// layout at render time.
#[test]
fn weighted_vertices_are_validated() {
    assert_corrupt(&Rig {
        weighted_bone: 3,
        ..Rig::default()
    });
    // A layout of 4 entries holds two vertices, not the declared three.
    assert_corrupt(&Rig {
        weighted_total: 4,
        ..Rig::default()
    });
    // A layout of 5 entries ends inside the third vertex.
    assert_corrupt(&Rig {
        weighted_total: 5,
        ..Rig::default()
    });
}

// Triangle indices past the vertex count were passed to rendering.
#[test]
fn mesh_triangle_out_of_range_is_rejected() {
    assert_corrupt(&Rig {
        triangle: 3,
        ..Rig::default()
    });
    assert_corrupt(&Rig {
        triangle: 70_000,
        ..Rig::default()
    });
}

// Skin slots, clipping end slots, and string references past their tables
// are corrupt.
#[test]
fn skin_references_out_of_range_are_rejected() {
    let rigs = [
        Rig {
            skin_slot: 2,
            ..Rig::default()
        },
        Rig {
            clip_end: 2,
            ..Rig::default()
        },
        Rig {
            setup_attachment: 4,
            ..Rig::default()
        },
    ];
    for rig in &rigs {
        assert_corrupt(rig);
    }
}

// A deform run that ends past the mesh's deform values was silently cut
// short. A start near u32::MAX also overflowed the offset index on 32-bit
// targets.
#[test]
fn deform_run_past_the_mesh_is_rejected() {
    // The run's last value lands at index 6 of a six-value deform.
    assert_corrupt(&Rig {
        deform_start: 5,
        ..Rig::default()
    });
    assert_corrupt(&Rig {
        deform_start: u32::MAX,
        ..Rig::default()
    });
    // A run that ends exactly at the last value is valid.
    let rig = Rig {
        deform_start: 4,
        ..Rig::default()
    };
    assert!(rig.load().is_ok());
}

// A step divisor of 0 gives an infinite physics step, as in Spine's reader.
// It loads, and posing it finishes without integrating.
#[test]
fn zero_physics_fps_loads_and_poses() {
    let data = Rig {
        physics_fps: 0,
        ..Rig::default()
    }
    .load()
    .expect("a zero step divisor loads");
    assert!(data.physics_constraints[0].step.is_infinite());
    let mut sk = crate::skel::Skeleton::new(Arc::new(data));
    for _ in 0..10 {
        sk.update(1.0 / 60.0);
        sk.update_world_transform();
    }
    assert!(sk.bone(1).is_some());
}

// Attachment type 7 does not exist, and its fields cannot be skipped, so the
// rest of the skin was read misaligned.
#[test]
fn unknown_attachment_type_is_rejected() {
    assert_corrupt(&Rig {
        region_flags: 7,
        ..Rig::default()
    });
}

// An unknown transform or slider property ordinal left the constraint's
// remaining fields unread, misaligning every later read.
#[test]
fn unknown_constraint_property_is_corrupt() {
    // One bone, source 0, one mapping from property 9.
    let mut r = BinaryReader::new(&[1, 0, 0, 32, 9, 0, 0]);
    let _ = parse_transform(&mut r, "tc".into(), 0);
    assert_eq!(r.error(), Some(&BinaryError::CorruptLength));
    // A mapping from rotation to property 9.
    let mut b = vec![1_u8, 0, 0, 32, 0];
    b.extend_from_slice(&0.0_f32.to_be_bytes());
    b.extend_from_slice(&[1, 9, 0, 0]);
    let mut r = BinaryReader::new(&b);
    let _ = parse_transform(&mut r, "tc".into(), 0);
    assert_eq!(r.error(), Some(&BinaryError::CorruptLength));
    // A slider driven by bone 0's property 9.
    let mut b = vec![64_u8, 0];
    b.extend_from_slice(&0.0_f32.to_be_bytes());
    b.push(9);
    let mut r = BinaryReader::new(&b);
    let _ = parse_slider(&mut r, "slider".into(), 0, false);
    assert_eq!(r.error(), Some(&BinaryError::CorruptLength));
}

// A cut-off export is an error at every length, never a panic or a partial
// rig.
#[test]
fn every_truncation_is_an_error() {
    let bytes = Rig::default().bytes();
    for len in 0..bytes.len() {
        assert!(
            from_binary(&bytes[..len]).is_err(),
            "{len}-byte prefix loaded"
        );
    }
}

// Flipping any single bit of a valid rig must not panic the loader, and
// anything that still loads must build a skeleton and pose it.
#[test]
fn single_bit_flips_never_panic() {
    let bytes = Rig::default().bytes();
    for i in 0..bytes.len() {
        for bit in 0..8 {
            let mut mutated = bytes.clone();
            mutated[i] ^= 1 << bit;
            if let Ok(data) = from_binary(&mutated) {
                let mut sk = crate::skel::Skeleton::new(Arc::new(data));
                sk.update_world_transform();
            }
        }
    }
}
