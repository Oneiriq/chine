//! Tests for Spine 4.3 binary layout details that the older section tests do
//! not cover: flag-coded constraint defaults and bone inherit timelines.

use std::sync::Arc;

use super::*;
use crate::anim::MixFrom;
use crate::skel::Skeleton;

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

    fn floats(&mut self, values: &[f32]) {
        for v in values {
            self.0.extend_from_slice(&v.to_be_bytes());
        }
    }

    /// A length-prefixed string.
    fn str(&mut self, s: &str) {
        self.var(u32::try_from(s.len() + 1).unwrap());
        self.0.extend_from_slice(s.as_bytes());
    }

    /// An essential 4.3 header with an empty string table.
    fn header(&mut self) {
        self.0.extend_from_slice(&[0; 8]);
        self.str("4.3.00");
        self.floats(&[0.0, 0.0, 100.0, 100.0, 1.0]);
        self.byte(0); // essential
        self.var(0); // string table
    }

    /// A bone at the setup pose with the `Normal` inherit mode. The root has
    /// no parent.
    fn bone(&mut self, name: &str, parent: Option<u32>) {
        self.str(name);
        if let Some(parent) = parent {
            self.var(parent);
        }
        // rotation, x, y, scaleX, scaleY, shearX, shearY.
        self.floats(&[0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]);
        self.byte(0); // inherit
        self.floats(&[0.0]); // length
        self.byte(0); // skin required
    }

    /// The timeline groups after the bone timelines, all empty: IK,
    /// transform, path, physics, slider, attachment, draw order, draw order
    /// folders, and events.
    fn no_more_timelines(&mut self) {
        for _ in 0..9 {
            self.var(0);
        }
    }
}

// Spine writes an IK, transform, or slider mix only when it is not 0, and
// marks a mix of exactly 1 with a flag and no value. A clear flag read as a
// mix of 1, so a disabled constraint applied in full.
#[test]
fn unset_constraint_mixes_are_zero() {
    // IK: 1 bone, bone 0, target 0, then the flags.
    let ik = |flags: u8| {
        let bytes = [1, 0, 0, flags];
        let mut r = BinaryReader::new(&bytes);
        parse_ik(&mut r, "ik".into(), 0).mix
    };
    assert_eq!(ik(0), 0.0);
    assert_eq!(ik(32), 1.0);

    // Transform: 1 bone, bone 0, source 0, flags, offset flags, mix flags.
    let mut r = BinaryReader::new(&[1, 0, 0, 0, 0, 0]);
    let tc = parse_transform(&mut r, "tc".into(), 0);
    let mixes = [
        tc.mix_rotate,
        tc.mix_x,
        tc.mix_y,
        tc.mix_scale_x,
        tc.mix_scale_y,
        tc.mix_shear_y,
    ];
    assert_eq!(mixes, [0.0; 6]);

    // Slider: the flags only (no bone property).
    let slider = |flags: u8| {
        let bytes = [flags];
        let mut r = BinaryReader::new(&bytes);
        parse_slider(&mut r, "slider".into(), 0, false).mix
    };
    assert_eq!(slider(0), 0.0);
    assert_eq!(slider(16), 1.0);
}

/// A root and a child bone, and one animation that keys the child's inherit
/// mode with `modes` (time, ordinal) keys.
fn inherit_rig(modes: &[(f32, u8)]) -> Vec<u8> {
    let mut o = Out::default();
    o.header();
    o.var(2);
    o.bone("root", None);
    o.bone("child", Some(0));
    o.var(0); // slots
    o.var(0); // constraints
    o.var(0); // default skin
    o.var(0); // named skins
    o.var(0); // events
    o.var(1); // animations
    o.str("a");
    o.var(1); // timeline count
    o.var(0); // slot timelines
    o.var(1); // bone timelines: one bone
    o.var(1); // the child
    o.var(1); // one timeline
    o.byte(10); // inherit
    o.var(u32::try_from(modes.len()).unwrap());
    for &(time, mode) in modes {
        o.floats(&[time]);
        o.byte(mode);
    }
    o.no_more_timelines();
    o.0
}

// Bone inherit timelines (type 10) were rejected as an unknown timeline type,
// so an export that used them failed to load.
#[test]
fn bone_inherit_timelines_load_and_apply() {
    let data = from_binary(&inherit_rig(&[(0.0, 1), (1.0, 3)])).expect("the rig loads");
    let anim = Arc::clone(&data.animations[0]);
    assert_eq!(anim.duration(), 1.0);
    let mut sk = Skeleton::new(Arc::new(data));
    anim.apply(&mut sk, -1.0, 0.5, 1.0, MixFrom::Setup, false);
    assert_eq!(sk.bone(1).unwrap().inherit(), Inherit::OnlyTranslation);
    anim.apply(&mut sk, -1.0, 1.0, 1.0, MixFrom::Setup, false);
    assert_eq!(sk.bone(1).unwrap().inherit(), Inherit::NoScale);
}

// Spine defines five inherit modes. Its reader cannot load a key past them.
#[test]
fn unknown_inherit_mode_is_corrupt() {
    assert!(matches!(
        from_binary(&inherit_rig(&[(0.0, 5)])),
        Err(BinaryError::CorruptLength)
    ));
}

/// String table references (one-based) in the linked mesh rigs.
const ARM: u32 = 1;
const SLEEVE: u32 = 2;
const GIRL_SLEEVE: u32 = 3;

impl Out {
    /// An essential header whose string table holds "arm", "sleeve", and
    /// "girl/sleeve", then the root bone and two slots on it: slot 0 showing
    /// "arm" and slot 1 showing "sleeve". No constraints follow.
    fn link_rig_start(&mut self) {
        self.0.extend_from_slice(&[0; 8]);
        self.str("4.3.00");
        self.floats(&[0.0, 0.0, 100.0, 100.0, 1.0]);
        self.byte(0); // essential
        self.var(3);
        for s in ["arm", "sleeve", "girl/sleeve"] {
            self.str(s);
        }
        self.var(1);
        self.bone("root", None);
        self.var(2);
        for (name, attachment) in [("s0", ARM), ("s1", SLEEVE)] {
            self.str(name);
            self.var(0); // bone
            self.0.extend_from_slice(&[0xFF; 8]); // white, no dark color
            self.var(attachment);
            self.var(0); // blend
        }
        self.var(0); // constraints
    }

    /// Skin entries for slot 0 holding "arm": a triangle mesh whose timelines
    /// also reach slot 1.
    fn arm_slot(&mut self) {
        self.var(0); // slot
        self.var(1); // attachments
        self.var(ARM);
        self.byte(2); // mesh
        self.var(3); // hull
        self.var(3); // vertex count
        self.floats(&[0.0, 0.0, 10.0, 0.0, 0.0, 10.0]);
        self.floats(&[0.0, 0.0, 1.0, 0.0, 0.0, 1.0]); // uvs
        for corner in [0, 1, 2] {
            self.var(corner);
        }
        self.var(1); // timeline slots
        self.var(1);
    }

    /// Skin entries for slot 1 holding "sleeve": a linked mesh named
    /// "girl/sleeve" with a three-frame sequence, inheriting the timelines of
    /// "arm" in slot 0 of the skin at `skin_index` in Spine's skin list.
    fn sleeve_slot(&mut self, skin_index: u32) {
        self.var(1); // slot
        self.var(1); // attachments
        self.var(SLEEVE);
        self.byte(3 | 8 | 64 | 128); // linked mesh, name, sequence, timelines
        self.var(GIRL_SLEEVE);
        for value in [3, 1, 2, 0] {
            self.var(value); // sequence count, start, digits, setup
        }
        self.var(0); // source slot
        self.var(skin_index);
        self.var(ARM);
    }

    /// A named skin with no bones or constraints, holding what `slots`
    /// writes for `slot_count` slots.
    fn named_skin(&mut self, name: &str, slot_count: u32, slots: impl FnOnce(&mut Self)) {
        self.str(name);
        self.var(0); // bones
        self.var(0); // constraints
        self.var(slot_count);
        slots(self);
    }

    /// No events, and one animation "flap" that moves vertex 1 of "arm" (in
    /// the skin at `skin_index`) by 5 in x at time 1.
    fn flap(&mut self, skin_index: u32) {
        self.var(0); // events
        self.var(1);
        self.str("flap");
        self.var(0); // timeline count
        for _ in 0..7 {
            self.var(0); // slot to slider timelines
        }
        self.var(1); // skins
        self.var(skin_index);
        self.var(1); // slots
        self.var(0);
        self.var(1); // attachments
        self.var(ARM);
        self.byte(0); // deform
        self.var(2); // frames
        self.var(0); // beziers
        self.floats(&[0.0]);
        self.var(0); // setup vertices
        self.floats(&[1.0]);
        self.byte(0); // linear
        self.var(2);
        self.var(2);
        self.floats(&[5.0, 0.0]);
        for _ in 0..3 {
            self.var(0); // draw order, draw order folders, events
        }
    }
}

/// The resolved mesh `name` in `slot` of `skin`.
fn mesh<'a>(skin: &'a Skin, slot: usize, name: &str) -> &'a MeshAttachment {
    match skin.attachment(slot, name) {
        Some(Attachment::Mesh(mesh)) => mesh,
        other => panic!("{name} is not a resolved mesh: {other:?}"),
    }
}

/// Play "flap" to its end with `skin` active, and return each slot's deform.
fn flapped(data: SkeletonData, skin: &str) -> Vec<Vec<f32>> {
    let anim = Arc::clone(&data.animations[0]);
    let mut sk = Skeleton::new(Arc::new(data));
    sk.set_skin(skin);
    anim.apply(&mut sk, -1.0, 1.0, 1.0, MixFrom::Setup, false);
    sk.slots().iter().map(|slot| slot.deform.clone()).collect()
}

// The loader read a linked mesh's source slot, source skin, and sequence and
// dropped them. A link to a mesh in another skin then never resolved, so it
// drew nothing. The mesh's timeline slots were dropped too, so the source's
// deforms never reached the link in its own slot.
#[test]
fn linked_meshes_keep_their_source_skin_slot_and_sequence() {
    let mut o = Out::default();
    o.link_rig_start();
    o.var(1); // default skin slots
    o.arm_slot();
    o.var(1); // named skins
    o.named_skin("boy", 1, |o| o.sleeve_slot(0));
    o.flap(0);
    let data = from_binary(&o.0).expect("the rig loads");

    let arm = mesh(&data.default_skin, 0, "arm");
    assert_eq!(&arm.timeline_slots[..], [1]);
    let sleeve = mesh(&data.skins[0], 1, "sleeve");
    assert_eq!(sleeve.path, "girl/sleeve");
    assert_eq!(sleeve.triangles, [0, 1, 2]);
    assert_eq!(sleeve.sequence.as_ref().map(|s| s.count), Some(3));
    let arm_key = AttachmentKey {
        skin: None,
        slot: 0,
        name: "arm".into(),
    };
    assert_eq!(sleeve.timeline_source, Some(arm_key));

    // The deform of "arm" reaches the sleeve in slot 1 as well.
    let deforms = flapped(data, "boy");
    for deform in &deforms {
        assert_eq!(deform, &[0.0, 0.0, 15.0, 0.0, 0.0, 10.0]);
    }
}

// Binary data indexes skins in Spine's skin list, which leaves out a default
// skin that lists no slots. The loader always read index 0 as the default
// skin, so here the link and the deform named the wrong skin.
#[test]
fn skin_indices_skip_an_empty_default_skin() {
    let mut o = Out::default();
    o.link_rig_start();
    o.var(0); // the default skin lists no slots
    o.var(2); // named skins
    o.named_skin("base", 1, Out::arm_slot);
    o.named_skin("girl", 1, |o| o.sleeve_slot(0));
    o.flap(0);
    let data = from_binary(&o.0).expect("the rig loads");

    let sleeve = mesh(&data.skins[1], 1, "sleeve");
    let base_arm = AttachmentKey {
        skin: Some("base".into()),
        slot: 0,
        name: "arm".into(),
    };
    assert_eq!(sleeve.timeline_source, Some(base_arm));
    // With "girl" active, slot 0 shows nothing and the sleeve still deforms.
    let deforms = flapped(data, "girl");
    assert!(deforms[0].is_empty());
    assert_eq!(deforms[1], [0.0, 0.0, 15.0, 0.0, 0.0, 10.0]);
}

// A skin index past Spine's skin list is corrupt.
#[test]
fn linked_mesh_skin_past_the_skin_list_is_corrupt() {
    let mut o = Out::default();
    o.link_rig_start();
    o.var(1);
    o.arm_slot();
    o.var(1);
    o.named_skin("boy", 1, |o| o.sleeve_slot(2));
    o.flap(0);
    assert_eq!(from_binary(&o.0).err(), Some(BinaryError::CorruptLength));
}

/// A rig of four slots whose animation holds one draw order folder timeline
/// over `folder`, with one key at time 0 that makes `moves`.
fn folder_rig(folder: &[u32], moves: &[(u32, u32)]) -> Vec<u8> {
    let mut o = Out::default();
    o.header();
    o.var(1);
    o.bone("root", None);
    o.var(4);
    for name in ["a", "b", "c", "d"] {
        o.str(name);
        o.var(0); // bone
        o.0.extend_from_slice(&[0xFF; 8]); // white, no dark color
        o.var(0); // no attachment
        o.var(0); // blend
    }
    o.var(0); // constraints
    o.var(0); // default skin
    o.var(0); // named skins
    o.var(0); // events
    o.var(1);
    o.str("a");
    o.var(0); // timeline count
    for _ in 0..9 {
        o.var(0); // slot to attachment timelines, and the draw order
    }
    o.var(1); // draw order folders
    o.var(u32::try_from(folder.len()).unwrap());
    for &slot in folder {
        o.var(slot);
    }
    o.var(1); // keys
    o.floats(&[0.0]);
    o.var(u32::try_from(moves.len()).unwrap());
    for &(position, offset) in moves {
        o.var(position);
        o.var(offset);
    }
    o.var(0); // events
    o.0
}

// Draw order folder timelines reorder a folder of slots among the positions
// those slots hold. The loader read past them, so the folder kept its setup
// order.
#[test]
fn draw_order_folder_timelines_load_and_apply() {
    let data = from_binary(&folder_rig(&[1, 3], &[(0, 1)])).expect("the rig loads");
    let anim = Arc::clone(&data.animations[0]);
    let mut sk = Skeleton::new(Arc::new(data));
    anim.apply(&mut sk, -1.0, 0.0, 1.0, MixFrom::Setup, false);
    assert_eq!(sk.draw_order(), [0, 3, 2, 1]);
}

// Spine never exports a folder slot past the slot table, a folder that lists
// a slot twice, or a key whose moves do not order the folder.
#[test]
fn malformed_draw_order_folders_are_corrupt() {
    let cases: [(&[u32], &[(u32, u32)]); 4] = [
        (&[1, 9], &[]),
        (&[1, 1], &[]),
        (&[1, 3], &[(0, 2)]),
        (&[1, 3], &[(0, 1), (0, 1)]),
    ];
    for (folder, moves) in cases {
        let loaded = from_binary(&folder_rig(folder, moves));
        assert_eq!(
            loaded.err(),
            Some(BinaryError::CorruptLength),
            "{folder:?} {moves:?}"
        );
    }
}
