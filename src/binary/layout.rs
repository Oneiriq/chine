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
        // rotation, x, y, scaleX, scaleY, shearX, shearY, length.
        self.floats(&[0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]);
        self.byte(0); // inherit
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
