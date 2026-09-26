//! Tests for Spine 4.3 binary layout details that the older section tests do
//! not cover: flag-coded constraint defaults.

use super::*;

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
