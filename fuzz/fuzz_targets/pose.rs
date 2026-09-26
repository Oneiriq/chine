//! Load a skeleton and an atlas, then play it: skins, tracks, queued and mixed
//! animations, physics time, world transforms, and rendering, over several
//! frames.
//!
//! Input layout:
//!
//! | Bytes | Meaning |
//! |---|---|
//! | 0 | flags: bit 0 JSON instead of binary, bit 1 pick a skin, bit 2 switch skins mid-play, bit 7 raw floats |
//! | 1..3 | payload length, little-endian `u16` |
//! | 3..35 | control bytes: animation picks, loop flags, times, mix |
//! | 35.. | the skeleton payload, then atlas text in whatever follows it |
//!
//! Times and scales come from a small table of ordinary values unless bit 7 is
//! set, in which case they are raw `f32` bits (NaN, infinities, huge values).
//! Hosts pass these values in, so they must not panic or hang either.

#![no_main]

use std::sync::Arc;

use chine::anim::AnimationState;
use chine::atlas::Atlas;
use chine::render::{bind_atlas, render_with, RenderScratch};
use chine::skel::Skeleton;
use libfuzzer_sys::fuzz_target;

const CTRL: usize = 32;
const HEADER: usize = 3 + CTRL;

const TIMES: [f32; 16] = [
    0.0,
    1.0 / 60.0,
    1.0 / 30.0,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    -0.1,
    -1.0,
    10.0,
    0.001,
    0.2,
    0.75,
    1.5,
    5.0,
];

/// A time or scale from the control bytes at `at`: a table value, or raw
/// `f32` bits when `raw` is set.
fn value(ctrl: &[u8], at: usize, raw: bool) -> f32 {
    let byte = |i: usize| ctrl.get((at + i) % CTRL).copied().unwrap_or(0);
    if raw {
        f32::from_le_bytes([byte(0), byte(1), byte(2), byte(3)])
    } else {
        TIMES[usize::from(byte(0) % 16)]
    }
}

fuzz_target!(|input: &[u8]| {
    if input.len() < HEADER {
        return;
    }
    let flags = input[0];
    let len = usize::from(u16::from_le_bytes([input[1], input[2]]));
    let ctrl = &input[3..HEADER];
    let rest = &input[HEADER..];
    let (payload, atlas_text) = rest.split_at(len.min(rest.len()));
    let raw = flags & 0x80 != 0;

    let loaded = if flags & 1 == 0 {
        chine::binary::from_binary(payload).ok()
    } else {
        std::str::from_utf8(payload)
            .ok()
            .and_then(|text| chine::load::from_json(text).ok())
    };
    let Some(mut data) = loaded else {
        return;
    };
    let atlas = Atlas::parse(&String::from_utf8_lossy(atlas_text));
    bind_atlas(&mut data, &atlas);
    let data = Arc::new(data);

    let mut skeleton = Skeleton::new(Arc::clone(&data));
    if flags & 2 != 0 && !data.skins.is_empty() {
        let skin = &data.skins[usize::from(ctrl[0]) % data.skins.len()];
        skeleton.set_skin(&skin.name);
    }
    if ctrl[1] & 0x80 != 0 {
        skeleton.scale_x = -1.0;
    }
    if ctrl[1] & 0x40 != 0 {
        skeleton.scale_y = value(ctrl, 28, raw);
    }

    let animations = &data.animations;
    let mut state = AnimationState::new();
    state.default_mix = value(ctrl, 2, raw);
    if !animations.is_empty() {
        let pick = |b: u8| Arc::clone(&animations[usize::from(b) % animations.len()]);
        let tracks = 1 + usize::from(ctrl[3] % 3);
        for track in 0..tracks {
            let entry = state.set_animation_on(track, pick(ctrl[4 + track]), ctrl[7] >> track & 1 == 1);
            entry.time_scale = if ctrl[8] >> track & 1 == 1 {
                value(ctrl, 9 + track, raw)
            } else {
                1.0
            };
            if ctrl[12] >> track & 1 == 1 {
                entry.alpha = value(ctrl, 13 + track, raw);
            }
            if ctrl[16] >> track & 1 == 1 {
                state.add_animation_on(track, pick(ctrl[17 + track]), ctrl[20] >> track & 1 == 1);
            }
        }
    }

    let mut scratch = RenderScratch::new();
    let frames = 1 + usize::from(ctrl[21] % 6);
    for frame in 0..frames {
        let dt = value(ctrl, 22 + frame, raw);
        if frame == 2 && ctrl[29] & 1 == 1 && !animations.is_empty() {
            // Replace track 0 mid-play so the crossfade path runs.
            let next = Arc::clone(&animations[usize::from(ctrl[30]) % animations.len()]);
            state.set_animation_on(0, next, ctrl[31] & 1 == 1);
        }
        if frame == 1 && flags & 4 != 0 {
            // Switch to the next skin, or back to the default skin after the
            // last one, as a host does between frames.
            let next = (usize::from(ctrl[0]) + 1) % (data.skins.len() + 1);
            match data.skins.get(next) {
                Some(skin) => skeleton.set_skin(&skin.name),
                None => skeleton.clear_skin(),
            }
        }
        state.update(dt);
        skeleton.set_bones_to_setup_pose();
        state.apply(&mut skeleton);
        skeleton.update(dt);
        skeleton.update_world_transform();
        let _ = render_with(&skeleton, &mut scratch);
    }
});
