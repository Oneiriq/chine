//! Attachment timelines: mesh deform and sequence (flipbook) keys, read from
//! the nested skin, slot, and attachment groups of an animation.

use super::*;

/// Consume one mesh-deform timeline frame: a run of changed vertices encoded as
/// a count, a start offset, then that many float offsets (count `0` means the
/// frame uses the setup vertices).
fn skip_deform_frame(r: &mut BinaryReader) {
    let end = r.count();
    if end != 0 {
        let _start = r.var_usize();
        for _ in 0..end {
            r.float();
        }
    }
}

/// Consume a mesh-deform timeline (Bezier count, then per-frame time and vertex
/// offsets, with stepped / linear / Bezier curves) without building anything.
pub(super) fn skip_deform_timeline(r: &mut BinaryReader, frames: usize) {
    let _bezier_count = r.var_usize();
    let last = frames.saturating_sub(1);
    r.float(); // first frame time
    skip_deform_frame(r);
    for frame in 0..frames {
        if frame == last {
            break;
        }
        r.float(); // time
        if r.byte() == 2 {
            // Bezier: four control floats.
            for _ in 0..4 {
                r.float();
            }
        }
        skip_deform_frame(r);
    }
}

/// The number of regions in a sequenced region/mesh attachment, for a sequence
/// timeline's index wrapping.
pub(super) fn sequence_count(
    data: &SkeletonData,
    skin_index: usize,
    slot: usize,
    name: &str,
) -> Option<usize> {
    let skin = if skin_index == 0 {
        None
    } else {
        data.skins.get(skin_index - 1)
    };
    match data.attachment(slot, name, skin) {
        Some(Attachment::Region(r)) => r.sequence.as_ref().map(|s| s.count),
        Some(Attachment::Mesh(m)) => m.sequence.as_ref().map(|s| s.count),
        _ => None,
    }
}

/// Read a sequence (flipbook) timeline: per frame a time, a packed mode-and-index
/// int, and a delay. Stepped, so there is no curve data.
pub(super) fn read_sequence_timeline(
    r: &mut BinaryReader,
    slot: usize,
    attachment: String,
    count: usize,
    frames: usize,
) -> (SequenceTimeline, f32) {
    let mut times = Vec::with_capacity(frames);
    let mut mode_and_index = Vec::with_capacity(frames);
    let mut delays = Vec::with_capacity(frames);
    let mut duration = 0.0_f32;
    for _ in 0..frames {
        let time = r.float();
        mode_and_index.push(r.u32());
        delays.push(r.float());
        times.push(time);
        duration = duration.max(time);
    }
    (
        SequenceTimeline::new(slot, attachment, count, times, mode_and_index, delays),
        duration,
    )
}

/// Resolve a deform timeline's mesh: the setup-pose deform length, the setup
/// vertices to add at apply time (zeros for a weighted mesh), and the
/// timeline's skin name (`None` for the default skin). Returns `None` when no
/// matching mesh is found, so the caller consumes the bytes instead.
pub(super) fn deform_mesh_info(
    data: &SkeletonData,
    skin_index: usize,
    slot: usize,
    name: &str,
) -> Option<(usize, Vec<f32>, Option<String>)> {
    let skin = if skin_index == 0 {
        None
    } else {
        data.skins.get(skin_index - 1)
    };
    let Some(Attachment::Mesh(mesh)) = data.attachment(slot, name, skin) else {
        return None;
    };
    let frame_len = mesh.deform_len();
    let setup = mesh
        .setup_vertices()
        .map_or_else(|| vec![0.0; frame_len], <[f32]>::to_vec);
    Some((frame_len, setup, skin.map(|s| s.name.clone())))
}

/// Read a mesh-deform timeline into chine's relative-offset model: per frame a
/// time and a sparse run of vertex offsets (zeros elsewhere; the setup vertices
/// are added at apply time), with stepped / linear / Bezier curves. The offsets
/// are read raw, never adding the setup, which matches the JSON loader.
pub(super) fn read_deform_timeline(
    r: &mut BinaryReader,
    slot: usize,
    attachment: String,
    skin: Option<String>,
    setup: Vec<f32>,
    frame_len: usize,
    frames: usize,
) -> (DeformTimeline, f32) {
    let bezier_count = r.count();
    let last = frames.saturating_sub(1);
    let mut times = Vec::with_capacity(frames);
    let mut offsets = Vec::with_capacity(frames);
    let mut segments: Vec<(u8, [f32; 4])> = Vec::new();
    let mut time = r.float();
    for frame in 0..frames {
        let mut deform = vec![0.0_f32; frame_len];
        let end = r.count();
        if end != 0 {
            let start = r.var_usize();
            for i in 0..end {
                let value = r.float();
                if let Some(v) = deform.get_mut(start + i) {
                    *v = value;
                }
            }
        }
        times.push(time);
        offsets.push(deform);
        if frame == last {
            break;
        }
        let time2 = r.float();
        let segment = match r.byte() {
            1 => (1, [0.0; 4]),
            2 => (2, [r.float(), r.float(), r.float(), r.float()]),
            _ => (0, [0.0; 4]),
        };
        segments.push(segment);
        time = time2;
    }
    let duration = times.last().copied().unwrap_or(0.0);
    let mut tl = DeformTimeline::new(
        slot,
        attachment,
        skin,
        setup,
        times.clone(),
        offsets,
        bezier_count,
    );
    let mut bezier = 0;
    for (frame, &(kind, c)) in segments.iter().enumerate() {
        match kind {
            1 => tl.set_stepped(frame),
            2 => {
                tl.set_bezier(
                    bezier,
                    frame,
                    0,
                    times[frame],
                    0.0,
                    c[0],
                    c[1],
                    c[2],
                    c[3],
                    times[frame + 1],
                    1.0,
                );
                bezier += 1;
            }
            _ => {}
        }
    }
    (tl, duration)
}
