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

/// The attachment an attachment timeline drives: `name` in `slot` of `skin`
/// (`None` for the default skin, else an index into `data.skins`). Spine looks
/// only in that skin. A mesh's timeline slots join the target.
pub(super) fn timeline_target<'d>(
    data: &'d SkeletonData,
    skin: Option<usize>,
    slot: usize,
    name: &str,
) -> Option<(AttachmentTarget, &'d Attachment)> {
    let (skin_name, skin) = match skin {
        None => (None, &data.default_skin),
        Some(i) => {
            let skin = data.skins.get(i)?;
            (Some(skin.name.clone()), skin)
        }
    };
    let attachment = skin.attachment(slot, name)?;
    let timeline_slots = match attachment {
        Attachment::Mesh(mesh) => Arc::clone(&mesh.timeline_slots),
        _ => Vec::new().into(),
    };
    let key = AttachmentKey {
        skin: skin_name,
        slot,
        name: name.to_string(),
    };
    Some((AttachmentTarget::new(key, timeline_slots), attachment))
}

/// A sequence timeline's keys: the times, the packed modes and indices,
/// and the delays.
pub(super) type SequenceKeys = (Vec<f32>, Vec<u32>, Vec<f32>);

/// Read a sequence (flipbook) timeline's keys: per frame a time, a packed
/// mode-and-index int, and a delay. Stepped, so there is no curve data.
/// Returns the keys and the duration.
pub(super) fn read_sequence_keys(r: &mut BinaryReader, frames: usize) -> (SequenceKeys, f32) {
    let frames = fitting_frames(r, frames, 12);
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
    ((times, mode_and_index, delays), duration)
}

/// Resolve a deform timeline's mesh: the target, the setup-pose deform
/// length, and the setup vertices to add at apply time (zeros for a weighted
/// mesh). Returns `None` when the target is not a mesh, so the caller
/// consumes the bytes instead.
pub(super) fn deform_mesh_info(
    (target, attachment): (AttachmentTarget, &Attachment),
) -> Option<(AttachmentTarget, usize, Vec<f32>)> {
    let Attachment::Mesh(mesh) = attachment else {
        return None;
    };
    let frame_len = mesh.deform_len();
    let setup = mesh
        .setup_vertices()
        .map_or_else(|| vec![0.0; frame_len], <[f32]>::to_vec);
    Some((target, frame_len, setup))
}

/// Read a mesh-deform timeline into chine's relative-offset model: per frame a
/// time and a sparse run of vertex offsets (zeros elsewhere, and the setup vertices
/// are added at apply time), with stepped / linear / Bezier curves. The offsets
/// are read raw, never adding the setup, which matches the JSON loader. A run
/// that ends past the mesh's `frame_len` deform values is corrupt.
pub(super) fn read_deform_timeline(
    r: &mut BinaryReader,
    target: AttachmentTarget,
    setup: Vec<f32>,
    frame_len: usize,
    frames: usize,
) -> (DeformTimeline, f32) {
    let bezier_count = r.count();
    let (frames, beziers) = curve_sizes(r, frames, bezier_count, 5, 1);
    let last = frames - 1;
    let mut times = Vec::with_capacity(frames);
    let mut offsets = Vec::with_capacity(frames);
    let mut segments: Vec<(u8, [f32; 4])> = Vec::new();
    let mut time = r.float();
    for frame in 0..frames {
        let mut deform = vec![0.0_f32; frame_len];
        let end = r.count();
        if end != 0 {
            let start = r.var_usize();
            let in_mesh = matches!(start.checked_add(end), Some(run_end) if run_end <= frame_len);
            if !in_mesh {
                corrupt(r);
            }
            for i in 0..end {
                let value = r.float();
                if let Some(v) = start.checked_add(i).and_then(|k| deform.get_mut(k)) {
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
    let mut tl = DeformTimeline::new(target, setup, times.clone(), offsets, beziers);
    let mut bezier = 0;
    // Segment `frame` runs from `times[frame]` to `times[frame + 1]`.
    let spans = times.iter().zip(times.iter().skip(1));
    for (frame, (&(kind, c), (&time1, &time2))) in segments.iter().zip(spans).enumerate() {
        match kind {
            1 => tl.set_stepped(frame),
            2 => {
                if let Some(b) = next_bezier(r, &mut bezier, beziers) {
                    let [cx1, cy1, cx2, cy2] = c;
                    tl.set_bezier(b, frame, 0, time1, 0.0, cx1, cy1, cx2, cy2, time2, 1.0);
                }
            }
            _ => {}
        }
    }
    (tl, duration)
}
