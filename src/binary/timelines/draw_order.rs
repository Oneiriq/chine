//! Draw order timelines: the skeleton's draw order, and (new in Spine 4.3)
//! the draw order of folders of slots.

use super::*;

/// Read a draw-order timeline: per frame a time and a set of slot moves (each a
/// slot index and an offset), resolved into a full slot permutation. A frame
/// with no moves keeps the setup order.
pub(in crate::binary) fn read_draw_order_timeline(
    r: &mut BinaryReader,
    frames: usize,
    slot_count: usize,
) -> (DrawOrderTimeline, f32) {
    let frames = fitting_frames(r, frames, 5);
    let mut times = Vec::with_capacity(frames);
    let mut orders = Vec::with_capacity(frames);
    let mut duration = 0.0_f32;
    for _ in 0..frames {
        let time = r.float();
        let change_count = r.count();
        let mut offsets: Vec<(usize, i32)> = Vec::with_capacity(change_count);
        for _ in 0..change_count {
            let slot = r.var_usize();
            // The offset is a signed 32-bit value stored as a var_uint, so
            // this cast reinterprets its bits.
            let offset = r.var_uint() as i32;
            offsets.push((slot, offset));
        }
        let order = if offsets.is_empty() {
            (0..slot_count).collect()
        } else {
            resolve_draw_order(r, slot_count, &mut offsets)
        };
        times.push(time);
        orders.push(order);
        duration = duration.max(time);
    }
    (DrawOrderTimeline::new(times, orders), duration)
}

/// Resolve one draw-order key's slot moves into a full slot permutation.
///
/// Spine moves each slot at most once and keeps it inside the slot list, so
/// the key must name distinct slots in range, move each to a position in
/// range, and move no two slots to the same position. Any other key is
/// corrupt: this records the error and keeps the setup order.
fn resolve_draw_order(
    r: &mut BinaryReader,
    slot_count: usize,
    offsets: &mut [(usize, i32)],
) -> Vec<usize> {
    let in_range = |&(slot, offset): &(usize, i32)| {
        i32::try_from(slot_count).is_ok()
            && slot < slot_count
            && isize::try_from(offset)
                .ok()
                .and_then(|offset| slot.checked_add_signed(offset))
                .is_some_and(|position| position < slot_count)
    };
    offsets.sort_by_key(|&(slot, _)| slot);
    let distinct = offsets
        .iter()
        .zip(offsets.iter().skip(1))
        .all(|(a, b)| a.0 < b.0);
    if distinct && offsets.iter().all(in_range) {
        let order = compute_draw_order(slot_count, offsets);
        // Two slots moved to one position leave a gap the permutation
        // cannot fill, marked by `usize::MAX`.
        if !order.contains(&usize::MAX) {
            return order;
        }
    }
    corrupt(r);
    (0..slot_count).collect()
}

/// Read the draw order folder timelines: per folder, its slots in setup
/// order, then per key a time and a set of moves over positions in the
/// folder, as a draw order key stores them. A folder slot past the slot
/// table, a slot listed twice, or a key whose moves do not order the folder
/// is corrupt. A folder with no slots or no keys has nothing to drive and
/// is dropped.
pub(super) fn read_draw_order_folders(
    r: &mut BinaryReader,
    slot_count: usize,
    timelines: &mut Vec<Timeline>,
) -> f32 {
    let mut duration = 0.0_f32;
    let folder_count = r.count();
    for _ in 0..folder_count {
        let folder_size = r.count();
        let slots: Vec<usize> = (0..folder_size).map(|_| r.var_usize()).collect();
        let mut sorted = slots.clone();
        sorted.sort_unstable();
        let distinct = sorted.windows(2).all(|pair| pair[0] < pair[1]);
        if !distinct || !all_below(&slots, slot_count) {
            corrupt(r);
        }
        let key_count = r.count();
        let key_count = fitting_frames(r, key_count, 5);
        let mut times = Vec::with_capacity(key_count);
        let mut keys = Vec::with_capacity(key_count);
        for _ in 0..key_count {
            let time = r.float();
            let change_count = r.count();
            let mut moves: Vec<(usize, i32)> = (0..change_count)
                .map(|_| {
                    let position = r.var_usize();
                    // A signed 32-bit offset stored as a var_uint.
                    (position, r.var_uint() as i32)
                })
                .collect();
            if !sort_draw_order_moves(slots.len(), &mut moves) {
                corrupt(r);
                moves.clear();
            }
            times.push(time);
            keys.push(moves);
            duration = duration.max(time);
        }
        if !slots.is_empty() && !keys.is_empty() {
            let folder = DrawOrderFolderTimeline::new(slots, times, keys);
            timelines.push(Timeline::DrawOrderFolder(folder));
        }
    }
    duration
}
