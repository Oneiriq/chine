//! Regression tests for corrupt timeline data and hostile host values. Each
//! case panicked, overflowed, or indexed out of bounds before the matching fix.

use std::sync::Arc;

use super::curve::Curve;
use super::*;
use crate::attach::{Attachment, AttachmentKey, RegionAttachment, Sequence};
use crate::data::{BlendMode, BoneData, Color, SkeletonData, SlotData};
use crate::event::Event;
use crate::skin::Skin;

/// One bone and one slot showing the attachment `att`, a region with a
/// four-frame sequence.
fn rig() -> Skeleton {
    rig_with_sequence(4)
}

/// [`rig`] with a sequence of `count` frames.
fn rig_with_sequence(count: usize) -> Skeleton {
    let mut region = RegionAttachment::new("att", "att");
    region.sequence = Some(Sequence::new(count, 0, 0, 0));
    let mut skin = Skin::new("default");
    skin.set(0, "att", Attachment::Region(region));
    let data = SkeletonData {
        default_skin: skin,
        bones: vec![BoneData {
            index: 0,
            name: "root".into(),
            ..Default::default()
        }],
        slots: vec![SlotData {
            index: 0,
            name: "slot".into(),
            bone: 0,
            color: Color::new(1.0, 1.0, 1.0, 1.0),
            dark_color: None,
            attachment: Some("att".into()),
            blend: BlendMode::Normal,
        }],
        ..Default::default()
    };
    Skeleton::new(Arc::new(data))
}

fn apply(timeline: &Timeline, sk: &mut Skeleton, time: f32) {
    timeline.apply(sk, time - 0.5, time, 1.0, MixFrom::Setup, false, false);
}

/// The attachment `att` in `slot` of the default skin, as a timeline target.
fn target(slot: usize) -> AttachmentTarget {
    let key = AttachmentKey {
        skin: None,
        slot,
        name: "att".into(),
    };
    AttachmentTarget::new(key, Vec::new().into())
}

fn event() -> Event {
    Event {
        name: "hit".into(),
        time: 0.0,
        int_value: 0,
        float_value: 0.0,
        string_value: String::new(),
        volume: 1.0,
        balance: 0.0,
    }
}

#[test]
fn zero_frame_curve_reads_as_empty() {
    let mut c = Curve::new(0, 3, 2);
    c.set_frame1(0, 0.0, 1.0);
    c.set_stepped(0);
    c.set_bezier(0, 0, 0, 0.0, 0.0, 0.25, 0.0, 0.75, 1.0, 1.0, 1.0);
    assert_eq!(c.first_time(), None);
    assert_eq!(c.value(0.5, 1), 0.0);
    assert_eq!(c.percent(0.5), 0.0);
    assert_eq!(c.frame_value(0.5, 1), 0.0);
}

#[test]
fn bezier_index_past_the_declared_count_is_skipped() {
    // The export declares no Bezier segments but then keys one.
    let mut c = Curve::new(2, 0, 2);
    c.set_frame1(0, 0.0, 0.0);
    c.set_frame1(1, 1.0, 90.0);
    c.set_bezier(0, 0, 0, 0.0, 0.0, 0.25, 0.0, 0.75, 90.0, 1.0, 90.0);
    c.set_bezier(7, 0, 1, 0.0, 0.0, 0.25, 0.0, 0.75, 90.0, 1.0, 90.0);
    // The frame stays linear.
    assert!((c.value(0.5, 1) - 45.0).abs() < 1e-4);
}

#[test]
fn frame_index_past_the_frame_count_is_skipped() {
    let mut c = Curve::new(2, 0, 2);
    c.set_frame1(0, 0.0, 0.0);
    c.set_frame1(1, 1.0, 90.0);
    c.set_frame1(2, 5.0, 5.0);
    c.set_frame2(1, 5.0, 5.0, 5.0);
    c.set_frame_n(3, 5.0, &[5.0]);
    c.set_frame_n(1, 1.0, &[90.0, 5.0, 5.0]);
    c.set_stepped(9);
    c.set_bezier(0, 9, 0, 0.0, 0.0, 0.25, 0.0, 0.75, 90.0, 1.0, 90.0);
    assert!((c.value(0.5, 1) - 45.0).abs() < 1e-4);
    assert!((c.value(3.0, 1) - 90.0).abs() < 1e-4);
}

#[test]
fn huge_declared_counts_do_not_overflow() {
    let c = Curve::new(usize::MAX, usize::MAX, 3);
    assert_eq!(c.first_time(), None);
    let mut c = Curve::new(2, usize::MAX, 2);
    c.set_frame1(0, 0.0, 0.0);
    c.set_frame1(1, 1.0, 90.0);
    assert!((c.value(0.5, 1) - 45.0).abs() < 1e-4);
}

#[test]
fn percent_at_nan_time_or_the_last_frame_is_zero() {
    let mut c = Curve::new(2, 0, 2);
    c.set_frame1(0, 0.0, 0.0);
    c.set_frame1(1, 1.0, 0.0);
    assert_eq!(c.percent(f32::NAN), 0.0);
    assert_eq!(c.percent(5.0), 0.0);
}

#[test]
fn bezier_on_the_last_frame_holds_its_value() {
    let mut c = Curve::new(2, 2, 2);
    c.set_frame1(0, 0.0, 0.0);
    c.set_frame1(1, 1.0, 90.0);
    c.set_bezier(0, 1, 0, 1.0, 90.0, 1.25, 90.0, 1.75, 90.0, 2.0, 90.0);
    assert!((c.value(3.0, 1) - 90.0).abs() < 1e-4);
    assert_eq!(c.percent(3.0), 0.0);
}

#[test]
fn zero_entry_curve_does_not_divide_by_zero() {
    let c = Curve::new(3, 0, 0);
    assert_eq!(c.value(0.0, 1), 0.0);
    assert_eq!(c.percent(0.0), 0.0);
}

#[test]
fn draw_order_skips_a_slot_past_the_end() {
    assert_eq!(compute_draw_order(4, &mut [(7, -1)]), vec![0, 1, 2, 3]);
}

#[test]
fn draw_order_skips_a_slot_listed_twice() {
    assert_eq!(
        compute_draw_order(2, &mut [(0, 0), (0, 0), (1, 0)]),
        vec![0, 1]
    );
}

#[test]
fn draw_order_skips_an_offset_that_overflows() {
    assert_eq!(
        compute_draw_order(3, &mut [(1, i32::MAX)]),
        vec![usize::MAX, 0, 2]
    );
}

#[test]
fn draw_order_matches_spine_for_valid_offsets() {
    assert_eq!(
        compute_draw_order(4, &mut [(1, 2), (3, -3)]),
        vec![3, 0, 2, 1]
    );
}

#[test]
fn zero_frame_timelines_leave_the_pose_alone() {
    let mut sk = rig();
    sk.bone_mut(0).unwrap().rotation = 7.0;
    let timelines = [
        Timeline::Rotate(BoneTimeline::one_value(0, 0, 0)),
        Timeline::Translate(BoneTimeline::two_value(0, 0, 4)),
        Timeline::Scale(BoneTimeline::two_value(0, 0, 0)),
        Timeline::Shear(BoneTimeline::two_value(0, 0, 0)),
        Timeline::BoneAxis(BoneTimeline::one_value(0, 0, 0), BoneAxis::ScaleX),
        Timeline::SlotColor(ConstraintTimeline::new(0, 0, 0, 5), true),
        Timeline::SlotAlpha(ConstraintTimeline::new(0, 0, 0, 2)),
        Timeline::SlotTwoColor(ConstraintTimeline::new(0, 0, 0, 8), true),
        Timeline::Deform(DeformTimeline::new(
            target(0),
            vec![0.0; 4],
            Vec::new(),
            Vec::new(),
            0,
        )),
    ];
    for timeline in &timelines {
        apply(timeline, &mut sk, 0.5);
    }
    let bone = sk.bone(0).unwrap();
    assert_eq!(bone.rotation, 7.0);
    assert_eq!((bone.scale_x, bone.x), (1.0, 0.0));
}

#[test]
fn out_of_range_indices_leave_the_skeleton_alone() {
    let mut sk = rig();
    let mut rotate = BoneTimeline::one_value(9, 1, 0);
    rotate.set_frame1(0, 0.0, 45.0);
    let mut color = ConstraintTimeline::new(9, 1, 0, 5);
    color.set_frame(0, 0.0, &[0.0, 0.0, 0.0, 0.0]);
    let timelines = [
        Timeline::Rotate(rotate),
        Timeline::SlotColor(color.clone(), true),
        Timeline::Ik(color.clone()),
        Timeline::TransformMix(color.clone()),
        Timeline::PathMix(color.clone()),
        Timeline::Physics(color.clone(), PhysicsProperty::Mass),
        Timeline::SliderTime(color),
        Timeline::PhysicsReset(PhysicsResetTimeline::new(9, vec![0.0])),
        Timeline::Attachment(AttachmentTimeline::new(9, vec![0.0], vec![None])),
        Timeline::Sequence(SequenceTimeline::new(
            target(9),
            vec![0.0],
            vec![2],
            vec![0.1],
        )),
    ];
    for timeline in &timelines {
        apply(timeline, &mut sk, 0.5);
    }
    assert_eq!(sk.bone(0).unwrap().rotation, 0.0);
    assert_eq!(sk.slot(0).unwrap().attachment.as_deref(), Some("att"));
}

#[test]
fn stepped_timelines_with_short_value_arrays_are_skipped() {
    let mut sk = rig();
    let attachment = Timeline::Attachment(AttachmentTimeline::new(
        0,
        vec![0.0, 1.0],
        vec![Some("other".into())],
    ));
    apply(&attachment, &mut sk, 1.5);
    assert_eq!(sk.slot(0).unwrap().attachment.as_deref(), Some("att"));

    let draw_order = Timeline::DrawOrder(DrawOrderTimeline::new(vec![0.0, 1.0], vec![vec![0]]));
    apply(&draw_order, &mut sk, 1.5);
    assert_eq!(sk.draw_order(), &[0]);

    // Both keys fall in the window, but only the first has an event.
    let events = Timeline::Event(EventTimeline::new(vec![0.25, 0.75], vec![event()]));
    events.apply(&mut sk, 0.0, 1.0, 1.0, MixFrom::Setup, false, false);
    assert_eq!(sk.events().len(), 1);
}

#[test]
fn deform_timeline_with_short_frames_or_nan_time() {
    let mut sk = rig();
    let short = Timeline::Deform(DeformTimeline::new(
        target(0),
        vec![0.0; 4],
        vec![0.0, 1.0],
        vec![vec![1.0; 4]],
        0,
    ));
    apply(&short, &mut sk, 0.5);
    assert_eq!(sk.slot(0).unwrap().deform, vec![1.0; 4]);
    apply(&short, &mut sk, 2.0);
    assert_eq!(sk.slot(0).unwrap().deform, vec![0.0; 4]);

    let full = Timeline::Deform(DeformTimeline::new(
        target(0),
        vec![0.0; 4],
        vec![0.0, 1.0],
        vec![vec![0.0; 4], vec![2.0; 4]],
        0,
    ));
    apply(&full, &mut sk, f32::NAN);
    assert_eq!(sk.slot(0).unwrap().deform.len(), 4);
}

#[test]
fn sequence_timeline_with_zero_delay_or_huge_count() {
    let mut sk = rig();
    // Loop mode, start index 1, zero delay: the advance saturates.
    let zero_delay = Timeline::Sequence(SequenceTimeline::new(
        target(0),
        vec![0.0],
        vec![(1 << 4) | 2],
        vec![0.0],
    ));
    apply(&zero_delay, &mut sk, 1.0);
    assert_eq!(sk.slot(0).unwrap().sequence_index, 3);

    // Ping-pong reverse over a count near `usize::MAX`.
    let huge = Timeline::Sequence(SequenceTimeline::new(
        target(0),
        vec![0.0],
        vec![(5 << 4) | 6],
        vec![0.1],
    ));
    let mut huge_rig = rig_with_sequence(usize::MAX);
    apply(&huge, &mut huge_rig, 1.0);
    apply(&huge, &mut huge_rig, f32::INFINITY);

    // More times than modes or delays.
    let short = Timeline::Sequence(SequenceTimeline::new(
        target(0),
        vec![0.0, 1.0],
        vec![(1 << 4) | 2],
        vec![0.1],
    ));
    apply(&short, &mut sk, 1.5);
}

#[test]
fn hostile_host_values_do_not_panic() {
    let mut rotate = BoneTimeline::one_value(0, 2, 0);
    rotate.set_frame1(0, 0.0, 0.0);
    rotate.set_frame1(1, 1.0, 90.0);
    let deform = DeformTimeline::new(
        target(0),
        vec![0.0; 4],
        vec![0.0, 1.0],
        vec![vec![0.0; 4], vec![2.0; 4]],
        0,
    );
    let sequence = SequenceTimeline::new(target(0), vec![0.0], vec![2], vec![0.0]);
    let anim = Arc::new(Animation::new(
        "hostile",
        1.0,
        vec![
            Timeline::Rotate(rotate),
            Timeline::Deform(deform),
            Timeline::Sequence(sequence),
            Timeline::Event(EventTimeline::new(vec![0.5], vec![event()])),
        ],
    ));
    let values = [
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::MAX,
        -1.0,
        0.0,
    ];
    for &v in &values {
        let mut sk = rig();
        let mut state = AnimationState::new();
        state.default_mix = v;
        state.set_animation(Arc::clone(&anim), true);
        state.add_animation(Arc::clone(&anim), false);
        state
            .set_animation_on(1, Arc::clone(&anim), true)
            .time_scale = v;
        let entry = state.set_animation_on(2, Arc::clone(&anim), false);
        entry.alpha = v;
        entry.track_time = v;
        state.add_animation_on(2, Arc::clone(&anim), true);
        for &dt in &values {
            state.update(dt);
            state.apply(&mut sk);
        }
    }
}
