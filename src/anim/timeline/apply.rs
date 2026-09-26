//! Apply functions for bone, constraint, physics reset, and event timelines.
//!
//! A timeline whose bone or constraint index is out of range for the skeleton,
//! or whose curve has no frames, leaves the skeleton unchanged.

use super::*;

pub(super) fn apply_rotate(
    t: &BoneTimeline,
    skel: &mut Skeleton,
    time: f32,
    alpha: f32,
    from: MixFrom,
    add: bool,
) {
    if let Some((bone, setup)) = skel.bone_and_setup(t.bone) {
        bone.rotation =
            t.curve
                .relative_value(time, alpha, from, add, bone.rotation, setup.rotation);
    }
}

pub(super) fn apply_translate(
    t: &BoneTimeline,
    skel: &mut Skeleton,
    time: f32,
    alpha: f32,
    from: MixFrom,
    add: bool,
) {
    let Some((bone, setup)) = skel.bone_and_setup(t.bone) else {
        return;
    };
    let Some(first) = t.curve.first_time() else {
        return;
    };
    if time < first {
        match from {
            MixFrom::Setup => {
                bone.x = setup.position.x;
                bone.y = setup.position.y;
            }
            MixFrom::First => {
                bone.x += (setup.position.x - bone.x) * alpha;
                bone.y += (setup.position.y - bone.y) * alpha;
            }
            MixFrom::Current => {}
        }
        return;
    }
    let x = t.curve.value(time, 1);
    let y = t.curve.value(time, 2);
    if matches!(from, MixFrom::Setup) {
        bone.x = setup.position.x + x * alpha;
        bone.y = setup.position.y + y * alpha;
    } else if add {
        bone.x += x * alpha;
        bone.y += y * alpha;
    } else {
        bone.x += (setup.position.x + x - bone.x) * alpha;
        bone.y += (setup.position.y + y - bone.y) * alpha;
    }
}

pub(super) fn apply_scale(
    t: &BoneTimeline,
    skel: &mut Skeleton,
    time: f32,
    alpha: f32,
    from: MixFrom,
    add: bool,
    out: bool,
) {
    let Some((bone, setup)) = skel.bone_and_setup(t.bone) else {
        return;
    };
    let Some(first) = t.curve.first_time() else {
        return;
    };
    if time < first {
        match from {
            MixFrom::Setup => {
                bone.scale_x = setup.scale.x;
                bone.scale_y = setup.scale.y;
            }
            MixFrom::First => {
                bone.scale_x += (setup.scale.x - bone.scale_x) * alpha;
                bone.scale_y += (setup.scale.y - bone.scale_y) * alpha;
            }
            MixFrom::Current => {}
        }
        return;
    }
    let x = t.curve.value(time, 1) * setup.scale.x;
    let y = t.curve.value(time, 2) * setup.scale.y;
    if alpha == 1.0 && !add {
        bone.scale_x = x;
        bone.scale_y = y;
        return;
    }
    let (mut bx, mut by) = if matches!(from, MixFrom::Setup) {
        (setup.scale.x, setup.scale.y)
    } else {
        (bone.scale_x, bone.scale_y)
    };
    if add {
        bone.scale_x = bx + (x - setup.scale.x) * alpha;
        bone.scale_y = by + (y - setup.scale.y) * alpha;
    } else if out {
        bone.scale_x = bx + (x.abs() * signum(bx) - bx) * alpha;
        bone.scale_y = by + (y.abs() * signum(by) - by) * alpha;
    } else {
        bx = bx.abs() * signum(x);
        by = by.abs() * signum(y);
        bone.scale_x = bx + (x - bx) * alpha;
        bone.scale_y = by + (y - by) * alpha;
    }
}

/// Sign of `x`, matching Java's `Math.signum` (zero stays zero, unlike
/// `f32::signum`, which returns 1 or -1 for a signed zero).
fn signum(x: f32) -> f32 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

pub(super) fn apply_shear(
    t: &BoneTimeline,
    skel: &mut Skeleton,
    time: f32,
    alpha: f32,
    from: MixFrom,
    add: bool,
) {
    let Some((bone, setup)) = skel.bone_and_setup(t.bone) else {
        return;
    };
    let Some(first) = t.curve.first_time() else {
        return;
    };
    if time < first {
        match from {
            MixFrom::Setup => {
                bone.shear_x = setup.shear.x;
                bone.shear_y = setup.shear.y;
            }
            MixFrom::First => {
                bone.shear_x += (setup.shear.x - bone.shear_x) * alpha;
                bone.shear_y += (setup.shear.y - bone.shear_y) * alpha;
            }
            MixFrom::Current => {}
        }
        return;
    }
    let x = t.curve.value(time, 1);
    let y = t.curve.value(time, 2);
    if matches!(from, MixFrom::Setup) {
        bone.shear_x = setup.shear.x + x * alpha;
        bone.shear_y = setup.shear.y + y * alpha;
    } else if add {
        bone.shear_x += x * alpha;
        bone.shear_y += y * alpha;
    } else {
        bone.shear_x += (setup.shear.x + x - bone.shear_x) * alpha;
        bone.shear_y += (setup.shear.y + y - bone.shear_y) * alpha;
    }
}

/// Apply a single-axis bone timeline. Translation and shear axes add to the
/// setup. Scale axes multiply it with sign-aware mixing.
#[allow(clippy::too_many_arguments)]
pub(super) fn apply_bone_axis(
    t: &BoneTimeline,
    axis: BoneAxis,
    skel: &mut Skeleton,
    time: f32,
    alpha: f32,
    from: MixFrom,
    add: bool,
    out: bool,
) {
    let Some((bone, setup)) = skel.bone_and_setup(t.bone) else {
        return;
    };
    let Some(first) = t.curve.first_time() else {
        return;
    };
    match axis {
        BoneAxis::TranslateX => {
            bone.x = t
                .curve
                .relative_value(time, alpha, from, add, bone.x, setup.position.x);
        }
        BoneAxis::TranslateY => {
            bone.y = t
                .curve
                .relative_value(time, alpha, from, add, bone.y, setup.position.y);
        }
        BoneAxis::ShearX => {
            bone.shear_x =
                t.curve
                    .relative_value(time, alpha, from, add, bone.shear_x, setup.shear.x);
        }
        BoneAxis::ShearY => {
            bone.shear_y =
                t.curve
                    .relative_value(time, alpha, from, add, bone.shear_y, setup.shear.y);
        }
        BoneAxis::ScaleX => {
            if time < first {
                apply_scale_setup(&mut bone.scale_x, setup.scale.x, alpha, from);
            } else {
                bone.scale_x = scale_channel(
                    t.curve.value(time, 1),
                    setup.scale.x,
                    bone.scale_x,
                    alpha,
                    from,
                    add,
                    out,
                );
            }
        }
        BoneAxis::ScaleY => {
            if time < first {
                apply_scale_setup(&mut bone.scale_y, setup.scale.y, alpha, from);
            } else {
                bone.scale_y = scale_channel(
                    t.curve.value(time, 1),
                    setup.scale.y,
                    bone.scale_y,
                    alpha,
                    from,
                    add,
                    out,
                );
            }
        }
    }
}

/// Reset one scale channel toward its setup value (the before-first-frame case).
pub(super) fn apply_scale_setup(scale: &mut f32, setup: f32, alpha: f32, from: MixFrom) {
    match from {
        MixFrom::Setup => *scale = setup,
        MixFrom::First => *scale += (setup - *scale) * alpha,
        MixFrom::Current => {}
    }
}

/// Blend one scale channel: the keyed value scales the setup, with Spine's
/// sign-aware mixing. `out` is the mix-out direction.
fn scale_channel(
    value: f32,
    setup: f32,
    current: f32,
    alpha: f32,
    from: MixFrom,
    add: bool,
    out: bool,
) -> f32 {
    let target = value * setup;
    if alpha == 1.0 && !add {
        return target;
    }
    let base = if matches!(from, MixFrom::Setup) {
        setup
    } else {
        current
    };
    if add {
        base + (target - setup) * alpha
    } else if out {
        base + (target.abs() * signum(base) - base) * alpha
    } else {
        let signed = base.abs() * signum(target);
        signed + (target - signed) * alpha
    }
}

/// IK constraint timeline: mix and softness are interpolated. Bend direction,
/// compress, and stretch are stepped (read from the frame).
pub(super) fn apply_ik(
    t: &ConstraintTimeline,
    skel: &mut Skeleton,
    time: f32,
    alpha: f32,
    from: MixFrom,
    out: bool,
) {
    let Some((pose, setup)) = skel.ik_pose_and_setup(t.constraint) else {
        return;
    };
    let Some(first) = t.curve.first_time() else {
        return;
    };
    if time < first {
        match from {
            MixFrom::Setup => {
                pose.mix = setup.mix;
                pose.softness = setup.softness;
                pose.bend_direction = setup.bend_direction;
                pose.compress = setup.compress;
                pose.stretch = setup.stretch;
            }
            MixFrom::First => {
                pose.mix += (setup.mix - pose.mix) * alpha;
                pose.softness += (setup.softness - pose.softness) * alpha;
                pose.bend_direction = setup.bend_direction;
                pose.compress = setup.compress;
                pose.stretch = setup.stretch;
            }
            MixFrom::Current => {}
        }
        return;
    }
    let mix = t.curve.value(time, 1);
    let softness = t.curve.value(time, 2);
    let (base_mix, base_soft) = if matches!(from, MixFrom::Setup) {
        (setup.mix, setup.softness)
    } else {
        (pose.mix, pose.softness)
    };
    pose.mix = base_mix + (mix - base_mix) * alpha;
    pose.softness = base_soft + (softness - base_soft) * alpha;
    if out {
        if matches!(from, MixFrom::Setup) {
            pose.bend_direction = setup.bend_direction;
            pose.compress = setup.compress;
            pose.stretch = setup.stretch;
        }
    } else {
        pose.bend_direction = t.curve.frame_value(time, 3) as i32;
        pose.compress = t.curve.frame_value(time, 4) != 0.0;
        pose.stretch = t.curve.frame_value(time, 5) != 0.0;
    }
}

/// Transform constraint timeline: six interpolated mixes.
pub(super) fn apply_transform_mix(
    t: &ConstraintTimeline,
    skel: &mut Skeleton,
    time: f32,
    alpha: f32,
    from: MixFrom,
    add: bool,
) {
    let Some((pose, setup)) = skel.transform_pose_and_setup(t.constraint) else {
        return;
    };
    let Some(first) = t.curve.first_time() else {
        return;
    };
    if time < first {
        match from {
            MixFrom::Setup => {
                pose.mix_rotate = setup.mix_rotate;
                pose.mix_x = setup.mix_x;
                pose.mix_y = setup.mix_y;
                pose.mix_scale_x = setup.mix_scale_x;
                pose.mix_scale_y = setup.mix_scale_y;
                pose.mix_shear_y = setup.mix_shear_y;
            }
            MixFrom::First => {
                pose.mix_rotate += (setup.mix_rotate - pose.mix_rotate) * alpha;
                pose.mix_x += (setup.mix_x - pose.mix_x) * alpha;
                pose.mix_y += (setup.mix_y - pose.mix_y) * alpha;
                pose.mix_scale_x += (setup.mix_scale_x - pose.mix_scale_x) * alpha;
                pose.mix_scale_y += (setup.mix_scale_y - pose.mix_scale_y) * alpha;
                pose.mix_shear_y += (setup.mix_shear_y - pose.mix_shear_y) * alpha;
            }
            MixFrom::Current => {}
        }
        return;
    }
    pose.mix_rotate = absolute_value_with(
        t.curve.value(time, 1),
        alpha,
        from,
        add,
        pose.mix_rotate,
        setup.mix_rotate,
    );
    pose.mix_x = absolute_value_with(
        t.curve.value(time, 2),
        alpha,
        from,
        add,
        pose.mix_x,
        setup.mix_x,
    );
    pose.mix_y = absolute_value_with(
        t.curve.value(time, 3),
        alpha,
        from,
        add,
        pose.mix_y,
        setup.mix_y,
    );
    pose.mix_scale_x = absolute_value_with(
        t.curve.value(time, 4),
        alpha,
        from,
        add,
        pose.mix_scale_x,
        setup.mix_scale_x,
    );
    pose.mix_scale_y = absolute_value_with(
        t.curve.value(time, 5),
        alpha,
        from,
        add,
        pose.mix_scale_y,
        setup.mix_scale_y,
    );
    pose.mix_shear_y = absolute_value_with(
        t.curve.value(time, 6),
        alpha,
        from,
        add,
        pose.mix_shear_y,
        setup.mix_shear_y,
    );
}

/// Path constraint position timeline (single absolute value).
pub(super) fn apply_path_position(
    t: &ConstraintTimeline,
    skel: &mut Skeleton,
    time: f32,
    alpha: f32,
    from: MixFrom,
    add: bool,
) {
    if let Some((pose, setup)) = skel.path_pose_and_setup(t.constraint) {
        pose.position =
            t.curve
                .absolute_value(time, alpha, from, add, pose.position, setup.position);
    }
}

/// Path constraint spacing timeline (single absolute value, never additive).
pub(super) fn apply_path_spacing(
    t: &ConstraintTimeline,
    skel: &mut Skeleton,
    time: f32,
    alpha: f32,
    from: MixFrom,
) {
    if let Some((pose, setup)) = skel.path_pose_and_setup(t.constraint) {
        pose.spacing =
            t.curve
                .absolute_value(time, alpha, from, false, pose.spacing, setup.spacing);
    }
}

/// Path constraint mix timeline (rotate / x / y).
pub(super) fn apply_path_mix(
    t: &ConstraintTimeline,
    skel: &mut Skeleton,
    time: f32,
    alpha: f32,
    from: MixFrom,
    add: bool,
) {
    let Some((pose, setup)) = skel.path_pose_and_setup(t.constraint) else {
        return;
    };
    let Some(first) = t.curve.first_time() else {
        return;
    };
    if time < first {
        match from {
            MixFrom::Setup => {
                pose.mix_rotate = setup.mix_rotate;
                pose.mix_x = setup.mix_x;
                pose.mix_y = setup.mix_y;
            }
            MixFrom::First => {
                pose.mix_rotate += (setup.mix_rotate - pose.mix_rotate) * alpha;
                pose.mix_x += (setup.mix_x - pose.mix_x) * alpha;
                pose.mix_y += (setup.mix_y - pose.mix_y) * alpha;
            }
            MixFrom::Current => {}
        }
        return;
    }
    pose.mix_rotate = absolute_value_with(
        t.curve.value(time, 1),
        alpha,
        from,
        add,
        pose.mix_rotate,
        setup.mix_rotate,
    );
    pose.mix_x = absolute_value_with(
        t.curve.value(time, 2),
        alpha,
        from,
        add,
        pose.mix_x,
        setup.mix_x,
    );
    pose.mix_y = absolute_value_with(
        t.curve.value(time, 3),
        alpha,
        from,
        add,
        pose.mix_y,
        setup.mix_y,
    );
}

/// Physics constraint timeline: drives one tunable on one constraint, or on
/// every constraint whose matching global flag is set when the target is
/// [`GLOBAL_PHYSICS`].
pub(super) fn apply_physics(
    t: &ConstraintTimeline,
    property: PhysicsProperty,
    skel: &mut Skeleton,
    time: f32,
    alpha: f32,
    from: MixFrom,
    add: bool,
) {
    if t.constraint == GLOBAL_PHYSICS {
        let data = skel.data_arc();
        for (pose, setup) in skel
            .physics_constraints_mut()
            .iter_mut()
            .zip(&data.physics_constraints)
        {
            if property_global(setup, property) {
                apply_physics_one(pose, setup, property, &t.curve, time, alpha, from, add);
            }
        }
    } else if let Some((pose, setup)) = skel.physics_pose_and_setup(t.constraint) {
        apply_physics_one(pose, setup, property, &t.curve, time, alpha, from, add);
    }
}

/// Whether `property` is flagged global on `data` (so a global timeline drives
/// it).
fn property_global(data: &PhysicsConstraintData, property: PhysicsProperty) -> bool {
    match property {
        PhysicsProperty::Inertia => data.inertia_global,
        PhysicsProperty::Strength => data.strength_global,
        PhysicsProperty::Damping => data.damping_global,
        PhysicsProperty::Mass => data.mass_global,
        PhysicsProperty::Wind => data.wind_global,
        PhysicsProperty::Gravity => data.gravity_global,
        PhysicsProperty::Mix => data.mix_global,
    }
}

/// Apply one physics tunable to a single constraint pose. Mass is animated as a
/// mass value but stored inverted. Wind and gravity blend additively.
#[allow(clippy::too_many_arguments)]
pub(super) fn apply_physics_one(
    pose: &mut PhysicsConstraint,
    setup: &PhysicsConstraintData,
    property: PhysicsProperty,
    curve: &Curve,
    time: f32,
    alpha: f32,
    from: MixFrom,
    add: bool,
) {
    match property {
        PhysicsProperty::Inertia => {
            pose.inertia =
                curve.absolute_value(time, alpha, from, add, pose.inertia, setup.inertia);
        }
        PhysicsProperty::Strength => {
            pose.strength =
                curve.absolute_value(time, alpha, from, add, pose.strength, setup.strength);
        }
        PhysicsProperty::Damping => {
            pose.damping =
                curve.absolute_value(time, alpha, from, add, pose.damping, setup.damping);
        }
        PhysicsProperty::Mass => {
            let cur = 1.0 / pose.mass_inverse;
            let base = 1.0 / setup.mass_inverse;
            pose.mass_inverse = 1.0 / curve.absolute_value(time, alpha, from, add, cur, base);
        }
        PhysicsProperty::Wind => {
            pose.wind = curve.absolute_value(time, alpha, from, true, pose.wind, setup.wind);
        }
        PhysicsProperty::Gravity => {
            pose.gravity =
                curve.absolute_value(time, alpha, from, true, pose.gravity, setup.gravity);
        }
        PhysicsProperty::Mix => {
            pose.mix = curve.absolute_value(time, alpha, from, add, pose.mix, setup.mix);
        }
    }
}

/// Physics reset timeline: if a keyframe time falls in the window
/// `(last_time, time]` (handling a loop wrap where `time < last_time`), reset
/// the target constraint, or every physics constraint when global.
pub(super) fn apply_physics_reset(
    t: &PhysicsResetTimeline,
    skel: &mut Skeleton,
    last_time: f32,
    time: f32,
) {
    let fired = if time >= last_time {
        t.times.iter().any(|&kt| kt > last_time && kt <= time)
    } else {
        t.times.iter().any(|&kt| kt > last_time || kt <= time)
    };
    if !fired {
        return;
    }
    if t.constraint == GLOBAL_PHYSICS {
        skel.request_all_physics_reset();
    } else {
        skel.request_physics_reset(t.constraint);
    }
}

/// Slider scrub-time timeline: sets the slider's pose time (used when the slider
/// has no driving bone).
pub(super) fn apply_slider_time(
    t: &ConstraintTimeline,
    skel: &mut Skeleton,
    time: f32,
    alpha: f32,
    from: MixFrom,
    add: bool,
) {
    if let Some((pose, setup)) = skel.slider_pose_and_setup(t.constraint) {
        pose.time = t
            .curve
            .absolute_value(time, alpha, from, add, pose.time, setup.time);
    }
}

/// Slider mix timeline: blends the slider's pose mix from its setup value.
pub(super) fn apply_slider_mix(
    t: &ConstraintTimeline,
    skel: &mut Skeleton,
    time: f32,
    alpha: f32,
    from: MixFrom,
    add: bool,
) {
    if let Some((pose, setup)) = skel.slider_pose_and_setup(t.constraint) {
        pose.mix = t
            .curve
            .absolute_value(time, alpha, from, add, pose.mix, setup.mix);
    }
}

/// Event timeline: fire each event whose keyframe time falls in the window
/// `(last_time, time]` (handling a loop wrap), collecting it on the skeleton.
pub(super) fn apply_event(t: &EventTimeline, skel: &mut Skeleton, last_time: f32, time: f32) {
    let wrapped = time < last_time;
    for (&kt, event) in t.times.iter().zip(&t.events) {
        let fired = if wrapped {
            kt > last_time || kt <= time
        } else {
            kt > last_time && kt <= time
        };
        if fired {
            skel.push_event(event.clone());
        }
    }
}
