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
    if time < t.curve.first_time() {
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

#[allow(clippy::similar_names)]
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
    if time < t.curve.first_time() {
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
/// `f32::signum`, which returns `±1` for `±0`).
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
    if time < t.curve.first_time() {
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
/// setup; scale axes multiply it with sign-aware mixing.
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
            if time < t.curve.first_time() {
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
            if time < t.curve.first_time() {
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

/// IK constraint timeline: mix and softness are interpolated; bend direction,
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
    if time < t.curve.first_time() {
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
    if time < t.curve.first_time() {
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
    if time < t.curve.first_time() {
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
/// mass value but stored inverted; wind and gravity blend additively.
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

/// Slot color timeline: blends the slot's RGB (and alpha when `has_alpha`) tint
/// from its setup color toward the keyed colors. The index is the slot.
pub(super) fn apply_slot_color(
    t: &ConstraintTimeline,
    has_alpha: bool,
    skel: &mut Skeleton,
    time: f32,
    alpha: f32,
    from: MixFrom,
    add: bool,
) {
    let Some((slot, setup)) = skel.slot_pose_and_setup(t.constraint) else {
        return;
    };
    if time < t.curve.first_time() {
        match from {
            MixFrom::Setup => {
                slot.color.r = setup.color.r;
                slot.color.g = setup.color.g;
                slot.color.b = setup.color.b;
                if has_alpha {
                    slot.color.a = setup.color.a;
                }
            }
            MixFrom::First => {
                slot.color.r += (setup.color.r - slot.color.r) * alpha;
                slot.color.g += (setup.color.g - slot.color.g) * alpha;
                slot.color.b += (setup.color.b - slot.color.b) * alpha;
                if has_alpha {
                    slot.color.a += (setup.color.a - slot.color.a) * alpha;
                }
            }
            MixFrom::Current => {}
        }
        return;
    }
    slot.color.r = absolute_value_with(
        t.curve.value(time, 1),
        alpha,
        from,
        add,
        slot.color.r,
        setup.color.r,
    );
    slot.color.g = absolute_value_with(
        t.curve.value(time, 2),
        alpha,
        from,
        add,
        slot.color.g,
        setup.color.g,
    );
    slot.color.b = absolute_value_with(
        t.curve.value(time, 3),
        alpha,
        from,
        add,
        slot.color.b,
        setup.color.b,
    );
    if has_alpha {
        slot.color.a = absolute_value_with(
            t.curve.value(time, 4),
            alpha,
            from,
            add,
            slot.color.a,
            setup.color.a,
        );
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

/// Slot alpha timeline: blends only the slot tint's alpha channel from its setup
/// value toward the keyed alpha.
pub(super) fn apply_slot_alpha(
    t: &ConstraintTimeline,
    skel: &mut Skeleton,
    time: f32,
    alpha: f32,
    from: MixFrom,
    add: bool,
) {
    let Some((slot, setup)) = skel.slot_pose_and_setup(t.constraint) else {
        return;
    };
    if time < t.curve.first_time() {
        match from {
            MixFrom::Setup => slot.color.a = setup.color.a,
            MixFrom::First => slot.color.a += (setup.color.a - slot.color.a) * alpha,
            MixFrom::Current => {}
        }
        return;
    }
    slot.color.a = absolute_value_with(
        t.curve.value(time, 1),
        alpha,
        from,
        add,
        slot.color.a,
        setup.color.a,
    );
}

/// Slot two-color timeline: blends the slot's light (RGB or RGBA) and dark (RGB)
/// tints from their setup colors toward the keyed colors.
pub(super) fn apply_slot_two_color(
    t: &ConstraintTimeline,
    light_alpha: bool,
    skel: &mut Skeleton,
    time: f32,
    alpha: f32,
    from: MixFrom,
    add: bool,
) {
    let Some((slot, setup)) = skel.slot_pose_and_setup(t.constraint) else {
        return;
    };
    let setup_dark = setup.dark_color.unwrap_or(Color::new(0.0, 0.0, 0.0, 1.0));
    if time < t.curve.first_time() {
        match from {
            MixFrom::Setup => {
                slot.color = setup.color;
                slot.dark_color = setup.dark_color;
            }
            MixFrom::First => {
                slot.color.r += (setup.color.r - slot.color.r) * alpha;
                slot.color.g += (setup.color.g - slot.color.g) * alpha;
                slot.color.b += (setup.color.b - slot.color.b) * alpha;
                if light_alpha {
                    slot.color.a += (setup.color.a - slot.color.a) * alpha;
                }
                let mut dark = slot.dark_color.unwrap_or(setup_dark);
                dark.r += (setup_dark.r - dark.r) * alpha;
                dark.g += (setup_dark.g - dark.g) * alpha;
                dark.b += (setup_dark.b - dark.b) * alpha;
                slot.dark_color = Some(dark);
            }
            MixFrom::Current => {}
        }
        return;
    }
    slot.color.r = absolute_value_with(
        t.curve.value(time, 1),
        alpha,
        from,
        add,
        slot.color.r,
        setup.color.r,
    );
    slot.color.g = absolute_value_with(
        t.curve.value(time, 2),
        alpha,
        from,
        add,
        slot.color.g,
        setup.color.g,
    );
    slot.color.b = absolute_value_with(
        t.curve.value(time, 3),
        alpha,
        from,
        add,
        slot.color.b,
        setup.color.b,
    );
    if light_alpha {
        slot.color.a = absolute_value_with(
            t.curve.value(time, 4),
            alpha,
            from,
            add,
            slot.color.a,
            setup.color.a,
        );
    }
    let base = if light_alpha { 4 } else { 3 };
    let mut dark = slot.dark_color.unwrap_or(setup_dark);
    dark.r = absolute_value_with(
        t.curve.value(time, base + 1),
        alpha,
        from,
        add,
        dark.r,
        setup_dark.r,
    );
    dark.g = absolute_value_with(
        t.curve.value(time, base + 2),
        alpha,
        from,
        add,
        dark.g,
        setup_dark.g,
    );
    dark.b = absolute_value_with(
        t.curve.value(time, base + 3),
        alpha,
        from,
        add,
        dark.b,
        setup_dark.b,
    );
    slot.dark_color = Some(dark);
}

/// Slot attachment timeline: a stepped switch to the keyed attachment name.
pub(super) fn apply_attachment(
    t: &AttachmentTimeline,
    skel: &mut Skeleton,
    time: f32,
    from: MixFrom,
) {
    let Some((slot, setup)) = skel.slot_pose_and_setup(t.slot) else {
        return;
    };
    if t.times.is_empty() || time < t.times[0] {
        if matches!(from, MixFrom::Setup | MixFrom::First) {
            slot.attachment = setup.attachment.clone();
        }
        return;
    }
    let idx = search_step(&t.times, time);
    slot.attachment = t.names[idx].clone();
}

/// Draw-order timeline: a stepped switch to the keyed slot ordering.
pub(super) fn apply_draw_order(t: &DrawOrderTimeline, skel: &mut Skeleton, time: f32) {
    if t.times.is_empty() || time < t.times[0] {
        return; // before the first key: keep the setup order
    }
    let idx = search_step(&t.times, time);
    skel.set_draw_order(&t.orders[idx]);
}

/// Event timeline: fire each event whose keyframe time falls in the window
/// `(last_time, time]` (handling a loop wrap), collecting it on the skeleton.
pub(super) fn apply_event(t: &EventTimeline, skel: &mut Skeleton, last_time: f32, time: f32) {
    let wrapped = time < last_time;
    for (i, &kt) in t.times.iter().enumerate() {
        let fired = if wrapped {
            kt > last_time || kt <= time
        } else {
            kt > last_time && kt <= time
        };
        if fired {
            skel.push_event(t.events[i].clone());
        }
    }
}

/// Mesh deform timeline: set the slot's deform buffer to the setup vertices plus
/// the interpolated keyframe offsets (scaled by `alpha`). Only applies while the
/// slot shows the timeline's attachment.
pub(super) fn apply_deform(
    t: &DeformTimeline,
    skel: &mut Skeleton,
    time: f32,
    alpha: f32,
    from: MixFrom,
) {
    let Some(slot) = skel.slot(t.slot) else {
        return;
    };
    if slot.attachment.as_deref() != Some(t.attachment.as_str()) {
        return;
    }
    // Skin-aware: apply only if the slot's current attachment draws its deform
    // from this timeline's authoring skin (so per-skin deforms and
    // non-inheriting linked meshes do not cross over).
    let same_skin = match skel
        .data()
        .attachment(t.slot, &t.attachment, skel.active_skin())
    {
        Some(Attachment::Mesh(m)) => m.deform_skin.as_deref() == t.skin.as_deref(),
        _ => t.skin.is_none(),
    };
    if !same_skin {
        return;
    }
    let Some((slot, _)) = skel.slot_pose_and_setup(t.slot) else {
        return;
    };
    let n = t.setup.len();
    if t.times.is_empty() || time < t.times[0] {
        if matches!(from, MixFrom::Setup) {
            slot.deform.clear();
        }
        return;
    }
    let offset = interp_deform(t, time);
    slot.deform.resize(n, 0.0);
    for (i, d) in slot.deform.iter_mut().enumerate() {
        *d = t.setup[i] + offset.get(i).copied().unwrap_or(0.0) * alpha;
    }
}

/// Sequence timeline: advance the slot's sequence frame index from the active
/// keyframe by the elapsed time over the delay, wrapped per the sequence mode.
/// Applies only while the slot shows the timeline's attachment.
pub(super) fn apply_sequence(t: &SequenceTimeline, skel: &mut Skeleton, time: f32) {
    if t.count == 0 || t.times.is_empty() || time < t.times[0] {
        return;
    }
    let Some((slot, _)) = skel.slot_pose_and_setup(t.slot) else {
        return;
    };
    if slot.attachment.as_deref() != Some(t.attachment.as_str()) {
        return;
    }
    let frame = search_step(&t.times, time);
    let mode_and_index = t.mode_and_index[frame];
    let count = t.count;
    let mut index = (mode_and_index >> 4) as usize;
    let mode = mode_and_index & 0xf;
    if mode != 0 {
        let delay = t.delays[frame];
        index += ((time - t.times[frame]) / delay + 0.000_01) as usize;
        index = match mode {
            1 => index.min(count - 1),                        // once
            2 => index % count,                               // loop
            3 => sequence_pingpong(index, count),             // pingpong
            4 => (count - 1).saturating_sub(index),           // once reverse
            5 => count - 1 - (index % count),                 // loop reverse
            6 => sequence_pingpong(index + count - 1, count), // pingpong reverse
            _ => index,
        };
    }
    slot.sequence_index = index as i32;
}

/// Wrap an index across a ping-pong sequence of `count` regions.
fn sequence_pingpong(index: usize, count: usize) -> usize {
    let n = (count * 2).saturating_sub(2);
    let i = if n == 0 { 0 } else { index % n };
    if i >= count {
        n - i
    } else {
        i
    }
}

/// Interpolate the deform offset frames at `time`, using the timeline's curve
/// (stepped / linear / Bezier) for the interpolation percent.
fn interp_deform(t: &DeformTimeline, time: f32) -> Vec<f32> {
    let times = &t.times;
    let last = times.len() - 1;
    if time >= times[last] {
        return t.frames[last].clone();
    }
    let i = search_step(times, time);
    let pct = t.percent(time);
    let a = &t.frames[i];
    let b = &t.frames[i + 1];
    let n = a.len().max(b.len());
    let mut out = vec![0.0; n];
    for (j, v) in out.iter_mut().enumerate() {
        let av = a.get(j).copied().unwrap_or(0.0);
        let bv = b.get(j).copied().unwrap_or(0.0);
        *v = av + (bv - av) * pct;
    }
    out
}

/// Index of the last keyframe at or before `time` (assumes `time >= times[0]`).
fn search_step(times: &[f32], time: f32) -> usize {
    times.iter().rposition(|&t| t <= time).unwrap_or(0)
}
