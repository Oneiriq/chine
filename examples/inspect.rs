//! Load a Spine `.json` + `.atlas`, print skeleton stats, and pose/render one
//! frame. Validates the loader, atlas binding, FK + constraints, and the render
//! stream on real exports.
//!
//! Usage: `cargo run --example inspect -- skeleton.json atlas.atlas`

use std::error::Error;
use std::sync::Arc;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    let [_, json_path, atlas_path, ..] = args.as_slice() else {
        return Err("usage: inspect <skeleton.json> <atlas.atlas>".into());
    };

    let json = std::fs::read_to_string(json_path)
        .map_err(|e| format!("cannot read skeleton {json_path}: {e}"))?;
    let atlas_text = std::fs::read_to_string(atlas_path)
        .map_err(|e| format!("cannot read atlas {atlas_path}: {e}"))?;

    let atlas = chine::atlas::Atlas::parse(&atlas_text);
    let mut data = chine::load::from_json(&json)?;
    chine::render::bind_atlas(&mut data, &atlas);

    println!(
        "bones={} slots={} skins={} animations={} ik={} transform={} path={} atlas_pages={} atlas_regions={}",
        data.bones.len(),
        data.slots.len(),
        data.skins.len() + 1,
        data.animations.len(),
        data.ik_constraints.len(),
        data.transform_constraints.len(),
        data.path_constraints.len(),
        atlas.pages.len(),
        atlas.regions.len(),
    );
    for a in &data.animations {
        println!("  anim {:<22} {:.2}s", a.name(), a.duration());
    }

    let first_anim = data.animations.first().cloned();
    let mut skeleton = chine::skel::Skeleton::new(Arc::new(data));
    if let Some(anim) = first_anim {
        let name = anim.name().to_string();
        let mut state = chine::anim::AnimationState::new();
        state.set_animation(anim, true);
        // Pose a few frames across the animation and render each.
        for step in 0..5 {
            state.update(0.1);
            skeleton.set_bones_to_setup_pose();
            state.apply(&mut skeleton);
            skeleton.update_world_transform();
            let commands = chine::render::render(&skeleton);
            let verts: usize = commands.iter().map(|c| c.positions.len()).sum();
            println!(
                "  frame {step} of '{name}': {} draw commands, {verts} vertices",
                commands.len()
            );
        }
    }
    println!("OK");
    Ok(())
}
