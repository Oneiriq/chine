//! The GL-free half of [`crate::WebSpine`]: the skeleton, its animation state,
//! the active skin, each frame's render commands, and the setup-pose fit. It
//! needs no browser, so its logic has host tests.

use std::sync::Arc;

use chine::anim::AnimationState;
use chine::data::SkeletonData;
use chine::render::{render_into, RenderCommand};
use chine::skel::Skeleton;

use crate::setup_fit;

/// The name [`Player::skin_names`] gives the default skin. Setting it shows
/// the default skin only.
const DEFAULT_SKIN: &str = "default";

/// A posed, animated skeleton and the draw data of its current frame.
pub(crate) struct Player {
    data: Arc<SkeletonData>,
    skeleton: Skeleton,
    state: AnimationState,
    commands: Vec<RenderCommand>,
    /// Setup-pose fit: world-space center and half-extents, used to auto-fit
    /// the skeleton to the canvas.
    fit: (f32, f32, f32, f32),
}

impl Player {
    /// A player for `data`, showing the default skin with no animation.
    pub(crate) fn new(data: SkeletonData) -> Self {
        let data = Arc::new(data);
        let skeleton = Skeleton::new(Arc::clone(&data));
        let fit = measure_fit(&data, DEFAULT_SKIN);
        Self {
            data,
            skeleton,
            state: AnimationState::new(),
            commands: Vec::new(),
            fit,
        }
    }

    /// The skin names, starting with `"default"`.
    pub(crate) fn skin_names(&self) -> Vec<String> {
        std::iter::once(DEFAULT_SKIN.to_string())
            .chain(self.data.skins.iter().map(|skin| skin.name.clone()))
            .collect()
    }

    /// Show the named skin. `"default"` or an unknown name shows the default
    /// skin only. Returns `false` for an unknown name. The fit is measured
    /// again, since it depends on the attachments the skin shows.
    pub(crate) fn set_skin(&mut self, name: &str) -> bool {
        self.skeleton.set_skin(name);
        self.fit = measure_fit(&self.data, name);
        name == DEFAULT_SKIN || self.skeleton.active_skin().is_some()
    }

    /// Play the named animation, looping or not. Unknown names are ignored.
    pub(crate) fn set_animation(&mut self, name: &str, looping: bool) {
        if let Some(anim) = self.data.find_animation(name) {
            self.state.set_animation(Arc::clone(anim), looping);
        }
    }

    /// The animation names.
    pub(crate) fn animation_names(&self) -> Vec<String> {
        self.data
            .animations
            .iter()
            .map(|anim| anim.name().to_string())
            .collect()
    }

    /// The setup-pose fit of the active skin.
    pub(crate) fn fit(&self) -> (f32, f32, f32, f32) {
        self.fit
    }

    /// Advance the animation by `delta` seconds, pose the skeleton, and build
    /// the frame's render commands.
    pub(crate) fn advance(&mut self, delta: f32) -> &[RenderCommand] {
        self.state.update(delta);
        self.skeleton.update(delta);
        self.skeleton.set_bones_to_setup_pose();
        self.skeleton.set_slots_to_setup_pose();
        self.state.apply(&mut self.skeleton);
        self.skeleton.update_world_transform();
        render_into(&self.skeleton, &mut self.commands);
        &self.commands
    }
}

/// The setup-pose fit of `data` showing the skin `skin`. It poses a fresh
/// skeleton, so the playing one keeps its pose and physics state.
fn measure_fit(data: &Arc<SkeletonData>, skin: &str) -> (f32, f32, f32, f32) {
    let mut skeleton = Skeleton::new(Arc::clone(data));
    skeleton.set_skin(skin);
    skeleton.set_bones_to_setup_pose();
    skeleton.set_slots_to_setup_pose();
    skeleton.update_world_transform();
    let mut commands = Vec::new();
    render_into(&skeleton, &mut commands);
    setup_fit(&commands)
}

#[cfg(test)]
mod tests {
    use super::Player;
    use chine::render::RenderCommand;

    // A root bone with a body mesh, and a skin-required hat bone 100 units
    // up with a hat mesh. The skin "hatted" lists the hat bone. The skin
    // "big" swaps in a larger body, and a deform key under "big" moves it.
    const RIG: &str = r#"{
        "bones": [
            { "name": "root" },
            { "name": "hat", "parent": "root", "y": 100, "skin": true }
        ],
        "slots": [
            { "name": "body", "bone": "root", "attachment": "body" },
            { "name": "hat", "bone": "hat", "attachment": "hat" }
        ],
        "skins": [
            { "name": "default", "attachments": {
                "body": { "body": { "type": "mesh", "uvs": [0,0, 1,0, 0,1],
                    "triangles": [0,1,2], "vertices": [0,0, 10,0, 0,10] } },
                "hat": { "hat": { "type": "mesh", "uvs": [0,0, 1,0, 0,1],
                    "triangles": [0,1,2], "vertices": [0,0, 10,0, 0,10] } }
            } },
            { "name": "hatted", "bones": [ "hat" ] },
            { "name": "big", "attachments": {
                "body": { "body": { "type": "mesh", "uvs": [0,0, 1,0, 0,1],
                    "triangles": [0,1,2], "vertices": [0,0, 40,0, 0,40] } }
            } }
        ],
        "animations": { "stretch": { "attachments": { "big": { "body": { "body": {
            "deform": [ { "time": 0, "offset": 2, "vertices": [20] } ]
        } } } } } }
    }"#;

    fn player() -> Player {
        Player::new(chine::load::from_json(RIG).unwrap())
    }

    fn max_x(commands: &[RenderCommand]) -> f32 {
        commands
            .iter()
            .flat_map(|cmd| &cmd.positions)
            .fold(f32::MIN, |max, p| max.max(p.x))
    }

    #[test]
    fn skin_names_start_with_default() {
        assert_eq!(player().skin_names(), ["default", "hatted", "big"]);
    }

    #[test]
    fn a_skin_required_bone_draws_only_with_its_skin() {
        let mut p = player();
        assert_eq!(p.advance(0.0).len(), 1, "the hat bone is inactive");

        assert!(p.set_skin("hatted"));
        let commands = p.advance(0.0);
        assert_eq!(commands.len(), 2, "the hat bone is active");
        assert!(
            commands[1].positions.iter().all(|pos| pos.y >= 100.0),
            "the hat sits on its bone"
        );

        assert!(!p.set_skin("missing"), "an unknown skin is reported");
        assert_eq!(p.advance(0.0).len(), 1, "an unknown skin shows the default");
    }

    #[test]
    fn a_skin_swaps_its_attachments_and_applies_its_deform_keys() {
        let mut p = player();
        p.set_animation("stretch", true);
        assert_eq!(max_x(p.advance(0.0)), 10.0, "the default body");

        // The deform key moves the big body's second vertex to x = 60.
        assert!(p.set_skin("big"));
        assert_eq!(max_x(p.advance(0.0)), 60.0, "the big body, deformed");

        assert!(p.set_skin("default"));
        assert_eq!(max_x(p.advance(0.0)), 10.0, "the default body again");
    }

    #[test]
    fn a_skin_change_measures_the_fit_again() {
        let mut p = player();
        let (_, _, _, half_height) = p.fit();
        assert!((half_height - 5.0).abs() < 1e-4, "the body alone");

        p.set_skin("hatted");
        let (_, center_y, _, half_height) = p.fit();
        assert!((half_height - 55.0).abs() < 1e-4, "the body and the hat");
        assert!((center_y - 55.0).abs() < 1e-4);

        p.set_skin("default");
        let (_, _, _, half_height) = p.fit();
        assert!((half_height - 5.0).abs() < 1e-4, "the body alone again");
    }
}
