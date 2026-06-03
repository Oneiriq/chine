//! Animation events: named markers fired during playback (footsteps, hit
//! frames, audio cues). The animation system collects fired [`Event`]s each
//! frame for the host to read via [`crate::skel::Skeleton::events`].

/// Setup data for a named event, declared on the skeleton: its default values.
#[derive(Debug, Clone)]
pub struct EventData {
    /// Event name, unique within the skeleton.
    pub name: String,
    /// Default integer value.
    pub int_value: i32,
    /// Default float value.
    pub float_value: f32,
    /// Default string value.
    pub string_value: String,
    /// Audio file path, if this event plays a sound.
    pub audio_path: Option<String>,
    /// Default playback volume.
    pub volume: f32,
    /// Default stereo balance.
    pub balance: f32,
}

/// A fired event instance: its name, fire time, and (possibly keyframe-
/// overridden) values. Collected each frame for the host to read.
#[derive(Debug, Clone)]
pub struct Event {
    /// The event's name.
    pub name: String,
    /// The animation time at which it fired.
    pub time: f32,
    /// Integer value.
    pub int_value: i32,
    /// Float value.
    pub float_value: f32,
    /// String value.
    pub string_value: String,
    /// Playback volume.
    pub volume: f32,
    /// Stereo balance.
    pub balance: f32,
}
