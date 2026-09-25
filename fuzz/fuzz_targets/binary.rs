//! Binary `.skel` loading, then one setup-pose frame of whatever loads.

#![no_main]

use std::sync::Arc;

use chine::render::render;
use chine::skel::Skeleton;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|bytes: &[u8]| {
    if let Ok(data) = chine::binary::from_binary(bytes) {
        let mut skeleton = Skeleton::new(Arc::new(data));
        skeleton.update_world_transform();
        let _ = render(&skeleton);
    }
});
