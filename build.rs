use std::path::Path;

// The Spine rigs under `data/` are full editor exports kept out of version
// control. When all of them are present, enable the `skel_fixtures` cfg so the
// tests that read them run; otherwise those tests are marked `ignore`d (rather
// than silently passing as no-ops), and the synthetic in-memory tests carry the
// loader coverage in a clean checkout.
fn main() {
    let fixtures = [
        "data/diamond-pro.skel",
        "data/diamond-pro.atlas",
        "data/Spine.skel",
    ];
    for fixture in fixtures {
        println!("cargo:rerun-if-changed={fixture}");
    }
    if fixtures.iter().all(|fixture| Path::new(fixture).exists()) {
        println!("cargo:rustc-cfg=skel_fixtures");
    }
}
