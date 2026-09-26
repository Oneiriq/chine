use std::path::Path;

/// The official Spine 4.3 example rigs, each exported as `.json` and `.skel`.
const EXAMPLES: [&str; 8] = [
    "coin-pro",
    "diamond-pro",
    "mix-and-match-pro",
    "raptor-pro",
    "spineboy-pro",
    "stretchyman-pro",
    "tank-pro",
    "vine-pro",
];

// The Spine rigs under `data/` are full editor exports kept out of version
// control. When all of them are present, enable the `skel_fixtures` cfg so the
// tests that read them run; otherwise those tests are marked `ignore`d (rather
// than silently passing as no-ops), and the synthetic in-memory tests carry the
// loader coverage in a clean checkout.
//
// The official example exports (`data/examples/<name>.json` and `.skel`) turn
// on `spine_examples` the same way. The tests behind it load each rig from both
// formats and compare the results.
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

    let examples: Vec<String> = EXAMPLES
        .iter()
        .flat_map(|name| {
            [
                format!("data/examples/{name}.json"),
                format!("data/examples/{name}.skel"),
            ]
        })
        .collect();
    for example in &examples {
        println!("cargo:rerun-if-changed={example}");
    }
    if examples.iter().all(|example| Path::new(example).exists()) {
        println!("cargo:rustc-cfg=spine_examples");
    }
}
