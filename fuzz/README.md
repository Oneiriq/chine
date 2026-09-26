# Fuzzing

cargo-fuzz targets for every part of chine that reads untrusted input. They look for
panics, hangs, and runaway memory use on malformed skeletons and atlases.

| Target | What it feeds | Entry points |
|---|---|---|
| `binary` | `.skel` bytes | `from_binary`, then one setup-pose frame and `render` |
| `json` | JSON text | `from_json`, then one setup-pose frame and `render` |
| `atlas` | atlas text | `Atlas::parse`, `find_region`, and `bind_atlas` on a fixed skeleton with region, mesh, and sequence attachments |
| `pose` | control bytes, a skeleton, an atlas | loading, `bind_atlas`, skins, several tracks with queued and crossfaded animations, physics time, world transforms, and `render_with` over several frames |

The `pose` input starts with a 35-byte header: a flags byte (bit 0 reads the payload as
JSON instead of binary, bit 1 picks a skin, bit 2 switches skins mid-play, bit 7 makes
times and scales raw `f32` bits), the payload length as a little-endian `u16`, and 32
control bytes. The skeleton payload follows, and the atlas text is whatever comes after
it. See `fuzz_targets/pose.rs`.

## Running

cargo-fuzz needs nightly Rust and a Linux or macOS host.

```sh
cargo install cargo-fuzz
cd fuzz
cargo +nightly fuzz run -O -a binary -- -max_len=65536 -rss_limit_mb=2048 -timeout=10
```

Keep `-a`. It turns on overflow checks and debug assertions in the optimized build, and
integer overflow in index math is one of the main things these targets look for.

Good starting seeds are the inputs the unit tests already build. Real exports work too:
a `.skel` for `binary`, a `.json` for `json`, and an `.atlas` for `atlas`.

On Windows, run it in Docker:

```sh
docker run --rm -it -v "%cd%:/src" rust:1-trixie bash
rustup toolchain install nightly --profile minimal
cargo +nightly install cargo-fuzz
cd /src/fuzz && cargo +nightly fuzz run -O -a binary
```

## When a target finds a crash

libFuzzer writes the input to `fuzz/artifacts/<target>/`. Shrink it with
`cargo +nightly fuzz tmin -O -a <target> <file>`, fix the bug, and add a regression test
to the module that owns the code, built from the minimized input.
