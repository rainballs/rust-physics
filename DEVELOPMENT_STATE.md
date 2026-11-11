# Development checkpoint

The library, examples, and benchmark targets are in the Cargo workspace root. The architecture and current API are described in the README and ARCHITECTURE. The initial rigid-body commit is `20f5e74`. Particle-based simulation was added in `03a2e17`, and the first sandbox and BVH milestone is `72c5813`.

## Verification

Run these from the workspace root.

```sh
cargo fmt --all -- --check
cargo check --workspace --all-targets --all-features
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
cargo doc --workspace --all-features --no-deps
RUST_PHYSICS_SMOKE=1 cargo run --example sandbox --features macroquad
RAYON_NUM_THREADS=4 cargo bench --features rayon --bench scaling --bench throughput
```

The smoke command opens an OpenGL window, advances each sandbox scene, draws a frame, and exits. Running it needs a desktop session. Benchmarks report latency percentiles to stdout.

## Active priorities

Expand mesh and mixed convex coverage, profile contact hot paths, and improve fluid boundaries. The rigid body model supports weighted primitive and hull compounds. Rigid stepping and contact generation are serial. CCD is translational for dynamic spheres. The default static-target path allows four impacts per sphere per substep, and synchronized moving-sphere mode allows 32 world impacts per substep. The rigid solver clears its external force accumulator after the public step. The particle and soft-body solvers support discrete one-way boundary projection.

See ROADMAP for unfinished work and PERFORMANCE for measured workloads and limitations.
