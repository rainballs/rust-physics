# Development checkpoint

The workspace contains one Rust library with independent modules for rigid bodies, collision geometry, constraints, terrain, spatial queries, particles, deformables, fluids, and gameplay utilities. See the README for the public entry points, ARCHITECTURE for data flow, and PHYSICS_THEORY for solver assumptions.

The initial rigid-body milestone is `20f5e74`, particle-based simulation is `03a2e17`, the first sandbox and BVH milestone is `72c5813`, and compound, terrain, moving CCD, and gameplay support began in `7be6819`. Planar dynamics and broad-phase optimization are recorded in `473a24e`. Later commits record validation and performance work.

## Reproduce validation

```sh
cargo fmt --all -- --check
cargo check --workspace --all-targets --all-features
cargo clippy --workspace --all-targets --all-features -- -D warnings
RAYON_NUM_THREADS=4 cargo test --workspace --all-features
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --all-features --no-deps
cargo run --example drop
cargo run --example deform
cargo run --release --example profile -- 100000
RUST_PHYSICS_SMOKE=1 RUST_PHYSICS_SMOKE_STEPS=90 cargo run --release --example sandbox --features macroquad
cargo run --release --example stress -- 192 3600
RAYON_NUM_THREADS=4 cargo bench --features rayon --bench scaling --bench throughput --bench fluids --bench allocations
```

The graphical smoke test advances all ten scenes and both fluid solvers in a desktop OpenGL context. The 192-seed rigid soak ran 691,200 fixed steps. The original stress replay at seed 30 exposed a synchronized CCD time-advance defect. The regression test checks that unrelated free-falling bodies still advance through event-budget exhaustion. Seed 54 later exposed a degenerate EPA horizon. The face budget and closed-horizon checks bounded that path, and the targeted replay completed 3,600 steps.

## Current limits and next work

- Rotating convex CCD against static targets is experimental. Synchronized moving-target CCD beyond spheres remains open. Budget fallbacks use discrete contacts for the remaining interval and report a statistic.
- Serial rigid solving is the default. Rayon is optional for particle updates, narrow-phase contact generation, and disjoint coupled islands.
- Mixed convex and mesh contact manifolds use bounded, experimental GJK/EPA and can miss difficult degeneracies. Improve mixed-shape stacking tests and terrain seam handling.
- PBF uses one-way discrete boundary projection. SPH, particles, and deformable nodes can exchange normal impulses with rigid bodies. Cloth surface self-contact is discrete.
- `PlanarWorld` exposes circle, rectangle, and half-plane dynamics. Additional planar shape types and query helpers remain open.
- Compare workloads in PERFORMANCE before changing the default broad phase.
