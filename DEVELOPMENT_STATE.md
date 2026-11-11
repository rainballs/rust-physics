# Development checkpoint

The workspace contains one Rust library with independent modules for rigid bodies, collision geometry, constraints, terrain, spatial queries, particles, deformables, fluids, and gameplay utilities. See the README for the public entry points, ARCHITECTURE for data flow, and PHYSICS_THEORY for solver assumptions.

The initial rigid-body milestone is `20f5e74`, particle-based simulation is `03a2e17`, the first sandbox and BVH milestone is `72c5813`, and compound, terrain, moving CCD, and gameplay support began in `7be6819`. Later commits record validation and optimization.

## Reproduce validation

```sh
cargo fmt --all -- --check
cargo check --workspace --all-targets --all-features
cargo clippy --workspace --all-targets --all-features -- -D warnings
RAYON_NUM_THREADS=4 cargo test --workspace --all-features
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --all-features --no-deps
cargo run --example drop
cargo run --example deform
RUST_PHYSICS_SMOKE=1 RUST_PHYSICS_SMOKE_STEPS=90 cargo run --release --example sandbox --features macroquad
cargo run --release --example stress -- 192 3600
RAYON_NUM_THREADS=4 cargo bench --features rayon --bench scaling --bench throughput --bench fluids
```

The graphical smoke test advances all ten scenes and both fluid solvers in a desktop OpenGL context. The 192-seed rigid soak ran 691,200 fixed steps. The original stress replay at seed 30 exposed a synchronized CCD time-advance defect. The regression test checks that unrelated free-falling bodies still advance through event-budget exhaustion.

## Current limits and next work

- Rotational CCD and moving-target CCD for shapes other than spheres remain open. Budget fallbacks use discrete contacts for the remaining interval and report a statistic.
- Rigid constraint solving is serial. Rayon is optional for independent particle updates and narrow-phase contact generation.
- Mixed convex and mesh contact manifolds use bounded, experimental GJK/EPA and can miss difficult degeneracies. Improve mixed-shape stacking tests and terrain seam handling.
- PBF and cloth use one-way discrete boundary projection. SPH has optional two-way normal impulses. Cloth self-contact is node-only.
- `PlanarWorld` exposes circle, rectangle, and half-plane dynamics. Additional planar shape types and query helpers remain open.
- Compare workloads in PERFORMANCE before changing the default broad phase.
