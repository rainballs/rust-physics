# Rust Physics

Rust Physics is a double-precision 3D simulation library and sandbox. It implements rigid-body dynamics, collision detection and response, constraints, and spatial queries in safe Rust. The default build uses Glam for vector, matrix, and quaternion operations. Particle, cloth, and fluid modules can run independently of the rigid-body world.

## Build and run

Use a Rust 2024 toolchain. The checked-in lockfile and all optional targets were exercised with Rust 1.99.1 on Arch Linux.

```sh
cargo test --workspace --all-features
cargo run --example drop
cargo run --example deform
cargo run --release --example stress -- 8 600
cargo run --release --example sandbox --features macroquad
cargo doc --workspace --no-deps
```

The graphical example uses Macroquad and needs a working desktop and OpenGL context. In the sandbox, use **1–0** to select a scene, **Space** to pause, **N** to advance a paused frame, **R** to reset, **S/X** to spawn a sphere or box, and **G** to toggle rigid-body gravity. Right-drag orbits the view and the wheel zooms. **B/C/V/F/Q** toggle bounds, contact normals, velocities, forces, and centers of mass. **M** changes broad phase, and **P** switches the fluid tank between SPH and PBF.

## Rigid-body world

```rust
use rust_physics::{PhysicsWorld, RigidBodyBuilder, Shape, Vec3};

let mut world = PhysicsWorld::new();
world.add_body(RigidBodyBuilder::static_body().shape(Shape::Plane).build());
let ball = world.add_body(
    RigidBodyBuilder::dynamic()
        .position(Vec3::Y * 5.0)
        .shape(Shape::sphere(0.5))
        .mass(2.0)
        .build(),
);

for _ in 0..180 {
    world.step_fixed()?;
}
assert!(world.body(ball).unwrap().position().y >= 0.498);
# Ok::<(), rust_physics::StepError>(())
```

- Rigid bodies support static, kinematic, and dynamic motion, force and torque accumulation, world-space angular momentum, point impulses, gravity scaling, damping, axis locks, origin rebasing, and contact-island sleeping. `PlanarWorld` provides a Vec2 API with circles, rectangles, half-planes, and solid-disk inertia. `planar_xy()` exposes the underlying axis locks.
- Primitive colliders include spheres, capsules, cylinders, cuboids, and half-spaces. The engine also supports weighted compounds, convex hulls from closed triangle surfaces, and static triangle meshes and heightfields. Mixed convex contacts can use experimental GJK/EPA. Authored compounds can split into child bodies with inherited momentum. Contact response includes restitution, two-regime friction, cached impulses, and separate penetration correction. Layers, masks, and sensors control collision and overlap behavior.
- Joint types include ball, distance, rope, damped spring, fixed, hinge, and slider. Hinges and sliders support effort-limited motors and limits. Joint and contact impulses warm-start across comparable substeps.
- Spatial queries include rays, batches, AABBs, sphere overlaps, nearest surfaces, and translational convex casts. Build a borrowed BVH with `world.spatial_index()` for repeated queries. Broad-phase choices include sweep-and-prune, a rebuilt BVH, a uniform grid, and a persistent fat-AABB tree.
- `softbody` implements XPBD cloth, tetrahedral volume constraints, and optional non-adjacent node collision. `particles` implements seeded emitters, lifetimes, fields, and equal-mass pair and rigid-boundary collisions. `fluid` implements experimental SPH and PBF, with optional two-way SPH boundary impulses. Rayon can parallelize independent particle motion and rigid narrow-phase work.
- `character` implements sweep-and-slide sphere motion. `vehicle` applies raycast suspension and load-limited tire forces. `buoyancy` computes displaced volume and drag for partially submerged spheres.
- `debug` streams renderer-independent lines. The sandbox renders ten scenes including stacks, a pendulum, a jointed ragdoll, cloth, particles, and a fluid tank.

## Current limits

Convex pairs that depend on EPA may miss poorly conditioned contacts. Static mesh contacts for cylinders and hulls use experimental GJK/EPA per triangle. CCD sweeps fast dynamic spheres against static geometry, with optional synchronized sweeps against moving spheres. It reports budget exhaustion and completes the remaining time discretely. Other moving shape pairs use discrete contacts. Capsule pairs usually produce one contact, and contact position correction translates bodies without rotating them. Very stiff joints, extreme mass ratios, dense stacks, and extreme coordinate or timestep scales need further validation. Rigid constraint solving is serial. Cloth self-collision approximates vertices as spheres. Fluid and cloth boundary contacts are discrete, and PBF and cloth boundaries are one-way.

See [architecture](ARCHITECTURE.md), [physics theory](PHYSICS_THEORY.md), [performance](PERFORMANCE.md), [roadmap](ROADMAP.md), and the [development checkpoint](DEVELOPMENT_STATE.md).
