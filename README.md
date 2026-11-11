# Rust Physics

Rust Physics is a double-precision 3D simulation library and sandbox. It implements rigid-body dynamics, collision detection and response, constraints, and spatial queries in safe Rust. The default build uses Glam for vector, matrix, and quaternion operations. Particle, cloth, and fluid modules can run independently of the rigid-body world.

## Build and run

Use a Rust 2024 toolchain. The checked-in lockfile and all optional targets were exercised with Rust 1.99.1 on Arch Linux.

```sh
cargo test --workspace --all-features
cargo run --example drop
cargo run --release --example sandbox --features macroquad
cargo doc --workspace --no-deps
```

The graphical example uses Macroquad and needs a working desktop and OpenGL context. In the sandbox, use **1–0** to select a scene, **Space** to pause, **N** to advance a paused frame, **R** to reset, **S/X** to spawn a sphere or box, and **G** to toggle rigid-body gravity. Right-drag orbits the view and the wheel zooms. **B/C/V** toggle bounds, contact normals, and velocities.

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

- Rigid bodies support static, kinematic, and dynamic motion, force and torque accumulation, world-space angular momentum, point impulses, gravity scaling, damping, and contact-island sleeping.
- Spheres, capsules, cylinders, oriented cuboids, and half-spaces use shape-specific narrow-phase collision and experimental GJK/EPA for mixed cylinder pairs. Contact response includes restitution, two-regime friction, cached impulses, and separate penetration correction. Layers, masks, and sensors control collision and overlap behavior.
- Joint types include ball, distance, rope, damped spring, fixed, hinge, and slider. Hinges and sliders support effort-limited motors and limits.
- Spatial queries include rays, batches, AABBs, sphere overlaps and casts, and nearest surfaces. Build a borrowed BVH with `world.spatial_index()` to accelerate repeated rays and AABB queries.
- `softbody` implements XPBD cloth and tetrahedral volume constraints. `particles` implements seeded emitters, lifetimes, fields, optional Rayon updates, and rigid boundaries. `fluid` implements experimental weakly compressible SPH with symmetric pair forces and a spatial hash.
- `debug` streams renderer-independent lines. The sandbox renders ten scenes including stacks, a pendulum, a jointed ragdoll, cloth, particles, and a fluid tank.

## Current limits

The rigid world uses one centered collider per body. Convex hulls support standalone GJK/EPA queries. Cylinder pairs that depend on EPA may miss poorly conditioned contacts. CCD sweeps dynamic spheres against static geometry. Kinematic and dynamic pairs otherwise use discrete contacts. Capsule pairs usually produce one contact, and the box solver uses translation-only penetration correction. Very stiff joints, extreme mass ratios, dense stacks, and extreme coordinate or timestep scales need further validation. Rigid solving is serial. Soft and fluid boundary coupling is one-way and discrete, and cloth has no self-collision. Meshes, terrain, and compound colliders remain on the roadmap.

See [architecture](ARCHITECTURE.md), [physics theory](PHYSICS_THEORY.md), [performance](PERFORMANCE.md), [roadmap](ROADMAP.md), and the [development checkpoint](DEVELOPMENT_STATE.md).
