# Roadmap

## Implemented and exercised

- Deterministic rigid stepping with static, kinematic, and dynamic bodies, rotational inertia, contacts, sleep and wakes, sensors, and filtering.
- Sphere, capsule, cylinder, cuboid, and plane contacts and queries. Standalone convex hull queries and mixed cylinder contacts use experimental GJK/EPA. Sweep-and-prune, BVH broad phase, and sphere CCD against static targets.
- Distance, rope, ball, damped spring, fixed, slider, and hinge constraints, with motors and limits for slider and hinge.
- Renderer-independent debug geometry and a ten-scene interactive sandbox.
- XPBD cloth and tetrahedral volume, deterministic particles, and experimental SPH. Independent particle motion can run with Rayon.

## Next implementation priorities

1. Strengthen the experimental convex narrow phase, add hull mass properties, and extend the collider model to multiple shapes per body.
2. Broaden contact tests around grazing capsule edges, clipped box manifolds, mixed mass stacks, and long chains of joints.
3. Add triangle meshes and heightfields with local acceleration, then character controllers and raycast vehicles.
4. Add swept relative motion and rotational CCD. Measure the cost of island-parallel solving against serial in realistic scenes.
5. Add cloth self-collision, two-way particle/rigid coupling, and an incompressibility-oriented fluid solver.

## Further work

Dedicated 2D dynamics, persistent dynamic AABB trees, joint warm-starting, asynchronous world snapshots, compound mass properties, buoyancy, and fracture.
