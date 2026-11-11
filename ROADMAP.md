# Roadmap

## Implemented and exercised

- Deterministic rigid stepping with static, kinematic, and dynamic bodies, rotational inertia, contacts, sleep and wakes, sensors, and filtering.
- Sphere, capsule, cylinder, cuboid, and plane contacts and queries. Convex hull and mixed cylinder contacts use experimental GJK/EPA. Compounds, static triangle meshes, and heightfields have targeted regression tests. Sweep-and-prune, BVH broad phase, and sphere CCD against static targets.
- Distance, rope, ball, damped spring, fixed, slider, and hinge constraints, with motors and limits for slider and hinge.
- Sphere character movement and raycast vehicle suspension, with analytical and regression checks.
- Renderer-independent debug geometry and a ten-scene interactive sandbox.
- XPBD cloth and tetrahedral volume, deterministic particles, and experimental SPH. Independent particle motion can run with Rayon.

## Next implementation priorities

1. Strengthen manifold reduction for mixed convex, compound, and mesh contacts.
2. Broaden contact tests around grazing capsule edges, clipped box manifolds, mixed mass stacks, and long chains of joints.
3. Expand mesh contacts to hulls and cylinders, and add swept capsules and box casts.
4. Extend moving-target sweeps beyond spheres and add rotational CCD. Measure the cost of island-parallel solving against serial in realistic scenes.
5. Add cloth self-collision, two-way particle/rigid coupling, and an incompressibility-oriented fluid solver.

## Further work

Dedicated 2D dynamics, persistent dynamic AABB trees, joint warm-starting, asynchronous world snapshots, buoyancy, and fracture.
