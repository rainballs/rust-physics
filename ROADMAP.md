# Roadmap

## Implemented and exercised

- Deterministic rigid stepping with static, kinematic, and dynamic bodies, rotational inertia, contacts, sleep and wakes, sensors, and filtering.
- Sphere, capsule, cylinder, cuboid, and plane contacts and queries. Convex hull and mixed cylinder contacts use experimental GJK/EPA. Compounds, static triangle meshes, and heightfields have targeted regression tests. Sweep-and-prune, rebuilt BVH, uniform grid, persistent tree, and sphere CCD against static and optional moving sphere targets. Rotating convex CCD against static geometry is experimental.
- Distance, rope, ball, damped spring, fixed, slider, and hinge constraints, with motors and limits for slider and hinge.
- A Vec2 planar world with circle, rectangle, and half-plane contacts.
- Sphere character movement and raycast vehicle suspension, with analytical and regression checks.
- Renderer-independent debug geometry and a ten-scene interactive sandbox.
- XPBD cloth and tetrahedral volume, deterministic particles, and experimental SPH. Independent particle motion, narrow-phase contacts, and coupled island solving can run with Rayon. PBF density projection and two-way SPH boundary impulses are experimental.

## Next implementation priorities

1. Strengthen manifold reduction for mixed convex, compound, and mesh contacts.
2. Broaden contact tests around grazing capsule edges, clipped box manifolds, mixed mass stacks, and long chains of joints.
3. Improve mesh edge handling and broaden validation of hull and cylinder terrain manifolds.
4. Extend synchronized moving-target sweeps beyond spheres and harden rotating convex CCD. Measure the cost of island-parallel solving against serial in realistic scenes.
5. Add continuous cloth surface contacts and improve fluid boundary sampling.

## Further work

Additional 2D geometry and shape queries, portable snapshot serialization, and geometric fracture beyond authored compounds.
