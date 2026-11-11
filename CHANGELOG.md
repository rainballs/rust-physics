# Changelog

## 0.1.0 development

- Established a safe Rust rigid-body world, stable handles, double-precision rotation and inertia, collision filtering, and events.
- Added analytical primitive collisions, experimental GJK/EPA and convex hull queries, clipped box manifolds, cached contact impulses, friction, restitution, and island sleeping.
- Added swept sphere impacts, ray and shape queries, and optional median BVH broad phase.
- Added ball, fixed, distance, rope, spring, hinge, and prismatic constraints.
- Added weighted compound colliders, solid convex hulls, static meshes, and heightfields.
- Added synchronized moving-sphere CCD, a sphere character controller, and raycast suspension.
- Added XPBD cloth, volume preservation, particles, optional parallel updates, and an experimental SPH solver.
- Added planar circle and rectangle dynamics, world-axis locks, origin rebasing, compound breakup, and spherical buoyancy.
- Added PBF, SPH boundary impulses, XPBD node self-collision, particle pairs, and parallel narrow-phase work.
- Added a uniform grid, fat-AABB tree, and translational convex casting.
- Fixed CCD event-budget exhaustion so unrelated bodies advance the full timestep.
- Added headless drop, deformable tetrahedron, and stress examples, a graphical sandbox, regression tests, and release benchmark targets.
