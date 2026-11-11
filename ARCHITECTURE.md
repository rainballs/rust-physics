# Architecture

A Cargo workspace contains one library crate. Module boundaries keep shape geometry, integration, constraints, and particle-based solvers independently callable. The renderer and benchmarks are separate Cargo targets. Glam supplies double-precision math. Optional dependencies add a renderer to the sandbox and Rayon to independent particle motion.

| Module | Responsibility |
|---|---|
| `body`, `integrators` | Validated rigid-body construction, mass properties, forces, pose integration, and standalone ODE methods |
| `collision`, `spatial` | Narrow-phase contact generation, AABBs, shape projections, casts, and flat BVH |
| `convex`, `hull` | Experimental GJK/EPA, support maps, and indexed convex hull mass properties |
| `compound`, `terrain` | Child colliders, static triangle BVHs, and heightfield triangulation |
| `character`, `vehicle` | Swept kinematic sphere motion and raycast suspension |
| `world`, `joints` | Handles, dense storage, broad phase, solver rows, warm starts, sleeping, events, and queries |
| `grid`, `particles`, `softbody`, `fluid` | Spatial hash, particle storage, XPBD, and experimental SPH |
| `debug` | Renderer-independent line geometry |

## Rigid substep

1. Accumulate gravity, forces and torque into velocities and world-space angular momentum. Apply exponential damping.
2. Prepare joint rows and iterate them before pose integration. This avoids advancing through constraint motion that the current impulses can already resolve.
3. Advance poses. Opted-in spheres sweep against filtered static targets, solve impacts, and advance the remaining time up to four impacts per sphere. Optional synchronized CCD sweeps against translating spheres and advances the world through up to 32 impact events.
4. Build finite AABB candidates with sweep-and-prune or a median-split BVH. Test planes separately. Sort candidate pairs by handle before narrow-phase contact generation.
5. Find dynamic contact islands, wake connected bodies as needed, and prepare contact rows using local-anchor impulse matching.
6. Warm-start contacts and interleave joint and contact rows. Joint rows retain their pre-advance Jacobians for this pass. Translate bodies to reduce residual contact penetration without changing velocities.
7. Update island sleep timers and publish begin/end transitions for substep overlaps. Swept impacts are also available as an independently recorded, time-ordered sequence.
8. On the final substep, retain sampled debug contacts. Clear external force and torque after the public step.

Bodies live in a dense vector and a handle-to-index map. World-scoped serial IDs are never reused. Removal swap-compacts storage, updates the moved index, and removes linked joints. Sorted flat buffers retain contact manifolds, events, and impulses across substeps, minimizing per-contact allocations. Public mutable body access invalidates that body's cached impulses.

Candidate order, contact order, and joint order are stable for an identical construction and input sequence on the same build and machine. A fixed timestep helps reproducibility. Floating-point results across targets, compilers, or different insertion histories may differ.

Spatial queries can scan dense storage or borrow a BVH snapshot that prohibits world mutation during reuse. Particle-based solvers optionally borrow the rigid world for boundary projections. Their updates do not alter rigid bodies.
