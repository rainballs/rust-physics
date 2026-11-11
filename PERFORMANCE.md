# Performance

## Method

Release benchmarks ran on an AMD Ryzen AI 9 465 with 20 logical CPUs and 30 GiB RAM, on Arch Linux with Rust 1.99.1. Rigid solving was serial. The parallel particle and narrow-phase cases used four Rayon workers. Each scenario recorded 20 timed samples after 8 or 10 warmups. The desktop session ran without fixed CPU clocks or affinity.

```sh
RAYON_NUM_THREADS=4 cargo bench --features rayon --bench scaling --bench throughput --bench fluids
```

The rigid benchmarks use a 60 Hz fixed step, two substeps, and twelve velocity iterations. Separated spheres fall along a line. The floor scenario has one contact per sphere, with sleeping disabled. The grid scenario has unit-diameter spheres spaced 1.05 units apart in a stationary 3D lattice with zero gravity, isolating broad-phase overhead. The query benchmark reuses an index for batches of 256 vertical rays. The mixed convex sensor workload generates 1,000 cylinder–box contact manifolds per substep. Sensors exercise geometry and event generation without applying solver impulses.

## Rigid step latency

| Dynamic bodies | Separated median / p95, ms | Independent floor contacts median / p95, ms |
|---:|---:|---:|
| 100 | 0.0037 / 0.0046 | 0.0239 / 0.0240 |
| 1,000 | 0.0374 / 0.0399 | 0.2404 / 0.2448 |
| 10,000 | 0.4310 / 0.4508 | 2.6037 / 2.7710 |
| 50,000 | 4.2383 / 4.5956 | 18.3840 / 20.6107 |
| 100,000 | 10.9654 / 11.7064 | 38.5309 / 38.9377 |

The 100,000-contact median fell from 300.96 ms in the original tree-backed implementation to 161.08 ms with reusable flat buffers, then about 86 ms after scheduling independent manifolds together. Early exit when an independent manifold reaches a fixed point brought the median to 38.53 ms in the measured run. These scenarios do not represent densely coupled stacks.

## Broad phase, queries, and parallel narrow phase

| Workload | Objects or pairs | Median / p95, ms |
|---|---:|---:|
| SweepAndPrune 3D grid | 1,000 | 0.2547 / 0.2577 |
| Bvh 3D grid | 1,000 | 0.2632 / 0.2662 |
| UniformGrid 3D grid | 1,000 | 0.3577 / 0.3820 |
| DynamicTree 3D grid | 1,000 | 0.0872 / 0.1150 |
| SweepAndPrune 3D grid | 10,648 | 6.3283 / 6.3756 |
| Bvh 3D grid | 10,648 | 3.4244 / 3.5142 |
| UniformGrid 3D grid | 10,648 | 4.3781 / 8.3483 |
| DynamicTree 3D grid | 10,648 | 1.2860 / 1.3574 |
| SweepAndPrune 3D grid | 50,653 | 86.3199 / 87.2444 |
| Bvh 3D grid | 50,653 | 23.1210 / 23.3542 |
| UniformGrid 3D grid | 50,653 | 48.6686 / 59.1795 |
| DynamicTree 3D grid | 50,653 | 13.9874 / 14.1702 |
| direct 256 rays | 10,648 | 29.4355 / 30.2718 |
| indexed 256 rays | 10,648 | 0.5219 / 0.5409 |
| BVH index build | 10,648 | 1.0282 / 1.0406 |
| serial mixed convex sensors | 1,000 | 4.3080 / 4.8534 |
| parallel mixed convex sensors | 1,000 | 1.0296 / 1.3594 |

The fat-AABB tree was fastest on this stationary lattice after replacing repeated root-to-leaf queries with a paired tree traversal. It retains its hierarchy between steps, while the median BVH rebuilds. Rapidly changing scenes can require enough reinsertion to change that ranking. Sweep-and-prune remains the default and works well on the separated line. The uniform grid bounds cell replication by using a direct fallback for oversized proxies.

The indexed ray batch amortizes a separate snapshot build. Rayon improved the mixed-convex sensor workload in this run. The parallel path adds scratch space and can be slower for inexpensive contact geometry or smaller scenes.

## Particle and fluid updates

| Particles | Specialized AoS median, ms | Configurable serial SoA median, ms | Configurable four-worker SoA median, ms |
|---:|---:|---:|---:|
| 10,000 | 0.0166 | 0.0431 | 0.0272 |
| 100,000 | 0.1813 | 0.4336 | 0.2190 |
| 1,000,000 | 2.7200 | 4.3731 | 1.6401 |

The particle benchmark applies a vortex and gravity, without collision or expiration during measurement. The AoS baseline specializes the loop for that field. The layout difference alone cannot explain the result.

Fluid timings use a compressed 3D lattice with a 240 Hz step, no external boundary, and a rest density of 960. PBF runs five projection iterations per step.

| Solver | Particles | Median / p95, ms |
|---|---:|---:|
| SPH | 216 | 0.2269 / 0.2388 |
| PBF (5 projections) | 216 | 0.8391 / 0.9175 |
| SPH | 1,000 | 0.8622 / 0.8830 |
| PBF (5 projections) | 1,000 | 4.9220 / 5.0235 |
| SPH | 4,096 | 3.2344 / 3.2443 |
| PBF (5 projections) | 4,096 | 22.2404 / 22.8370 |

Reusing exact neighbor pairs reduced the 4,096-particle SPH median from 6.01 to 3.23 ms and PBF from 38.78 to 22.24 ms on the measured lattice. Pair storage grows with the number of actual neighbors.

## Memory and stress validation

The scaling benchmark reached 170,536 KiB peak RSS and the throughput benchmark reached 202,392 KiB. These are process-level Linux child resource peaks. The throughput program holds all three particle layouts simultaneously at its largest size. Per-step allocator activity and hardware cache misses were not instrumented.

The release soak ran 192 seeds and 691,200 fixed steps with 48 mixed rigid bodies per seed, repeated removal and insertion, impulses, CCD, and all four broad phases. It recorded 137,882,267 contact points and 7,826 swept impacts in 174.81 seconds. The soak checked finite state, normalized contacts and orientations, and bounded poses. It is a stability check rather than an isolated throughput benchmark.
