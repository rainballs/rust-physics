# Performance

## Method

Release benchmarks ran on an AMD Ryzen AI 9 465 with 20 logical CPUs and 30 GiB RAM, on Arch Linux with Rust 1.99.1. The rigid solver was serial. The Rayon particle run used four workers. Each scenario recorded 20 timed samples after 8 or 10 warmups. These are wall-clock results from a desktop session, without fixed CPU clocks or affinity.

```sh
RAYON_NUM_THREADS=4 cargo bench --features rayon --bench scaling --bench throughput
```

The rigid benchmarks use a 60 Hz fixed step, two substeps, and twelve solver iterations. Separated spheres fall on a line with no contacts. The floor scenario has one static plane and one contact per sphere, with sleeping disabled to keep the contacts active. The grid scenario has unit-diameter spheres on a 1.05-unit 3D lattice with zero gravity. It isolates broad-phase and step overhead from the contact solver. The query benchmark reuses an index for each batch of 256 vertical rays through a 10,648-body grid. The particle benchmark applies a vortex and gravity, without collisions or expiration during measurement. Its AoS baseline uses a specialized loop for that field, while the SoA versions use the configurable particle API.

## Rigid step latency

| Dynamic bodies | Separated median / p95, ms | Independent floor contacts median / p95, ms |
|---:|---:|---:|
| 100 | 0.0058 / 0.0058 | 0.1043 / 0.1094 |
| 1,000 | 0.0390 / 0.0422 | 0.6922 / 0.6982 |
| 10,000 | 0.4425 / 0.4502 | 7.0662 / 7.2118 |
| 50,000 | 4.5390 / 4.9991 | 41.6118 / 42.5225 |
| 100,000 | 10.8147 / 11.1211 | 85.8144 / 87.5003 |

The 100,000-contact median fell from 300.96 ms in the original tree-backed implementation to 161.08 ms after reusable flat buffers, then 85.81 ms after scheduling independent manifolds together and avoiding repeated work for zero slip. The initial and final runs used the same floor geometry and solver settings. These timings do not represent dense stacked scenes.

## Broad phase and queries

| Workload | Median / p95, ms |
|---|---:|
| 1,000 grid, sweep-and-prune | 0.2557 / 0.4494 |
| 1,000 grid, BVH | 0.2364 / 0.2457 |
| 10,648 grid, sweep-and-prune | 6.0452 / 6.0770 |
| 10,648 grid, BVH | 5.7439 / 5.8101 |
| 50,653 grid, sweep-and-prune | 81.9253 / 84.5149 |
| 50,653 grid, BVH | 34.0471 / 34.9106 |
| 256 direct rays | 27.0221 / 28.7765 |
| 256 indexed rays | 0.4845 / 0.4947 |
| 10,648-body index build | 1.0148 / 1.0312 |

Sweep-and-prune is sensitive to overlap along the selected axis, even when full AABBs are disjoint. The BVH rebuilds each step and improves this large 3D grid. A previous run slightly favored sweep-and-prune for the 1,000-body scene, so the small-scene ranking is workload- and run-dependent.

## Particle update

| Particles | Specialized AoS median, ms | Configurable serial SoA median, ms | Configurable 4-worker SoA median, ms |
|---:|---:|---:|---:|
| 10,000 | 0.0169 | 0.0430 | 0.0741 |
| 100,000 | 0.1844 | 0.4312 | 0.3286 |
| 1,000,000 | 2.8238 | 4.3739 | 1.7142 |

Rayon helped at larger counts in this run and cost more at 10,000 particles. The AoS result includes a simpler dispatch path, so the difference cannot be attributed to layout alone.

The largest scaling process reached 153,264 KiB peak RSS. The throughput process reached 212,056 KiB while holding three particle pools simultaneously at its largest size. These are process-level peaks from Linux child resource usage. Per-step allocations and cache misses were not instrumented. Coupled contact scenes, cylinder/EPA costs, and particle boundary handling still need targeted profiles.
