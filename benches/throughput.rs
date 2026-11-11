use rust_physics::particles::{Field, Particle, ParticleSettings, Particles};
use rust_physics::{BroadPhase, PhysicsWorld, QueryFilter, Ray, RigidBodyBuilder, Settings, Vec3};
use std::hint::black_box;
use std::time::Instant;

fn measure(label: &str, n: usize, mut run: impl FnMut()) {
    for _ in 0..8 {
        run();
    }
    let mut samples = [0.0; 20];
    for sample in &mut samples {
        let start = Instant::now();
        run();
        *sample = start.elapsed().as_secs_f64() * 1000.0;
    }
    samples.sort_unstable_by(f64::total_cmp);
    println!(
        "{label},{n},20,{:.4},{:.4}",
        (samples[9] + samples[10]) * 0.5,
        samples[18]
    );
}

fn grid(side: i32, broad_phase: BroadPhase) -> PhysicsWorld {
    let mut world = PhysicsWorld::new();
    world
        .set_settings(Settings {
            broad_phase,
            gravity: Vec3::ZERO,
            ..Settings::default()
        })
        .unwrap();
    for x in 0..side {
        for y in 0..side {
            for z in 0..side {
                world.add_body(
                    RigidBodyBuilder::dynamic()
                        .position(Vec3::new(f64::from(x), f64::from(y), f64::from(z)) * 1.05)
                        .build(),
                );
            }
        }
    }
    world
}

fn main() {
    println!("scenario,objects,samples,median_ms,p95_ms");
    for side in [10, 22, 37] {
        for broad_phase in [
            BroadPhase::SweepAndPrune,
            BroadPhase::Bvh,
            BroadPhase::UniformGrid,
            BroadPhase::DynamicTree,
        ] {
            let mut world = grid(side, broad_phase);
            let count = world.bodies().len();
            measure(&format!("{broad_phase:?} 3D grid"), count, || {
                black_box(world.step_fixed().unwrap());
            });
        }
    }
    let world = grid(22, BroadPhase::Bvh);
    let rays: Vec<_> = (0..256)
        .map(|i| Ray {
            origin: Vec3::new(f64::from(i % 16) * 1.05, 30.0, f64::from(i / 16) * 1.05),
            direction: -Vec3::Y,
            max_distance: 100.0,
        })
        .collect();
    let filter = QueryFilter::default();
    let index = world.spatial_index();
    measure("direct 256 rays", world.bodies().len(), || {
        for hit in world.raycast_batch(&rays, filter) {
            black_box(hit);
        }
    });
    measure("indexed 256 rays", world.bodies().len(), || {
        for hit in index.raycast_batch(&rays, filter) {
            black_box(hit);
        }
    });
    measure("BVH index build", world.bodies().len(), || {
        black_box(world.spatial_index());
    });

    #[cfg(feature = "rayon")]
    for parallel_narrow_phase in [false, true] {
        use rust_physics::Shape;
        let mut world = PhysicsWorld::new();
        world
            .set_settings(Settings {
                parallel_narrow_phase,
                ..Default::default()
            })
            .unwrap();
        for i in 0..1_000 {
            let position = Vec3::X * (f64::from(i) * 4.0);
            world.add_body(
                RigidBodyBuilder::static_body()
                    .position(position)
                    .shape(Shape::cuboid(Vec3::ONE))
                    .build(),
            );
            world.add_body(
                RigidBodyBuilder::static_body()
                    .sensor(true)
                    .position(position + Vec3::X * 1.4)
                    .shape(Shape::cylinder(0.5, 0.7))
                    .build(),
            );
        }
        let label = if parallel_narrow_phase {
            "parallel mixed convex sensors"
        } else {
            "serial mixed convex sensors"
        };
        measure(label, 1_000, || {
            let stats = world.step_fixed().unwrap();
            assert_eq!(stats.contact_points, 2_000);
            black_box(stats);
        });
    }

    for n in [10_000, 100_000, 1_000_000] {
        let mut aos = vec![
            Particle {
                position: Vec3::X,
                velocity: Vec3::Y,
                remaining: 60.0
            };
            n
        ];
        let mut serial = Particles::default();
        #[cfg(feature = "rayon")]
        let mut parallel = Particles::default();
        for &p in &aos {
            serial.push(p);
            #[cfg(feature = "rayon")]
            parallel.push(p);
        }
        let dt = 1.0 / 60.0;
        let strength = 1.2;
        let gravity = Vec3::new(0.0, -9.81, 0.0);
        let settings = ParticleSettings {
            gravity,
            ..Default::default()
        };
        let fields = [Field::Vortex {
            center: Vec3::ZERO,
            axis: Vec3::Y,
            strength,
        }];
        measure("particles AoS", n, || {
            for p in black_box(&mut aos) {
                p.remaining -= dt;
                if p.remaining > 0.0 {
                    p.velocity += (gravity + Vec3::Y.cross(p.position) * strength) * dt;
                    p.position += p.velocity * dt;
                }
            }
            aos.retain(|p| p.remaining > 0.0);
        });
        measure("particles SoA", n, || {
            black_box(serial.step(dt, settings, &fields));
        });
        #[cfg(feature = "rayon")]
        measure("particles SoA Rayon", n, || {
            black_box(parallel.step_parallel(dt, settings, &fields));
        });
    }
}
