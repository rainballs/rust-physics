use std::hint::black_box;
use std::time::Instant;

use rust_physics::{PhysicsWorld, RigidBodyBuilder, Vec3};

fn main() {
    println!("scenario,bodies,samples,median_ms,p95_ms");
    for n in [100, 1_000, 10_000, 50_000, 100_000] {
        for contacts in [false, true] {
            let mut world = PhysicsWorld::new();
            if contacts {
                world.add_body(
                    RigidBodyBuilder::static_body()
                        .shape(rust_physics::Shape::Plane)
                        .build(),
                );
                let mut settings = world.settings().clone();
                settings.sleep_time = 0.0;
                world.set_settings(settings).unwrap();
            }
            for i in 0..n {
                world.add_body(
                    RigidBodyBuilder::dynamic()
                        .position(Vec3::new(
                            i as f64 * 2.0,
                            if contacts { 0.5 } else { 100.0 },
                            0.0,
                        ))
                        .build(),
                );
            }
            for _ in 0..10 {
                black_box(world.step_fixed().unwrap());
            }
            let mut times: Vec<f64> = (0..20)
                .map(|_| {
                    let start = Instant::now();
                    black_box(world.step_fixed().unwrap());
                    start.elapsed().as_secs_f64() * 1000.0
                })
                .collect();
            times.sort_unstable_by(f64::total_cmp);
            let scenario = if contacts {
                "independent floor contacts"
            } else {
                "separated free fall"
            };
            println!(
                "{scenario},{n},20,{:.4},{:.4}",
                (times[9] + times[10]) * 0.5,
                times[18]
            );
        }
    }
}
