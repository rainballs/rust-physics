use rust_physics::{BroadPhase, PhysicsWorld, RigidBodyBuilder, Settings, Shape, Vec3};
use stats_alloc::{INSTRUMENTED_SYSTEM, Region, StatsAlloc};
use std::alloc::System;
use std::hint::black_box;

#[global_allocator]
static GLOBAL: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;

fn main() {
    println!(
        "mode,bodies,contacts,steps,setup_bytes,allocations,reallocations,deallocations,requested_bytes"
    );
    for count in [1_000, 10_000, 100_000] {
        for (broad_phase, floor) in [
            (BroadPhase::SweepAndPrune, false),
            (BroadPhase::SweepAndPrune, true),
            (BroadPhase::DynamicTree, false),
        ] {
            let setup = Region::new(GLOBAL);
            let mut world = PhysicsWorld::new();
            world
                .set_settings(Settings {
                    broad_phase,
                    sleep_time: 0.0,
                    ..Default::default()
                })
                .unwrap();
            if floor {
                world.add_body(RigidBodyBuilder::static_body().shape(Shape::Plane).build());
            }
            for i in 0..count {
                world.add_body(
                    RigidBodyBuilder::dynamic()
                        .position(Vec3::new(
                            f64::from(i) * 2.0,
                            if floor { 0.5 } else { 100.0 },
                            0.0,
                        ))
                        .build(),
                );
            }
            for _ in 0..10 {
                black_box(world.step_fixed().unwrap());
            }
            let setup_bytes = setup.change().bytes_allocated;
            let measure = Region::new(GLOBAL);
            for _ in 0..20 {
                black_box(world.step_fixed().unwrap());
            }
            let stats = measure.change();
            println!(
                "{broad_phase:?},{count},{floor},20,{setup_bytes},{},{},{},{}",
                stats.allocations, stats.reallocations, stats.deallocations, stats.bytes_allocated
            );
        }
    }
}
