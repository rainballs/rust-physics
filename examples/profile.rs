use rust_physics::{PhysicsWorld, RigidBodyBuilder, Shape, Vec3};
use std::hint::black_box;
use std::time::Duration;
type Phase = (&'static str, fn(rust_physics::StepTimings) -> Duration);

fn main() {
    let mut world = PhysicsWorld::new();
    let mut settings = world.settings().clone();
    settings.sleep_time = 0.0;
    world.set_settings(settings).unwrap();
    world.add_body(RigidBodyBuilder::static_body().shape(Shape::Plane).build());
    let count = std::env::args()
        .nth(1)
        .map_or(10_000, |s| s.parse::<usize>().expect("body count"));
    for i in 0..count {
        world.add_body(
            RigidBodyBuilder::dynamic()
                .position(Vec3::new(i as f64 * 2.0, 0.5, 0.0))
                .build(),
        );
    }
    for _ in 0..10 {
        black_box(world.step_fixed().unwrap());
    }
    let samples: Vec<_> = (0..20)
        .map(|_| world.step_fixed_profiled().unwrap().timings)
        .collect();
    let phases: [Phase; 7] = [
        ("forces and joints", |t: rust_physics::StepTimings| {
            t.forces_and_joints
        }),
        ("pose and CCD", |t: rust_physics::StepTimings| {
            t.pose_and_ccd
        }),
        ("broad phase", |t: rust_physics::StepTimings| t.broad_phase),
        ("narrow phase", |t: rust_physics::StepTimings| {
            t.narrow_phase
        }),
        ("solver", |t: rust_physics::StepTimings| t.solver),
        ("sleep and events", |t: rust_physics::StepTimings| {
            t.sleep_and_events
        }),
        ("total", |t: rust_physics::StepTimings| t.total),
    ];
    for (name, phase) in phases {
        let mut times: Vec<_> = samples
            .iter()
            .map(|&t| phase(t).as_secs_f64() * 1000.0)
            .collect();
        times.sort_unstable_by(f64::total_cmp);
        println!("{name:18} {:>9.3} ms", (times[9] + times[10]) * 0.5);
    }
}
