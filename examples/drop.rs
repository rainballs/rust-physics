use rust_physics::{PhysicsWorld, RigidBodyBuilder, Shape, Vec3};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut world = PhysicsWorld::new();
    world.add_body(RigidBodyBuilder::static_body().shape(Shape::Plane).build());
    let ball = world.add_body(RigidBodyBuilder::dynamic().position(Vec3::Y * 5.0).build());

    for frame in 0..180 {
        let stats = world.step_fixed()?;
        if frame % 30 == 29 {
            let body = world.body(ball).expect("live body");
            println!(
                "{:.1} s  y={:.4} m  vy={:.4} m/s  {} contacts",
                (frame + 1) as f64 * world.settings().fixed_dt,
                body.position().y,
                body.velocity().y,
                stats.contact_points
            );
        }
    }
    Ok(())
}
