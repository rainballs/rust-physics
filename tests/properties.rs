use rust_physics::collision::contacts;
use rust_physics::{PhysicsWorld, Quat, RigidBody, RigidBodyBuilder, Shape, Vec3};

fn vector(t: f64) -> Vec3 {
    Vec3::new((t * 0.729).sin(), (t * 0.381).cos(), (t * 1.147).sin())
}

fn shape(shape: Shape, position: Vec3, rotation: Quat) -> RigidBody {
    RigidBodyBuilder::static_body()
        .shape(shape)
        .position(position)
        .rotation(rotation)
        .build()
}

#[test]
fn generated_sphere_contacts_match_analytical_depth_and_reverse() {
    let mut ab = Vec::new();
    let mut ba = Vec::new();
    for i in 1..=512 {
        let t = f64::from(i);
        let r1 = 0.2 + (t * 0.17).sin().abs();
        let r2 = 0.2 + (t * 0.39).cos().abs();
        let p1 = vector(t + 2.0);
        let p2 = p1 + vector(t) * 1.7;
        let a = shape(Shape::sphere(r1), p1, Quat::IDENTITY);
        let b = shape(Shape::sphere(r2), p2, Quat::IDENTITY);
        ab.clear();
        ba.clear();
        contacts(&a, &b, &mut ab);
        contacts(&b, &a, &mut ba);
        let depth = r1 + r2 - p1.distance(p2);
        assert_eq!(ab.len(), usize::from(depth >= 0.0));
        assert_eq!(ab.len(), ba.len());
        if let (Some(a), Some(b)) = (ab.first(), ba.first()) {
            assert!((a.depth - depth).abs() < 1e-14);
            assert!(a.point.distance(b.point) < 1e-14);
            assert!(a.normal.distance(-b.normal) < 1e-14);
        }
    }
}

#[test]
fn generated_box_manifolds_transform_with_the_scene() {
    let rotation = Quat::from_scaled_axis(Vec3::new(0.5, -0.9, 0.2));
    let translation = Vec3::new(10.0, -5.0, 8.0);
    let mut reference = Vec::new();
    let mut changed = Vec::new();
    for i in 1..=256 {
        let t = f64::from(i);
        let pos = vector(t) * 1.1;
        let qa = Quat::from_scaled_axis(vector(t + 1.0) * 0.8);
        let qb = Quat::from_scaled_axis(vector(t + 7.0) * 0.9);
        let ha = Shape::cuboid(Vec3::new(0.4, 0.6, 0.8));
        let hb = Shape::cuboid(Vec3::new(0.7, 0.9, 0.5));
        let a = shape(ha.clone(), Vec3::ZERO, qa);
        let b = shape(hb.clone(), pos, qb);
        let moved_a = shape(ha, translation, rotation * qa);
        let moved_b = shape(hb, translation + rotation * pos, rotation * qb);
        reference.clear();
        changed.clear();
        contacts(&a, &b, &mut reference);
        contacts(&moved_a, &moved_b, &mut changed);
        assert_eq!(reference.len(), changed.len(), "{i}");
        for contact in &reference {
            assert!(
                changed
                    .iter()
                    .any(|moved| (contact.depth - moved.depth).abs() < 1e-10
                        && (rotation * contact.normal).distance(moved.normal) < 1e-10
                        && (translation + rotation * contact.point).distance(moved.point) < 1e-10),
                "{i}: {contact:?} / {changed:?}"
            );
        }
    }
}

#[test]
fn torque_free_rotation_bounds_energy_drift_at_small_steps() {
    let mut world = PhysicsWorld::new();
    let id = world.add_body(
        RigidBodyBuilder::dynamic()
            .shape(Shape::cuboid(Vec3::new(0.5, 1.0, 1.5)))
            .angular_velocity(Vec3::new(1.3, -0.8, 2.1))
            .build(),
    );
    let kinetic = |body: &RigidBody| 0.5 * body.angular_momentum().dot(body.angular_velocity());
    let reference = kinetic(world.body(id).unwrap());
    let mut maximum: f64 = 0.0;
    for _ in 0..10_000 {
        world.step(0.002).unwrap();
        maximum = maximum.max((kinetic(world.body(id).unwrap()) / reference - 1.0).abs());
    }
    assert!(maximum < 1e-5, "maximum relative drift {maximum}");
}

#[test]
fn settings_validation_is_non_mutating_and_invalid_dt_preserves_force() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vec3::ZERO).unwrap();
    let id = world.add_body(RigidBodyBuilder::dynamic().build());
    assert!(world.add_force_at_point(id, Vec3::X * 2.0, Vec3::ZERO));
    let settings = rust_physics::Settings {
        iterations: 0,
        ..world.settings().clone()
    };
    assert!(world.set_settings(settings).is_err());
    assert!(world.step(f64::NAN).is_err());
    assert_eq!(world.body(id).unwrap().velocity(), Vec3::ZERO);
    world.step_fixed().unwrap();
    assert!(world.body(id).unwrap().velocity().distance(Vec3::X / 30.0) < 1e-13);
}

#[test]
fn kinematic_surface_transfers_tangential_motion_to_dynamic_body() {
    let mut world = PhysicsWorld::new();
    world.add_body(
        RigidBodyBuilder::kinematic()
            .shape(Shape::cuboid(Vec3::new(8.0, 0.2, 2.0)))
            .position(-Vec3::Y * 0.2)
            .velocity(Vec3::X)
            .build(),
    );
    let id = world.add_body(
        RigidBodyBuilder::dynamic()
            .shape(Shape::cuboid(Vec3::splat(0.3)))
            .position(Vec3::Y * 0.3)
            .build(),
    );
    for _ in 0..120 {
        world.step_fixed().unwrap();
    }
    let body = world.body(id).unwrap();
    assert!(body.velocity().distance(Vec3::X) < 0.02);
    assert!((body.position().y - 0.3).abs() < 0.01);
}
