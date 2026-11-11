use rust_physics::world::QueryFilter;
use rust_physics::{Material, PhysicsWorld, Quat, RigidBodyBuilder, Shape, Vec3};

fn floor(world: &mut PhysicsWorld) {
    world.add_body(RigidBodyBuilder::static_body().shape(Shape::Plane).build());
}

#[test]
fn free_fall_matches_semi_implicit_solution() {
    let mut world = PhysicsWorld::new();
    let y = 20.0;
    let id = world.add_body(RigidBodyBuilder::dynamic().position(Vec3::Y * y).build());
    for _ in 0..60 {
        world.step_fixed().unwrap();
    }
    let b = world.body(id).unwrap();
    assert!((b.velocity().y + 9.81).abs() < 1e-11);
    assert!((b.position().y - (y - 9.81 * 0.5 * (1.0 + 1.0 / 120.0))).abs() < 1e-11);
}

#[test]
fn isolated_anisotropic_body_preserves_world_momentum() {
    let mut world = PhysicsWorld::new();
    let id = world.add_body(
        RigidBodyBuilder::dynamic()
            .shape(Shape::cuboid(Vec3::new(1.0, 2.0, 3.0)))
            .angular_velocity(Vec3::new(1.0, 2.0, -0.5))
            .build(),
    );
    let initial = world.body(id).unwrap().angular_momentum();
    for _ in 0..600 {
        world.step_fixed().unwrap();
    }
    let b = world.body(id).unwrap();
    assert_eq!(initial, b.angular_momentum());
    assert!((b.rotation().length() - 1.0).abs() < 1e-13);
}

#[test]
fn collisions_preserve_momentum_and_obey_restitution() {
    for restitution in [0.0, 0.4, 1.0] {
        let mut w = PhysicsWorld::new();
        w.set_gravity(Vec3::ZERO).unwrap();
        let material = Material {
            restitution,
            ..Material::default()
        };
        let a = w.add_body(
            RigidBodyBuilder::dynamic()
                .position(-Vec3::X)
                .velocity(Vec3::X * 3.0)
                .mass(2.0)
                .material(material)
                .build(),
        );
        let b = w.add_body(
            RigidBodyBuilder::dynamic()
                .position(Vec3::X)
                .velocity(-Vec3::X)
                .material(material)
                .build(),
        );
        for _ in 0..30 {
            w.step_fixed().unwrap();
        }
        let va = w.body(a).unwrap().velocity().x;
        let vb = w.body(b).unwrap().velocity().x;
        assert!((2.0 * va + vb - 5.0).abs() < 1e-12);
        assert!((vb - va - 4.0 * restitution).abs() < 1e-12);
        let energy = va * va + 0.5 * vb * vb;
        let expected = 25.0 / 6.0 + 16.0 / 3.0 * restitution * restitution;
        assert!((energy - expected).abs() < 1e-12);
    }
}

#[test]
fn aligned_stack_stays_upright() {
    let mut world = PhysicsWorld::new();
    floor(&mut world);
    let handles: Vec<_> = (0..5)
        .map(|i| {
            world.add_body(
                RigidBodyBuilder::dynamic()
                    .shape(Shape::cuboid(Vec3::splat(0.5)))
                    .position(Vec3::Y * (0.5 + i as f64))
                    .build(),
            )
        })
        .collect();
    for _ in 0..720 {
        world.step_fixed().unwrap();
    }
    for (i, id) in handles.into_iter().enumerate() {
        let b = world.body(id).unwrap();
        assert!(
            (b.position() - Vec3::Y * (0.5 + i as f64)).length() < 0.05,
            "{i}: {b:?}"
        );
        assert!((b.rotation() * Vec3::Y).dot(Vec3::Y) > 0.99);
    }
}

#[test]
fn ccd_stops_a_fast_sphere_crossing_a_static_sphere() {
    for enabled in [true, false] {
        let mut w = PhysicsWorld::new();
        w.set_gravity(Vec3::ZERO).unwrap();
        w.add_body(RigidBodyBuilder::static_body().build());
        let bullet = w.add_body(
            RigidBodyBuilder::dynamic()
                .position(-Vec3::X * 5.0)
                .velocity(Vec3::X * 1_000.0)
                .ccd(enabled)
                .build(),
        );
        let step = w.step_fixed().unwrap();
        let b = w.body(bullet).unwrap();
        if enabled {
            assert!(step.ccd_impacts > 0);
            assert!(b.position().x < -0.99);
        } else {
            assert_eq!(step.ccd_impacts, 0);
            assert!(b.position().x > 5.0);
        }
    }
}

#[test]
fn sensors_report_transitions_without_impulses() {
    let mut w = PhysicsWorld::new();
    w.set_gravity(Vec3::ZERO).unwrap();
    let sensor = w.add_body(RigidBodyBuilder::static_body().sensor(true).build());
    let moving = w.add_body(
        RigidBodyBuilder::kinematic()
            .position(-Vec3::X * 2.0)
            .velocity(Vec3::X * 4.0)
            .build(),
    );
    let mut transitions = Vec::new();
    for _ in 0..60 {
        w.step_fixed().unwrap();
        transitions.extend_from_slice(w.events());
    }
    assert_eq!(transitions.len(), 2);
    assert!(transitions[0].started && !transitions[1].started);
    assert!(
        transitions
            .iter()
            .all(|e| e.sensor && e.a == sensor && e.b == moving)
    );
    assert_eq!(w.body(moving).unwrap().velocity(), Vec3::X * 4.0);
}

#[test]
fn raycasts_report_world_normals_and_stale_handles_stay_invalid() {
    let mut w = PhysicsWorld::new();
    let other = PhysicsWorld::new();
    let id = w.add_body(
        RigidBodyBuilder::static_body()
            .shape(Shape::cuboid(Vec3::new(1.0, 2.0, 0.5)))
            .rotation(Quat::from_rotation_z(std::f64::consts::FRAC_PI_2))
            .build(),
    );
    assert!(other.body(id).is_none());
    let hit = w
        .raycast(Vec3::X * 10.0, -Vec3::X * 7.0, 10.0, QueryFilter::default())
        .unwrap();
    assert_eq!(hit.body, id);
    assert!((hit.distance - 8.0).abs() < 1e-12);
    assert!(hit.normal.distance(Vec3::X) < 1e-12);
    let removed = w.remove_body(id).unwrap();
    assert_ne!(w.add_body(removed), id);
    assert!(w.body(id).is_none());
}
