use rust_physics::{PhysicsWorld, Quat, QueryFilter, RigidBodyBuilder, Shape, Vec3};

#[test]
fn sphere_cast_hits_rounded_box_edge_with_correct_normal() {
    let mut w = PhysicsWorld::new();
    let q = Quat::from_rotation_y(0.34);
    let id = w.add_body(
        RigidBodyBuilder::static_body()
            .rotation(q)
            .shape(Shape::cuboid(Vec3::ONE))
            .build(),
    );
    let filter = QueryFilter::default();
    let hit = w
        .cast_sphere(
            q * Vec3::new(-4.0, 1.4, 0.0),
            0.5,
            q * Vec3::X,
            f64::INFINITY,
            filter,
        )
        .unwrap();
    assert_eq!(id, hit.body);
    assert!((hit.distance - 2.7).abs() < 1e-10);
    assert!(hit.normal.distance(q * Vec3::new(-0.6, 0.8, 0.0)) < 1e-10);
    assert!(hit.position.distance(q * Vec3::new(-1.0, 1.0, 0.0)) < 1e-10);
    assert!(
        w.cast_sphere(
            q * Vec3::new(-4.0, 1.4, 1.4),
            0.5,
            q * Vec3::X,
            10.0,
            filter
        )
        .is_none()
    );
    let inside = w.nearest_surface(Vec3::ZERO, filter).unwrap();
    assert_eq!(inside.body, id);
    assert!((inside.distance + 1.0).abs() < 1e-12);
    assert_eq!(
        w.overlap_sphere(hit.position, 0.1, filter)
            .collect::<Vec<_>>(),
        [id]
    );
}

#[test]
fn ccd_preserves_restitution_and_remaining_motion_against_a_thin_box() {
    let mut w = PhysicsWorld::new();
    w.set_gravity(Vec3::ZERO).unwrap();
    w.add_body(
        RigidBodyBuilder::static_body()
            .shape(Shape::cuboid(Vec3::new(0.01, 2.0, 2.0)))
            .build(),
    );
    let bullet = w.add_body(
        RigidBodyBuilder::dynamic()
            .position(-Vec3::X * 2.0)
            .velocity(Vec3::X * 400.0)
            .material(rust_physics::Material {
                restitution: 1.0,
                ..Default::default()
            })
            .ccd(true)
            .build(),
    );
    let stats = w.step_fixed().unwrap();
    let b = w.body(bullet).unwrap();
    assert_eq!(stats.ccd_impacts, 1);
    assert!(b.position().x < -5.0);
    assert!((b.velocity().x + 400.0).abs() < 1e-9);
}
