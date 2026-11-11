use rust_physics::planar::{PlanarBodyBuilder, PlanarShape, PlanarWorld};
use rust_physics::{Material, QueryFilter, Vec2};

#[test]
fn planar_circle_uses_disk_inertia_and_ray_geometry() {
    let mut world = PlanarWorld::new();
    world.set_gravity(Vec2::ZERO).unwrap();
    let id = world
        .add_body(PlanarBodyBuilder::dynamic(PlanarShape::Circle { radius: 0.5 }).mass(2.0))
        .unwrap();
    assert!(world.apply_impulse(id, Vec2::X, -Vec2::Y * 0.5));
    let pose = world.pose(id).unwrap();
    assert!(pose.velocity.distance(Vec2::X * 0.5) < 1e-12);
    assert!((pose.spin - 2.0).abs() < 1e-12);
    let hit = world
        .raycast(-Vec2::X * 2.0, Vec2::X, 3.0, QueryFilter::default())
        .unwrap();
    assert_eq!(hit.body, id);
    assert_eq!(hit.normal, -Vec2::X);
    assert!((hit.distance - 1.5).abs() < 1e-12);
}

#[test]
fn planar_box_stack_uses_in_plane_separation_axes() {
    let mut w = PlanarWorld::new();
    w.add_body(PlanarBodyBuilder::static_body(PlanarShape::HalfPlane))
        .unwrap();
    let shape = PlanarShape::Rectangle {
        half_extents: Vec2::splat(0.3),
    };
    let ids: Vec<_> = (0..5)
        .map(|i| {
            w.add_body(
                PlanarBodyBuilder::dynamic(shape).position(Vec2::Y * (0.3 + f64::from(i) * 0.6)),
            )
            .unwrap()
        })
        .collect();
    for _ in 0..450 {
        w.step_fixed().unwrap();
    }
    for (i, id) in ids.into_iter().enumerate() {
        let pose = w.pose(id).unwrap();
        assert!(
            pose.position.distance(Vec2::Y * (0.3 + i as f64 * 0.6)) < 0.02,
            "{pose:?}"
        );
        assert!(pose.radians.abs() < 0.01);
    }
}

#[test]
fn planar_circles_exchange_velocity_on_elastic_collision() {
    let mut w = PlanarWorld::new();
    w.set_gravity(Vec2::ZERO).unwrap();
    let shape = PlanarShape::Circle { radius: 0.4 };
    let material = Material {
        restitution: 1.0,
        ..Default::default()
    };
    let a = w
        .add_body(
            PlanarBodyBuilder::dynamic(shape)
                .position(-Vec2::X)
                .velocity(Vec2::X * 2.0)
                .material(material),
        )
        .unwrap();
    let b = w
        .add_body(
            PlanarBodyBuilder::dynamic(shape)
                .position(Vec2::X)
                .velocity(-Vec2::X)
                .material(material),
        )
        .unwrap();
    for _ in 0..80 {
        w.step_fixed().unwrap();
    }
    assert!(w.pose(a).unwrap().velocity.distance(-Vec2::X) < 1e-12);
    assert!(w.pose(b).unwrap().velocity.distance(Vec2::X * 2.0) < 1e-12);
}
