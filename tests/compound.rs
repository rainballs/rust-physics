use rust_physics::compound::{ColliderPart, Compound};
use rust_physics::{PhysicsWorld, Quat, QueryFilter, RigidBodyBuilder, Shape, Vec3};
use std::sync::Arc;

fn part(position: Vec3, weight: f64) -> ColliderPart {
    ColliderPart {
        shape: Shape::sphere(0.25),
        position,
        rotation: Quat::IDENTITY,
        weight,
    }
}

#[test]
fn compound_recenters_weighted_parts_and_uses_parallel_axis_inertia() {
    let compound =
        Arc::new(Compound::new(vec![part(Vec3::ZERO, 1.0), part(Vec3::X * 2.0, 3.0)]).unwrap());
    assert_eq!(compound.input_center(), Vec3::X * 1.5);
    assert_eq!(compound.parts()[0].position, Vec3::X * -1.5);
    assert_eq!(compound.parts()[1].position, Vec3::X * 0.5);
    let body = RigidBodyBuilder::dynamic()
        .shape(Shape::Compound(compound))
        .mass(4.0)
        .angular_velocity(Vec3::new(1.0, 2.0, 3.0))
        .build();
    let inverse = body.world_inverse_inertia();
    assert!((inverse.x_axis.x - 10.0).abs() < 1e-12);
    assert!((inverse.y_axis.y - 1.0 / 3.1).abs() < 1e-12);
    assert!((inverse.z_axis.z - 1.0 / 3.1).abs() < 1e-12);
    assert!(body.angular_velocity().distance(Vec3::new(1.0, 2.0, 3.0)) < 1e-12);
}

#[test]
fn compound_children_collide_and_queries_return_the_parent() {
    let geometry = Arc::new(Compound::new(vec![part(-Vec3::X, 1.0), part(Vec3::X, 1.0)]).unwrap());
    let mut w = PhysicsWorld::new();
    w.add_body(RigidBodyBuilder::static_body().shape(Shape::Plane).build());
    let id = w.add_body(
        RigidBodyBuilder::dynamic()
            .shape(Shape::Compound(geometry))
            .position(Vec3::Y * 2.0)
            .build(),
    );
    for _ in 0..180 {
        w.step_fixed().unwrap();
    }
    let body = w.body(id).unwrap();
    assert!((body.position().y - 0.25).abs() < 0.01);
    let filter = QueryFilter::default();
    let hit = w
        .raycast(Vec3::new(1.0, 3.0, 0.0), -Vec3::Y, 4.0, filter)
        .unwrap();
    assert_eq!(hit.body, id);
    assert!((hit.position.y - 0.5).abs() < 0.01);
    assert!(
        w.overlap_sphere(Vec3::new(0.0, 0.25, 0.0), 0.1, filter)
            .next()
            .is_none()
    );
}
