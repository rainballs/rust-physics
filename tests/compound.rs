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

#[test]
fn ray_from_inside_overlapping_parts_hits_exposed_union_surface() {
    let geometry =
        Arc::new(Compound::new(vec![part(Vec3::X * -0.1, 1.0), part(Vec3::X * 0.1, 1.0)]).unwrap());
    let mut world = PhysicsWorld::new();
    let id = world.add_body(
        RigidBodyBuilder::static_body()
            .shape(Shape::Compound(geometry))
            .build(),
    );
    let hit = world
        .raycast(Vec3::ZERO, Vec3::X, 1.0, QueryFilter::default())
        .unwrap();
    assert_eq!(hit.body, id);
    assert!((hit.distance - 0.35).abs() < 1e-12);
    assert_eq!(hit.normal, Vec3::X);
}

#[test]
fn fracture_inherits_velocity_field_and_preserves_momenta_and_energy() {
    let geometry =
        Arc::new(Compound::new(vec![part(Vec3::X * -0.7, 1.0), part(Vec3::X * 0.9, 2.0)]).unwrap());
    let mut world = PhysicsWorld::new();
    let id = world.add_body(
        RigidBodyBuilder::dynamic()
            .shape(Shape::Compound(geometry))
            .mass(3.0)
            .position(Vec3::new(2.0, 1.0, -3.0))
            .rotation(Quat::from_rotation_y(0.4))
            .velocity(Vec3::new(3.0, -1.0, 2.0))
            .angular_velocity(Vec3::new(-0.5, 2.0, 4.0))
            .build(),
    );
    let invariants = |b: &rust_physics::RigidBody| {
        let p = b.velocity() / b.inverse_mass();
        let angular = b.angular_momentum() + b.position().cross(p);
        let energy = 0.5 * (p.dot(b.velocity()) + b.angular_momentum().dot(b.angular_velocity()));
        (p, angular, energy)
    };
    let (initial_p, initial_l, initial_e) = invariants(world.body(id).unwrap());
    let pieces = world.fracture_compound(id).unwrap();
    assert_eq!(pieces.len(), 2);
    assert!(world.body(id).is_none());
    let (p, l, e) = pieces
        .into_iter()
        .map(|id| invariants(world.body(id).unwrap()))
        .fold((Vec3::ZERO, Vec3::ZERO, 0.0), |(p, l, e), (a, b, c)| {
            (p + a, l + b, e + c)
        });
    assert!(p.distance(initial_p) < 1e-11);
    assert!(l.distance(initial_l) < 1e-11);
    assert!((e - initial_e).abs() < 1e-11);
}
