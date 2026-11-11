use rust_physics::character::SphereController;
use rust_physics::vehicle::{RaycastVehicle, VehicleInput, Wheel};
use rust_physics::{PhysicsWorld, QueryFilter, RigidBodyBuilder, Shape, Vec3};

#[test]
fn sphere_controller_slides_along_wall_and_snaps_to_floor() {
    let mut w = PhysicsWorld::new();
    let floor = w.add_body(RigidBodyBuilder::static_body().shape(Shape::Plane).build());
    let wall = w.add_body(
        RigidBodyBuilder::static_body()
            .position(Vec3::X * 2.0)
            .shape(Shape::cuboid(Vec3::new(0.05, 3.0, 10.0)))
            .build(),
    );
    let controller = SphereController::default();
    let start = Vec3::new(0.0, 0.5, 0.0);
    let motion = controller
        .move_by(&w, start, Vec3::new(5.0, 0.0, 3.0), QueryFilter::default())
        .unwrap();
    assert!((motion.position.x - 1.5).abs() < 0.003);
    assert!((motion.position.z - 3.0).abs() < 1e-12);
    assert!((motion.position.y - 0.451).abs() < 1e-12);
    assert_eq!(motion.ground.unwrap().body, floor);
    assert_eq!(motion.hits[0].body, wall);
}

#[test]
fn vehicle_suspension_supports_expected_static_load() {
    let mut w = PhysicsWorld::new();
    w.add_body(RigidBodyBuilder::static_body().shape(Shape::Plane).build());
    let chassis = w.add_body(
        RigidBodyBuilder::dynamic()
            .position(Vec3::Y)
            .shape(Shape::cuboid(Vec3::new(0.8, 0.2, 1.0)))
            .mass(1_000.0)
            .build(),
    );
    let wheels = [-0.7, 0.7]
        .into_iter()
        .flat_map(|x| {
            [-0.9, 0.9].into_iter().map(move |z| Wheel {
                anchor: Vec3::new(x, 0.0, z),
                radius: 0.25,
                rest_length: 0.7,
                stiffness: 20_000.0,
                damping: 1_500.0,
                friction: 1.1,
                lateral_stiffness: 1_000.0,
                steerable: z > 0.0,
                driven: true,
            })
        })
        .collect();
    let car = RaycastVehicle { chassis, wheels };
    for _ in 0..360 {
        car.apply(&mut w, VehicleInput::default(), QueryFilter::default())
            .unwrap();
        w.step_fixed().unwrap();
    }
    let body = w.body(chassis).unwrap();
    let expected = 0.95 - 1_000.0 * 9.81 / (4.0 * 20_000.0);
    assert!((body.position().y - expected).abs() < 1e-5);
    let contacts = car
        .apply(
            &mut w,
            VehicleInput {
                drive_force: 100_000.0,
                ..Default::default()
            },
            QueryFilter::default(),
        )
        .unwrap();
    assert_eq!(contacts.len(), 4);
    assert!(
        contacts
            .iter()
            .all(|c| (c.force.z - 1.1 * c.normal_load).abs() < 1e-9)
    );
}

#[test]
fn sphere_buoyancy_matches_hemisphere_volume_and_centroid() {
    use rust_physics::buoyancy::BuoyancyPlane;
    let mut w = PhysicsWorld::new();
    let mass = 500.0 * (4.0 / 3.0) * std::f64::consts::PI;
    let body = w.add_body(
        RigidBodyBuilder::dynamic()
            .shape(Shape::sphere(1.0))
            .mass(mass)
            .build(),
    );
    let water = BuoyancyPlane {
        point: Vec3::ZERO,
        up: Vec3::Y,
        density: 1_000.0,
        drag: 20.0,
        fluid_velocity: Vec3::ZERO,
    };
    let load = water.apply_sphere(&mut w, body).unwrap();
    assert!((load.displaced_volume - 2.0 * std::f64::consts::PI / 3.0).abs() < 1e-12);
    assert!(load.center.distance(Vec3::Y * -0.375) < 1e-12);
    w.step_fixed().unwrap();
    assert!(w.body(body).unwrap().position().length() < 1e-12);
}

#[test]
fn quadratic_drag_uses_projected_area_and_relative_wind() {
    use rust_physics::fields::Atmosphere;
    let mut w = PhysicsWorld::new();
    w.set_gravity(Vec3::ZERO).unwrap();
    let id = w.add_body(RigidBodyBuilder::dynamic().velocity(Vec3::X * 12.0).build());
    let atmosphere = Atmosphere {
        density: 1.2,
        drag_coefficient: 0.47,
        wind: Vec3::X * 2.0,
    };
    let force = atmosphere.apply(&mut w, id).unwrap();
    let expected = -0.5 * 1.2 * 0.47 * std::f64::consts::PI * 0.25 * 100.0;
    assert!(force.distance(Vec3::X * expected) < 1e-12);
    w.step_fixed().unwrap();
    assert!((w.body(id).unwrap().velocity().x - (12.0 + expected / 60.0)).abs() < 1e-12);
}

#[test]
fn mutual_gravity_preserves_momentum_and_tracks_two_body_orbit() {
    use rust_physics::fields::MutualGravity;
    let mut w = PhysicsWorld::new();
    w.set_settings(rust_physics::Settings {
        gravity: Vec3::ZERO,
        substeps: 1,
        ..Default::default()
    })
    .unwrap();
    let bodies = [-1.0, 1.0].map(|side| {
        w.add_body(
            RigidBodyBuilder::dynamic()
                .position(Vec3::X * side)
                .velocity(Vec3::Y * (0.5 * side))
                .shape(Shape::sphere(0.1))
                .build(),
        )
    });
    let field = MutualGravity {
        gravitational_constant: 1.0,
        softening: 0.0,
    };
    let dt = 1.0 / 240.0;
    for _ in 0..3_016 {
        field.apply(&mut w, &bodies).unwrap();
        w.step(dt).unwrap();
        let [a, b] = bodies.map(|id| w.body(id).unwrap());
        assert!((a.velocity() + b.velocity()).length() < 1e-11);
        assert!((a.position().distance(b.position()) - 2.0).abs() < 0.004);
    }
    assert!(w.body(bodies[0]).unwrap().position().distance(-Vec3::X) < 0.006);
}

#[test]
fn capsule_controller_climbs_low_step_and_stops_at_tall_wall() {
    use rust_physics::character::CapsuleController;
    let mut world = PhysicsWorld::new();
    world.add_body(RigidBodyBuilder::static_body().shape(Shape::Plane).build());
    let step = world.add_body(
        RigidBodyBuilder::static_body()
            .position(Vec3::new(1.5, 0.125, 0.0))
            .shape(Shape::cuboid(Vec3::new(0.65, 0.125, 1.0)))
            .build(),
    );
    let wall = world.add_body(
        RigidBodyBuilder::static_body()
            .position(Vec3::new(3.0, 1.0, 0.0))
            .shape(Shape::cuboid(Vec3::new(0.1, 1.0, 1.0)))
            .build(),
    );
    let controller = CapsuleController::default();
    let filter = QueryFilter::default();
    let start = Vec3::Y * 0.701;
    let up = controller
        .move_by(&world, start, Vec3::X * 1.6, filter)
        .unwrap();
    assert!(
        up.position.distance(Vec3::new(1.6, 0.951, 0.0)) < 0.005,
        "{up:?}"
    );
    assert_eq!(up.ground.unwrap().body, step);
    let blocked = controller
        .move_by(&world, up.position, Vec3::X * 2.0, filter)
        .unwrap();
    assert!(blocked.hits.iter().any(|hit| hit.body == wall));
    assert!(blocked.position.x < 2.61);
}
