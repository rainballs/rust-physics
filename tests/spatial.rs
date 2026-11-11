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

#[test]
fn both_broad_phases_produce_identical_world_states() {
    use rust_physics::BroadPhase;
    let build = |broad_phase| {
        let mut w = PhysicsWorld::new();
        let settings = rust_physics::Settings {
            broad_phase,
            ..Default::default()
        };
        w.set_settings(settings).unwrap();
        w.add_body(RigidBodyBuilder::static_body().shape(Shape::Plane).build());
        for i in 0..36 {
            let p = Vec3::new(f64::from(i % 6), 0.7 + f64::from(i / 6) * 0.75, 0.0);
            let shape = if i % 2 == 0 {
                Shape::sphere(0.45)
            } else {
                Shape::cuboid(Vec3::splat(0.42))
            };
            w.add_body(
                RigidBodyBuilder::dynamic()
                    .position(p)
                    .shape(shape)
                    .rotation(Quat::from_rotation_x(f64::from(i) * 0.37))
                    .build(),
            );
        }
        w
    };
    let (mut a, mut b) = (build(BroadPhase::SweepAndPrune), build(BroadPhase::Bvh));
    for _ in 0..160 {
        let sa = a.step_fixed().unwrap();
        let sb = b.step_fixed().unwrap();
        assert_eq!(
            (sa.candidate_pairs, sa.contact_points),
            (sb.candidate_pairs, sb.contact_points)
        );
        for ((_, a), (_, b)) in a.bodies().zip(b.bodies()) {
            assert_eq!(a.position(), b.position());
            assert_eq!(a.rotation(), b.rotation());
            assert_eq!(a.velocity(), b.velocity());
        }
    }
}

#[test]
fn indexed_queries_match_direct_queries() {
    use rust_physics::{Ray, spatial::Aabb};
    let mut w = PhysicsWorld::new();
    w.add_body(RigidBodyBuilder::static_body().shape(Shape::Plane).build());
    for i in -10..=10 {
        w.add_body(
            RigidBodyBuilder::static_body()
                .position(Vec3::new(f64::from(i), 1.0, 0.0))
                .build(),
        );
    }
    let index = w.spatial_index();
    let filter = QueryFilter::default();
    let rays: Vec<_> = (-50..=50)
        .map(|i| Ray {
            origin: Vec3::new(f64::from(i) * 0.25, 10.0, 0.0),
            direction: -Vec3::Y,
            max_distance: 10.0,
        })
        .collect();
    for (a, b) in index
        .raycast_batch(&rays, filter)
        .zip(w.raycast_batch(&rays, filter))
    {
        assert_eq!(
            a.map(|hit| (hit.body, hit.distance)),
            b.map(|hit| (hit.body, hit.distance))
        );
    }
    let bounds = Aabb {
        min: Vec3::new(-2.0, 0.0, -2.0),
        max: Vec3::new(2.0, 2.0, 2.0),
    };
    let mut accelerated = Vec::new();
    index.query_aabb(bounds, filter, |id| accelerated.push(id));
    accelerated.sort_unstable();
    assert_eq!(
        accelerated,
        w.query_aabb(bounds, filter).collect::<Vec<_>>()
    );
}

#[test]
fn capsule_rays_and_sphere_casts_hit_side_and_endcap() {
    let mut w = PhysicsWorld::new();
    let capsule = w.add_body(
        RigidBodyBuilder::static_body()
            .shape(Shape::capsule(0.5, 1.0))
            .build(),
    );
    let filter = QueryFilter::default();
    for (origin, direction, expected, normal) in [
        (Vec3::X * 3.0, -Vec3::X, 2.5, Vec3::X),
        (Vec3::Y * 3.0, -Vec3::Y, 1.5, Vec3::Y),
        (Vec3::ZERO, Vec3::X, 0.5, Vec3::X),
    ] {
        let hit = w.raycast(origin, direction, 5.0, filter).unwrap();
        assert_eq!(hit.body, capsule);
        assert!((hit.distance - expected).abs() < 1e-12);
        assert!(hit.normal.distance(normal) < 1e-12);
    }
    let hit = w
        .cast_sphere(Vec3::Y * 3.0, 0.25, -Vec3::Y, 5.0, filter)
        .unwrap();
    assert!((hit.distance - 1.25).abs() < 1e-12);
    assert_eq!(hit.position, Vec3::Y * 1.5);
}

#[test]
fn capsule_contact_pairings_report_expected_depth_and_orientation() {
    use rust_physics::collision::contacts;
    let collider = |shape, position, rotation| {
        RigidBodyBuilder::static_body()
            .shape(shape)
            .position(position)
            .rotation(rotation)
            .build()
    };
    let box_body = collider(Shape::cuboid(Vec3::ONE), Vec3::ZERO, Quat::IDENTITY);
    let ball = collider(
        Shape::sphere(0.5),
        Vec3::new(1.25, 0.0, 0.0),
        Quat::IDENTITY,
    );
    let floor = collider(Shape::Plane, Vec3::Y * -0.25, Quat::IDENTITY);
    let capsule = collider(Shape::capsule(0.5, 0.5), Vec3::X * 1.75, Quat::IDENTITY);
    let other = collider(
        Shape::capsule(0.5, 0.6),
        Vec3::X * 0.5,
        Quat::from_rotation_z(0.2),
    );
    let inner = collider(Shape::capsule(0.5, 0.4), Vec3::ZERO, Quat::IDENTITY);
    let side = collider(Shape::capsule(0.5, 0.7), Vec3::X * 1.25, Quat::IDENTITY);
    let horizontal = collider(
        Shape::capsule(0.5, 0.7),
        Vec3::ZERO,
        Quat::from_rotation_z(std::f64::consts::FRAC_PI_2),
    );
    let mut output = Vec::new();
    contacts(&box_body, &capsule, &mut output);
    assert!(output.is_empty());
    contacts(&box_body, &side, &mut output);
    assert_eq!(output.len(), 1);
    assert!((output[0].depth - 0.25).abs() < 1e-12);
    assert_eq!(output[0].normal, Vec3::X);
    output.clear();
    contacts(&side, &box_body, &mut output);
    assert_eq!(output[0].normal, -Vec3::X);
    output.clear();
    contacts(&box_body, &inner, &mut output);
    assert!((output[0].depth - 1.5).abs() < 1e-12);
    output.clear();
    contacts(&ball, &capsule, &mut output);
    assert!((output[0].depth - 0.5).abs() < 1e-12);
    output.clear();
    contacts(&other, &side, &mut output);
    assert!(output[0].depth > 0.25 && output[0].normal.x > 0.99);
    output.clear();
    contacts(&floor, &horizontal, &mut output);
    assert_eq!(output.len(), 2);
    assert!(output.iter().all(|c| (c.depth - 0.25).abs() < 1e-12));
}

#[test]
fn cylinder_ray_and_sphere_cast_distinguish_cap_rim_and_side() {
    let mut w = PhysicsWorld::new();
    let cylinder = w.add_body(
        RigidBodyBuilder::static_body()
            .shape(Shape::cylinder(1.0, 0.5))
            .build(),
    );
    let filter = QueryFilter::default();
    for (origin, direction, distance, normal) in [
        (Vec3::Y * 3.0, -Vec3::Y, 2.5, Vec3::Y),
        (Vec3::X * 3.0, -Vec3::X, 2.0, Vec3::X),
        (Vec3::ZERO, Vec3::Z, 1.0, Vec3::Z),
    ] {
        let hit = w.raycast(origin, direction, 5.0, filter).unwrap();
        assert_eq!(hit.body, cylinder);
        assert!((hit.distance - distance).abs() < 1e-12);
        assert!(hit.normal.distance(normal) < 1e-12);
    }
    let hit = w
        .cast_sphere(Vec3::new(3.0, 0.9, 0.0), 0.5, -Vec3::X, 5.0, filter)
        .unwrap();
    assert!((hit.distance - 1.7).abs() < 1e-7);
    assert!(hit.position.distance(Vec3::new(1.0, 0.5, 0.0)) < 1e-7);
}
