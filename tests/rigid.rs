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

#[test]
fn resting_island_wakes_when_support_is_removed() {
    let mut world = PhysicsWorld::new();
    let floor = world.add_body(RigidBodyBuilder::static_body().shape(Shape::Plane).build());
    let bottom = world.add_body(RigidBodyBuilder::dynamic().position(Vec3::Y * 0.5).build());
    let top = world.add_body(RigidBodyBuilder::dynamic().position(Vec3::Y * 1.5).build());
    for _ in 0..120 {
        world.step_fixed().unwrap();
    }
    assert!(world.body(bottom).unwrap().is_sleeping());
    assert!(world.body(top).unwrap().is_sleeping());
    world.remove_body(floor).unwrap();
    for _ in 0..15 {
        world.step_fixed().unwrap();
    }
    assert!(world.body(bottom).unwrap().position().y < 0.3);
    assert!(world.body(top).unwrap().position().y < 1.3);
}

#[test]
fn penetration_correction_recovers_a_sphere_inside_a_box() {
    let mut w = PhysicsWorld::new();
    w.set_gravity(Vec3::ZERO).unwrap();
    w.add_body(
        RigidBodyBuilder::static_body()
            .shape(Shape::cuboid(Vec3::new(1.0, 2.0, 3.0)))
            .build(),
    );
    let ball = w.add_body(RigidBodyBuilder::dynamic().build());
    for _ in 0..40 {
        w.step_fixed().unwrap();
    }
    let b = w.body(ball).unwrap();
    assert!((b.position().x - 1.499).abs() < 1e-10);
    assert_eq!(b.velocity(), Vec3::ZERO);
}

#[test]
fn friction_holds_a_box_on_a_shallow_slope_and_slows_sliding() {
    let mut w = PhysicsWorld::new();
    let angle = 0.15;
    let q = Quat::from_rotation_z(angle);
    w.add_body(
        RigidBodyBuilder::static_body()
            .rotation(q)
            .shape(Shape::Plane)
            .build(),
    );
    let b = w.add_body(
        RigidBodyBuilder::dynamic()
            .position(q * (Vec3::Y * 0.5))
            .rotation(q)
            .shape(Shape::cuboid(Vec3::splat(0.5)))
            .build(),
    );
    for _ in 0..120 {
        w.step_fixed().unwrap();
    }
    assert!((q.conjugate() * w.body(b).unwrap().position()).x.abs() < 0.04);
    w.body_mut(b).unwrap().set_velocity(q * (-Vec3::X * 3.0));
    for _ in 0..15 {
        w.step_fixed().unwrap();
    }
    let b = w.body(b).unwrap();
    assert!(b.velocity().dot(q * -Vec3::X) < 2.8);
}

#[test]
fn synchronized_ccd_resolves_moving_spheres_at_relative_toi() {
    for enabled in [true, false] {
        let mut w = PhysicsWorld::new();
        w.set_settings(rust_physics::Settings {
            gravity: Vec3::ZERO,
            ccd_moving_spheres: enabled,
            ..Default::default()
        })
        .unwrap();
        let material = Material {
            restitution: 1.0,
            ..Default::default()
        };
        let a = w.add_body(
            RigidBodyBuilder::dynamic()
                .position(-Vec3::X * 3.0)
                .velocity(Vec3::X * 600.0)
                .material(material)
                .ccd(true)
                .build(),
        );
        let b = w.add_body(
            RigidBodyBuilder::dynamic()
                .position(Vec3::X * 3.0)
                .velocity(-Vec3::X * 400.0)
                .material(material)
                .build(),
        );
        let stats = w.step_fixed().unwrap();
        if enabled {
            assert_eq!(stats.ccd_impacts, 1);
            let hit = w.ccd_impacts()[0];
            assert_eq!((hit.moving, hit.target), (a, b));
            assert!((hit.time - 0.005).abs() < 1e-12);
            assert_eq!(hit.normal, -Vec3::X);
            assert!(w.body(a).unwrap().velocity().distance(-Vec3::X * 400.0) < 1e-9);
            assert!(w.body(b).unwrap().velocity().distance(Vec3::X * 600.0) < 1e-9);
        } else {
            assert_eq!(stats.ccd_impacts, 0);
            assert!(w.body(a).unwrap().position().x > w.body(b).unwrap().position().x);
        }
    }
}

#[test]
fn planar_axis_locks_apply_to_forces_impulses_and_contact_correction() {
    let mut w = PhysicsWorld::new();
    floor(&mut w);
    let id = w.add_body(
        RigidBodyBuilder::dynamic()
            .planar_xy()
            .position(Vec3::Y * 2.0)
            .velocity(Vec3::Z * 3.0)
            .angular_velocity(Vec3::new(2.0, 1.0, 3.0))
            .build(),
    );
    assert!(w.apply_impulse(id, Vec3::Z, Vec3::Y * 2.0 + Vec3::X));
    for _ in 0..90 {
        w.add_force_at_point(id, Vec3::Z * 10.0, Vec3::ZERO);
        w.step_fixed().unwrap();
    }
    let body = w.body(id).unwrap();
    assert_eq!(body.position().z, 0.0);
    assert_eq!(body.velocity().z, 0.0);
    assert_eq!(body.angular_velocity().x, 0.0);
    assert_eq!(body.angular_velocity().y, 0.0);
    assert!((body.position().y - 0.5).abs() < 0.003);
}

#[test]
fn angular_lock_uses_restricted_inertia_for_a_rotated_cuboid() {
    let q = Quat::from_rotation_y(0.6);
    let builder = RigidBodyBuilder::dynamic()
        .shape(Shape::cuboid(Vec3::new(0.5, 0.7, 1.1)))
        .rotation(q);
    let free = builder.clone().build();
    let restricted = builder
        .rotation_axes([false, false, true])
        .angular_velocity(Vec3::Z)
        .build();
    let expected = free.world_inverse_inertia().inverse().z_axis.z;
    assert!((restricted.angular_momentum().z - expected).abs() < 1e-12);
    assert!(restricted.angular_velocity().distance(Vec3::Z) < 1e-12);
    assert_eq!(restricted.world_inverse_inertia().x_axis, Vec3::ZERO);
}

#[test]
fn synchronized_ccd_budget_cannot_pause_unrelated_falling_bodies() {
    let mut w = PhysicsWorld::new();
    w.set_settings(rust_physics::Settings {
        ccd_moving_spheres: true,
        ..Default::default()
    })
    .unwrap();
    // Two closely spaced finite walls create more than 32 impacts in one substep.
    for x in [-0.12, 0.12] {
        w.add_body(
            RigidBodyBuilder::static_body()
                .shape(Shape::cuboid(Vec3::new(0.01, 1.0, 1.0)))
                .position(Vec3::X * x)
                .material(Material {
                    restitution: 1.0,
                    ..Default::default()
                })
                .build(),
        );
    }
    w.add_body(
        RigidBodyBuilder::dynamic()
            .shape(Shape::sphere(0.1))
            .gravity_scale(0.0)
            .velocity(Vec3::X * 1_000.0)
            .ccd(true)
            .build(),
    );
    let falling = w.add_body(RigidBodyBuilder::dynamic().position(Vec3::Y * 5.0).build());
    let stats = w.step_fixed().unwrap();
    assert!(stats.ccd_budget_fallbacks > 0);
    let y = w.body(falling).unwrap().position().y;
    let expected = 5.0 - 9.81 * 3.0 / 120.0_f64.powi(2);
    assert!((y - expected).abs() < 1e-11);
}

#[test]
fn non_finite_external_impulse_is_rejected_without_changing_state() {
    let mut w = PhysicsWorld::new();
    let id = w.add_body(RigidBodyBuilder::dynamic().mass(0.01).build());
    assert!(!w.apply_impulse(id, Vec3::X * f64::MAX, Vec3::ZERO));
    assert_eq!(w.body(id).unwrap().velocity(), Vec3::ZERO);
    assert_eq!(w.body(id).unwrap().angular_momentum(), Vec3::ZERO);
}

#[test]
fn origin_rebase_keeps_relative_geometry_velocities_and_handles() {
    let mut world = PhysicsWorld::new();
    let offset = Vec3::new(1e9, -1e9, 1e9);
    world.add_body(
        RigidBodyBuilder::static_body()
            .shape(Shape::Plane)
            .position(offset)
            .build(),
    );
    let id = world.add_body(
        RigidBodyBuilder::dynamic()
            .position(offset + Vec3::Y * 2.0)
            .velocity(Vec3::X)
            .build(),
    );
    assert!(world.rebase_origin(offset));
    assert_eq!(world.body(id).unwrap().position(), Vec3::Y * 2.0);
    assert_eq!(world.body(id).unwrap().velocity(), Vec3::X);
    assert!(!world.rebase_origin(Vec3::splat(f64::NAN)));
    for _ in 0..120 {
        world.step_fixed().unwrap();
    }
    assert!((world.body(id).unwrap().position().y - 0.5).abs() < 0.003);
}
