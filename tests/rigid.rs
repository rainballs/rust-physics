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

#[test]
fn convex_ccd_detects_translating_box_and_rotating_tip() {
    let mut w = PhysicsWorld::new();
    w.set_settings(rust_physics::Settings {
        gravity: Vec3::ZERO,
        ccd_convex: true,
        ..Default::default()
    })
    .unwrap();
    let floor = w.add_body(RigidBodyBuilder::static_body().shape(Shape::Plane).build());
    let moving = w.add_body(
        RigidBodyBuilder::dynamic()
            .shape(Shape::cuboid(Vec3::new(0.2, 0.1, 0.3)))
            .position(Vec3::Y * 3.0)
            .velocity(-Vec3::Y * 700.0)
            .ccd(true)
            .build(),
    );
    let rotating = w.add_body(
        RigidBodyBuilder::dynamic()
            .shape(Shape::cuboid(Vec3::new(1.0, 0.05, 0.1)))
            .position(Vec3::new(5.0, 0.75, 0.0))
            .angular_velocity(Vec3::Z * 120.0)
            .ccd(true)
            .build(),
    );
    let stats = w.step_fixed().unwrap();
    assert!(stats.ccd_impacts >= 2);
    assert!(w.ccd_impacts().iter().any(|hit| hit.moving == moving
        && hit.target == floor
        && (hit.time - (3.0 - 0.1) / 700.0).abs() < 1e-7));
    assert!(
        w.ccd_impacts()
            .iter()
            .any(|hit| hit.moving == rotating && hit.target == floor && hit.time > 0.0)
    );
    let b = w.body(rotating).unwrap();
    assert!(b.position().y > 0.75);
    assert!(b.angular_velocity().z < 120.0);
}

#[test]
fn flat_box_ccd_uses_contact_feature_center_without_spurious_spin() {
    let mut w = PhysicsWorld::new();
    w.set_settings(rust_physics::Settings {
        gravity: Vec3::ZERO,
        ccd_convex: true,
        ..Default::default()
    })
    .unwrap();
    floor(&mut w);
    let id = w.add_body(
        RigidBodyBuilder::dynamic()
            .shape(Shape::cuboid(Vec3::new(0.5, 0.1, 0.5)))
            .position(Vec3::Y * 2.0)
            .velocity(-Vec3::Y * 300.0)
            .material(Material {
                restitution: 1.0,
                ..Default::default()
            })
            .ccd(true)
            .build(),
    );
    let stats = w.step_fixed().unwrap();
    assert_eq!(stats.ccd_impacts, 1);
    let b = w.body(id).unwrap();
    assert!(b.velocity().distance(Vec3::Y * 300.0) < 1e-10);
    assert_eq!(b.angular_velocity(), Vec3::ZERO);
}

#[test]
#[cfg(feature = "rayon")]
fn parallel_coupled_islands_match_serial_state() {
    let build = |parallel_islands| {
        let mut w = PhysicsWorld::new();
        w.set_settings(rust_physics::Settings {
            parallel_islands,
            sleep_time: 0.0,
            ..Default::default()
        })
        .unwrap();
        floor(&mut w);
        for x in 0..64 {
            for y in 0..6 {
                w.add_body(
                    RigidBodyBuilder::dynamic()
                        .shape(Shape::cuboid(Vec3::splat(0.4)))
                        .position(Vec3::new(f64::from(x) * 2.0, 0.4 + f64::from(y) * 0.8, 0.0))
                        .build(),
                );
            }
        }
        w
    };
    let (mut a, mut b) = (build(false), build(true));
    let mut parallel_work = 0;
    for _ in 0..40 {
        a.step_fixed().unwrap();
        parallel_work += b.step_fixed().unwrap().solved_parallel_islands;
        for ((_, a), (_, b)) in a.bodies().zip(b.bodies()) {
            assert_eq!(
                (
                    a.position(),
                    a.rotation(),
                    a.velocity(),
                    a.angular_momentum()
                ),
                (
                    b.position(),
                    b.rotation(),
                    b.velocity(),
                    b.angular_momentum()
                )
            );
        }
    }
    assert!(parallel_work > 0);
}

#[test]
fn profiled_step_has_same_physics_and_accounts_for_phases() {
    let mut a = PhysicsWorld::new();
    let mut b = PhysicsWorld::new();
    for world in [&mut a, &mut b] {
        floor(world);
        world.add_body(RigidBodyBuilder::dynamic().position(Vec3::Y * 0.5).build());
    }
    let plain = a.step_fixed().unwrap();
    let profile = b.step_fixed_profiled().unwrap();
    assert_eq!(plain.contact_points, profile.stats.contact_points);
    for ((_, a), (_, b)) in a.bodies().zip(b.bodies()) {
        assert_eq!((a.position(), a.velocity()), (b.position(), b.velocity()));
    }
    let t = profile.timings;
    assert!(
        t.total
            >= t.forces_and_joints
                + t.pose_and_ccd
                + t.broad_phase
                + t.narrow_phase
                + t.solver
                + t.sleep_and_events
    );
}

#[test]
fn deactivated_body_is_paused_excluded_and_reusable() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vec3::ZERO).unwrap();
    world.add_body(RigidBodyBuilder::static_body().sensor(true).build());
    let body = world.add_body(
        RigidBodyBuilder::dynamic()
            .position(Vec3::X * 0.1)
            .velocity(Vec3::Y)
            .build(),
    );
    world.step_fixed().unwrap();
    assert!(world.events().iter().any(|e| e.started && e.b == body));
    let position = world.body(body).unwrap().position();
    assert!(world.set_enabled(body, false));
    assert!(!world.apply_impulse(body, Vec3::X, position));
    world.step_fixed().unwrap();
    assert!(world.events().iter().any(|e| !e.started && e.b == body));
    let filter = QueryFilter {
        include_sensors: false,
        ..Default::default()
    };
    assert!(
        world
            .raycast(Vec3::Y * 2.0, -Vec3::Y, 4.0, filter)
            .is_none()
    );
    assert_eq!(world.body(body).unwrap().position(), position);
    assert!(world.set_enabled(body, true));
    world.step_fixed().unwrap();
    assert!(world.body(body).unwrap().position().y > position.y);
    assert!(world.events().iter().any(|e| e.started && e.b == body));
}

#[test]
fn torque_free_angular_acceleration_matches_local_velocity_change() {
    let mut world = PhysicsWorld::new();
    let id = world.add_body(
        RigidBodyBuilder::dynamic()
            .shape(Shape::cuboid(Vec3::new(0.5, 1.0, 1.5)))
            .angular_velocity(Vec3::new(1.0, -2.0, 0.7))
            .build(),
    );
    let b = world.body(id).unwrap();
    let initial = b.angular_velocity();
    let acceleration = b.acceleration(world.settings().gravity).angular;
    let dt = 1e-7;
    world.step(dt).unwrap();
    let derivative = (world.body(id).unwrap().angular_velocity() - initial) / dt;
    assert!(
        derivative.distance(acceleration) < 1e-6,
        "{derivative:?} {acceleration:?}"
    );
}

#[test]
fn sleeping_manifold_reuse_invalidates_when_support_moves() {
    let mut world = PhysicsWorld::new();
    let support = world.add_body(RigidBodyBuilder::static_body().shape(Shape::Plane).build());
    let sphere = world.add_body(RigidBodyBuilder::dynamic().position(Vec3::Y * 0.5).build());
    for _ in 0..120 {
        world.step_fixed().unwrap();
    }
    assert!(world.body(sphere).unwrap().is_sleeping());
    assert!(world.step_fixed().unwrap().reused_sleeping_manifolds > 0);
    world
        .body_mut(support)
        .unwrap()
        .set_pose(Vec3::Y * 0.2, Quat::IDENTITY);
    let stats = world.step_fixed().unwrap();
    assert_eq!(stats.reused_sleeping_manifolds, 0);
    assert!(!world.body(sphere).unwrap().is_sleeping());
    assert!(world.body(sphere).unwrap().position().y > 0.6);
}
