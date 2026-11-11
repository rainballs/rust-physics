use rust_physics::{Joint, JointKind, Motor, PhysicsWorld, Quat, RigidBodyBuilder, Vec3};

#[test]
fn distance_pendulum_tracks_small_angle_period_and_length() {
    let mut w = PhysicsWorld::new();
    let a = w.add_body(RigidBodyBuilder::static_body().filter(1, 0).build());
    let angle = 0.12_f64;
    let b = w.add_body(
        RigidBodyBuilder::dynamic()
            .position(Vec3::new(angle.sin(), -angle.cos(), 0.0))
            .build(),
    );
    w.add_joint(Joint::new(
        [a, b],
        [Vec3::ZERO; 2],
        JointKind::Distance { length: 1.0 },
    ))
    .unwrap();
    let mut crossing = None;
    for frame in 1..120 {
        w.step_fixed().unwrap();
        let position = w.body(b).unwrap().position();
        assert!((position.length() - 1.0).abs() < 0.003);
        if position.x < 0.0 && crossing.is_none() {
            crossing = Some(frame as f64 / 60.0);
        }
    }
    let expected = std::f64::consts::FRAC_PI_2 / 9.81_f64.sqrt();
    assert!((crossing.unwrap() - expected).abs() < 0.025);
}

#[test]
fn rope_allows_slack_then_stops_extension() {
    let mut w = PhysicsWorld::new();
    w.set_gravity(Vec3::ZERO).unwrap();
    let a = w.add_body(RigidBodyBuilder::static_body().filter(1, 0).build());
    let b = w.add_body(
        RigidBodyBuilder::dynamic()
            .position(Vec3::X * 0.5)
            .velocity(Vec3::X)
            .build(),
    );
    w.add_joint(Joint::new(
        [a, b],
        [Vec3::ZERO; 2],
        JointKind::Rope { max_length: 2.0 },
    ))
    .unwrap();
    for _ in 0..30 {
        w.step_fixed().unwrap();
    }
    assert!((w.body(b).unwrap().position().x - 1.0).abs() < 1e-12);
    for _ in 0..120 {
        w.step_fixed().unwrap();
    }
    assert!((w.body(b).unwrap().position().x - 2.0).abs() < 1e-12);
}

#[test]
fn slider_motor_respects_limits_and_removing_body_removes_joint() {
    let mut w = PhysicsWorld::new();
    let a = w.add_body(RigidBodyBuilder::static_body().filter(1, 0).build());
    let b = w.add_body(RigidBodyBuilder::dynamic().position(Vec3::Y * 0.25).build());
    let joint = w
        .add_joint(Joint::new(
            [a, b],
            [Vec3::ZERO; 2],
            JointKind::Prismatic {
                axis: Vec3::X,
                rotation: Quat::IDENTITY,
                limits: Some([-1.0, 1.0]),
                motor: Some(Motor {
                    speed: 3.0,
                    max_effort: 100.0,
                }),
            },
        ))
        .unwrap();
    for _ in 0..120 {
        w.step_fixed().unwrap();
    }
    assert!(w.body(b).unwrap().position().distance(Vec3::X) < 0.004);
    w.remove_body(a).unwrap();
    assert!(w.joint(joint).is_none());
}

#[test]
fn hinge_motor_rotates_around_axis_and_corrects_tilt() {
    let mut w = PhysicsWorld::new();
    w.set_gravity(Vec3::ZERO).unwrap();
    let a = w.add_body(RigidBodyBuilder::static_body().filter(1, 0).build());
    let b = w.add_body(
        RigidBodyBuilder::dynamic()
            .rotation(Quat::from_rotation_x(0.3))
            .build(),
    );
    w.add_joint(Joint::new(
        [a, b],
        [Vec3::ZERO; 2],
        JointKind::Revolute {
            axes: [Vec3::Y; 2],
            limits: None,
            motor: Some(Motor {
                speed: 2.0,
                max_effort: 5.0,
            }),
        },
    ))
    .unwrap();
    for _ in 0..60 {
        w.step_fixed().unwrap();
    }
    let b = w.body(b).unwrap();
    assert!((b.rotation() * Vec3::Y).distance(Vec3::Y) < 1e-6);
    assert!(b.angular_velocity().distance(2.0 * Vec3::Y) < 1e-6);
}

#[test]
fn implicit_damped_spring_settles_at_static_equilibrium() {
    let mut w = PhysicsWorld::new();
    let a = w.add_body(RigidBodyBuilder::static_body().filter(1, 0).build());
    let b = w.add_body(RigidBodyBuilder::dynamic().position(-Vec3::Y).build());
    w.add_joint(Joint::new(
        [a, b],
        [Vec3::ZERO; 2],
        JointKind::Spring {
            length: 1.0,
            stiffness: 80.0,
            damping: 12.0,
        },
    ))
    .unwrap();
    for _ in 0..180 {
        w.step_fixed().unwrap();
    }
    assert!((w.body(b).unwrap().position().y + 1.0 + 9.81 / 80.0).abs() < 1e-6);
}

#[test]
fn fixed_joint_restores_pose() {
    let mut w = PhysicsWorld::new();
    let a = w.add_body(RigidBodyBuilder::static_body().filter(1, 0).build());
    let b = w.add_body(
        RigidBodyBuilder::dynamic()
            .position(Vec3::X)
            .rotation(Quat::from_rotation_z(0.4))
            .build(),
    );
    w.add_joint(Joint::new(
        [a, b],
        [Vec3::X * 0.8, Vec3::ZERO],
        JointKind::Fixed {
            rotation: Quat::IDENTITY,
        },
    ))
    .unwrap();
    for _ in 0..120 {
        w.step_fixed().unwrap();
    }
    let b = w.body(b).unwrap();
    assert!(b.position().distance(Vec3::X * 0.8) < 0.004);
    assert!(b.rotation().angle_between(Quat::IDENTITY) < 1e-6);
}

#[test]
fn motor_stalls_at_angular_hinge_limit() {
    let mut w = PhysicsWorld::new();
    w.set_gravity(Vec3::ZERO).unwrap();
    let a = w.add_body(RigidBodyBuilder::static_body().filter(1, 0).build());
    let b = w.add_body(RigidBodyBuilder::dynamic().build());
    w.add_joint(Joint::new(
        [a, b],
        [Vec3::ZERO; 2],
        JointKind::Revolute {
            axes: [Vec3::Y; 2],
            motor: Some(Motor {
                speed: 2.0,
                max_effort: 5.0,
            }),
            limits: Some(rust_physics::HingeLimits {
                tangents: [Vec3::X; 2],
                angles: [-0.3, 0.4],
            }),
        },
    ))
    .unwrap();
    for _ in 0..100 {
        w.step_fixed().unwrap();
    }
    let b = w.body(b).unwrap();
    assert!(b.angular_velocity().length() < 1e-8);
    assert!(b.rotation().angle_between(Quat::from_rotation_y(0.4)) < 1e-6);
}

#[test]
fn supported_passive_joint_island_sleeps_and_wakes_as_a_group() {
    let mut w = PhysicsWorld::new();
    let a = w.add_body(RigidBodyBuilder::static_body().filter(1, 0).build());
    let b = w.add_body(
        RigidBodyBuilder::dynamic()
            .position(-Vec3::Y)
            .filter(1, 0)
            .build(),
    );
    let c = w.add_body(
        RigidBodyBuilder::dynamic()
            .position(-Vec3::Y * 2.0)
            .filter(1, 0)
            .build(),
    );
    for pair in [[a, b], [b, c]] {
        w.add_joint(Joint::new(
            pair,
            [Vec3::ZERO; 2],
            JointKind::Distance { length: 1.0 },
        ))
        .unwrap();
    }
    for _ in 0..180 {
        w.step_fixed().unwrap();
    }
    assert!(w.body(b).unwrap().is_sleeping() && w.body(c).unwrap().is_sleeping());
    let tip = w.body(c).unwrap().position();
    assert!(w.apply_impulse(c, Vec3::X * 0.4, tip));
    w.step_fixed().unwrap();
    assert!(!w.body(b).unwrap().is_sleeping() && !w.body(c).unwrap().is_sleeping());
    assert!(
        w.body(b)
            .unwrap()
            .position()
            .distance(w.body(c).unwrap().position())
            < 1.002
    );
}

#[test]
fn breakable_joint_reports_solved_reaction_and_releases_load() {
    let mut w = PhysicsWorld::new();
    let anchor = w.add_body(RigidBodyBuilder::static_body().filter(1, 0).build());
    let load = w.add_body(RigidBodyBuilder::dynamic().position(-Vec3::Y).build());
    let limit = rust_physics::JointBreakLimit {
        force: Some(5.0),
        torque: None,
    };
    let joint = w
        .add_joint(
            Joint::new(
                [anchor, load],
                [Vec3::ZERO; 2],
                JointKind::Distance { length: 1.0 },
            )
            .breakable(limit),
        )
        .unwrap();
    let stats = w.step_fixed().unwrap();
    assert_eq!(stats.broken_joints, 1);
    let event = w.joint_break_events()[0];
    assert_eq!(event.joint, joint);
    assert_eq!(event.bodies, [anchor, load]);
    assert!((event.force - 9.81).abs() < 1e-12);
    assert!(w.joint(joint).is_none());
    assert!(w.body(load).unwrap().position().y < -1.0);
}
