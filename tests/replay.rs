use rust_physics::{
    BroadPhase, Joint, JointKind, PhysicsWorld, RigidBodyBuilder, Settings, Shape, Vec3,
};

#[test]
fn warm_started_contacts_and_joints_replay_exactly_from_snapshot() {
    let mut world = PhysicsWorld::new();
    world
        .set_settings(Settings {
            broad_phase: BroadPhase::DynamicTree,
            ..Default::default()
        })
        .unwrap();
    world.add_body(RigidBodyBuilder::static_body().shape(Shape::Plane).build());
    let a = world.add_body(
        RigidBodyBuilder::dynamic()
            .shape(Shape::cuboid(Vec3::splat(0.5)))
            .position(Vec3::Y * 0.5)
            .build(),
    );
    let b = world.add_body(
        RigidBodyBuilder::dynamic()
            .position(Vec3::new(2.0, 0.5, 0.0))
            .build(),
    );
    world
        .add_joint(Joint::new(
            [a, b],
            [Vec3::ZERO; 2],
            JointKind::Distance { length: 2.0 },
        ))
        .unwrap();
    for _ in 0..22 {
        world.step_fixed().unwrap();
    }
    world.add_force_at_point(b, Vec3::Z * 1.1, world.body(b).unwrap().position());
    let checkpoint = world.snapshot();
    let replay = |world: &mut PhysicsWorld| {
        (0..40)
            .map(|frame| {
                if frame % 7 == 0 {
                    world.apply_impulse(
                        a,
                        Vec3::Z * 0.07,
                        world.body(a).unwrap().position() + Vec3::Y * 0.2,
                    );
                }
                let stats = world.step_fixed().unwrap();
                let state = world
                    .bodies()
                    .map(|(id, b)| {
                        (
                            id,
                            b.position(),
                            b.rotation(),
                            b.velocity(),
                            b.angular_momentum(),
                            b.is_sleeping(),
                        )
                    })
                    .collect::<Vec<_>>();
                (stats.contact_points, world.events().to_vec(), state)
            })
            .collect::<Vec<_>>()
    };
    let first = replay(&mut world);
    assert!(world.restore(&checkpoint));
    assert_eq!(first, replay(&mut world));
}

#[test]
fn snapshot_restore_rejects_foreign_world_and_retires_speculative_handles() {
    let mut world = PhysicsWorld::new();
    let mut foreign = PhysicsWorld::new();
    let checkpoint = world.snapshot();
    assert!(!foreign.restore(&checkpoint));
    let speculative = world.add_body(RigidBodyBuilder::dynamic().build());
    assert!(world.restore(&checkpoint));
    assert!(world.body(speculative).is_none());
    let replacement = world.add_body(RigidBodyBuilder::dynamic().build());
    assert_ne!(replacement, speculative);
}

#[test]
fn checked_step_rolls_back_after_nonfinite_force_integration() {
    use rust_physics::{NumericalStage, StepError};
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vec3::X * f64::MAX).unwrap();
    let id = world.add_body(RigidBodyBuilder::dynamic().build());
    let result = world.step_checked(4.0);
    assert!(matches!(
        result,
        Err(StepError::NumericalFailure(NumericalStage::Forces))
    ));
    let body = world.body(id).unwrap();
    assert_eq!(body.position(), Vec3::ZERO);
    assert_eq!(body.velocity(), Vec3::ZERO);
    world.set_gravity(-Vec3::Y * 9.81).unwrap();
    assert!(world.step_checked(1.0 / 60.0).is_ok());
    assert!(world.body(id).unwrap().position().y < 0.0);
}
