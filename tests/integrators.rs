use rust_physics::Vec3;
use rust_physics::integrators::{AdaptiveError, AdaptiveSettings, Method, State, adaptive_rk4};

#[test]
fn verlet_and_rk4_integrate_constant_acceleration_exactly() {
    let initial = State {
        position: Vec3::new(1.0, 4.0, 3.0),
        velocity: Vec3::new(3.0, 1.0, 2.0),
    };
    let g = Vec3::Y * -9.81;
    let dt = 0.73;
    for method in [Method::VelocityVerlet, Method::RungeKutta4] {
        let step = method.advance(initial, dt, |_| g);
        let p = initial.position + initial.velocity * dt + g * (0.5 * dt * dt);
        let v = initial.velocity + g * dt;
        assert!(step.position.distance(p) < 1e-12);
        assert!(step.velocity.distance(v) < 1e-12);
    }
}

#[test]
fn adaptive_rk4_controls_local_error_in_an_oscillator() {
    let initial = State {
        position: Vec3::X,
        velocity: Vec3::ZERO,
    };
    let duration = 12.7_f64;
    let settings = AdaptiveSettings {
        max_step: 1.0,
        ..Default::default()
    };
    let result = adaptive_rk4(initial, duration, settings, |x| -x).unwrap();
    assert!(result.rejected > 0 && result.accepted > 20);
    assert!(result.state.position.distance(Vec3::X * duration.cos()) < 1e-6);
    assert!(result.state.velocity.distance(-Vec3::X * duration.sin()) < 1e-6);
    assert_eq!(
        adaptive_rk4(
            initial,
            duration,
            AdaptiveSettings {
                max_attempts: 1,
                ..settings
            },
            |x| -x
        )
        .unwrap_err(),
        AdaptiveError::AttemptLimit
    );
}

#[test]
fn position_verlet_preserves_bounded_oscillator_energy() {
    use rust_physics::integrators::PositionVerlet;
    let initial = State {
        position: Vec3::X,
        velocity: Vec3::ZERO,
    };
    let mut verlet = PositionVerlet::new(initial, 0.02, -Vec3::X).unwrap();
    let mut euler = initial;
    for _ in 0..5000 {
        let position = verlet.position();
        let velocity = verlet.advance(|p| -p).unwrap();
        let energy = 0.5 * (position.length_squared() + velocity.length_squared());
        assert!((energy - 0.5).abs() < 0.0001);
        euler = Method::Euler.advance(euler, 0.02, |p| -p);
    }
    assert!(0.5 * (euler.position.length_squared() + euler.velocity.length_squared()) > 2.0);
}

#[test]
fn general_rk4_integrates_velocity_dependent_linear_drag() {
    use rust_physics::integrators::rk4;
    let mut state = State {
        position: Vec3::ZERO,
        velocity: Vec3::X * 3.0,
    };
    for i in 0..100 {
        state = rk4(state, f64::from(i) * 0.01, 0.01, |_, state| {
            -state.velocity * 2.0
        });
    }
    let expected = (-2.0_f64).exp();
    assert!(state.velocity.distance(Vec3::X * (3.0 * expected)) < 2e-9);
    assert!(state.position.distance(Vec3::X * (1.5 * (1.0 - expected))) < 1e-9);
}
