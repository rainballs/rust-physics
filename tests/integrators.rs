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
