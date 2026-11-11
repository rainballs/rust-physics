use rust_physics::timestep::FixedClock;

#[test]
fn clock_preserves_fractional_time_across_frames_and_bounds_catch_up() {
    let mut clock = FixedClock::new(0.01, 3).unwrap();
    let first = clock.plan(0.005).unwrap();
    assert_eq!((first.steps, first.interpolation), (0, 0.5));
    assert!(clock.plan(f64::NAN).is_none());
    let second = clock.plan(0.005).unwrap();
    assert_eq!((second.steps, second.interpolation), (1, 0.0));
    let burst = clock.plan(0.076).unwrap();
    assert_eq!(burst.steps, 3);
    assert!((burst.interpolation - 0.6).abs() < 1e-12);
    assert!((burst.dropped_time - 0.04).abs() < 1e-12);
    clock.reset();
    assert_eq!(clock.plan(0.0).unwrap().interpolation, 0.0);
}
