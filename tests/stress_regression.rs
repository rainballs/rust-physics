#[path = "../examples/stress_scene/mod.rs"]
mod stress_scene;

#[test]
fn seeded_convex_contact_regression_finishes_with_finite_state() {
    let (contacts, _) = stress_scene::soak(54, 350, false);
    assert!(contacts > 20_000);
}
