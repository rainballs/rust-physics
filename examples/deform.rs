use rust_physics::softbody::{SoftBody, SoftSettings};
use rust_physics::{PhysicsWorld, QueryFilter, RigidBodyBuilder, Shape, Vec3};

fn main() {
    let mut world = PhysicsWorld::new();
    world.add_body(RigidBodyBuilder::static_body().shape(Shape::Plane).build());
    let mut tetra = SoftBody::default();
    for p in [Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::Z] {
        tetra.push_node(p + Vec3::Y * 2.0, 1.0).unwrap();
    }
    for a in 0..4 {
        for b in a + 1..4 {
            assert!(tetra.add_distance([a, b], 0.0001));
        }
    }
    assert!(tetra.add_volume([0, 1, 2, 3], 0.00001));
    for frame in 0..360 {
        tetra.step(
            1.0 / 120.0,
            SoftSettings::default(),
            Some((&world, QueryFilter::default())),
        );
        if frame % 60 == 59 {
            let points: Vec<_> = tetra.positions().collect();
            let [a, b, c, d] = [points[0], points[1], points[2], points[3]];
            let volume = (b - a).dot((c - a).cross(d - a)) / 6.0;
            let lowest = points.iter().map(|p| p.y).fold(f64::INFINITY, f64::min);
            println!(
                "{:.1} s  volume={volume:.5}  lowest y={lowest:.4}",
                (frame + 1) as f64 / 120.0
            );
        }
    }
}
