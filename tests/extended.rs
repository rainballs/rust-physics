use rust_physics::fluid::{Fluid, FluidSettings};
use rust_physics::grid::SpatialHash;
use rust_physics::particles::{Emitter, Particle, ParticleSettings, Particles};
use rust_physics::softbody::{SoftBody, SoftSettings};
use rust_physics::{PhysicsWorld, QueryFilter, RigidBodyBuilder, Shape, Vec3};

#[test]
fn spatial_hash_matches_brute_force_across_signed_cell_boundaries() {
    let points: Vec<_> = (-10..10)
        .flat_map(|x| (-10..10).map(move |z| Vec3::new(x as f64 * 0.24, 0.0, z as f64 * 0.31)))
        .collect();
    let mut grid = SpatialHash::new(0.5).unwrap();
    for _ in 0..3 {
        grid.rebuild(&points);
        for &origin in &points {
            let inside = |p: Vec3| p.distance_squared(origin) < 0.25;
            let mut from_grid: Vec<_> = grid
                .neighbors(origin)
                .filter(|&i| inside(points[i]))
                .collect();
            from_grid.sort_unstable();
            let brute: Vec<_> = points
                .iter()
                .enumerate()
                .filter_map(|(i, &p)| inside(p).then_some(i))
                .collect();
            assert_eq!(from_grid, brute);
        }
    }
}

#[test]
fn deterministic_emitter_and_expiration() {
    let mut a = Particles::default();
    let mut b = Particles::default();
    for pool in [&mut a, &mut b] {
        let template = Particle {
            position: Vec3::ZERO,
            velocity: Vec3::Y,
            remaining: 0.3,
        };
        assert_eq!(
            Emitter::new(42).burst(pool, 30, template, Vec3::splat(0.2)),
            30
        );
        pool.push(Particle {
            remaining: 0.05,
            ..template
        });
        pool.step(0.1, ParticleSettings::default(), &[]);
    }
    assert_eq!(a.len(), 30);
    assert!(a.iter().eq(b.iter()));
    a.step(0.25, ParticleSettings::default(), &[]);
    assert!(a.is_empty());
}

#[test]
#[cfg(feature = "rayon")]
fn particle_parallel_update_matches_serial() {
    use rust_physics::particles::Field;
    let build = || {
        let mut pool = Particles::default();
        Emitter::new(13).burst(
            &mut pool,
            1_001,
            Particle {
                position: Vec3::X,
                velocity: Vec3::ZERO,
                remaining: 2.0,
            },
            Vec3::ONE,
        );
        pool
    };
    let (mut a, mut b) = (build(), build());
    let fields = [Field::Vortex {
        center: Vec3::ZERO,
        axis: Vec3::Y,
        strength: 1.25,
    }];
    a.step(0.02, ParticleSettings::default(), &fields);
    b.step_parallel(0.02, ParticleSettings::default(), &fields);
    assert!(a.iter().eq(b.iter()));
}

#[test]
fn cloth_keeps_pins_and_respects_floor_boundary() {
    let mut world = PhysicsWorld::new();
    world.add_body(RigidBodyBuilder::static_body().shape(Shape::Plane).build());
    let mut cloth = SoftBody::cloth([6, 6], 0.2, Vec3::Y * 0.6, 1e-6).unwrap();
    let settings = SoftSettings::default();
    for _ in 0..240 {
        assert!(cloth.step(
            1.0 / 120.0,
            settings,
            Some((&world, QueryFilter::default()))
        ));
    }
    assert_eq!(cloth.position(0).unwrap(), Vec3::Y * 0.6);
    assert_eq!(cloth.position(5).unwrap(), Vec3::new(1.0, 0.6, 0.0));
    assert_eq!(cloth.triangles().len(), 50);
    assert!(
        cloth
            .positions()
            .all(|p| p.is_finite() && p.y >= settings.radius - 1e-12)
    );
    assert!(cloth.positions().any(|p| p.y < 0.1));
}

#[test]
fn xpbd_tetrahedron_keeps_its_signed_volume() {
    let mut body = SoftBody::default();
    for (point, mass) in [
        (Vec3::ZERO, 0.0),
        (Vec3::X, 0.0),
        (Vec3::Z, 0.0),
        (Vec3::Y, 1.0),
    ] {
        body.push_node(point, mass).unwrap();
    }
    assert!(body.add_volume([0, 1, 2, 3], 0.0));
    for _ in 0..60 {
        body.step(1.0 / 60.0, SoftSettings::default(), None);
    }
    assert!((body.position(3).unwrap().y - 1.0).abs() < 1e-12);
}

#[test]
fn sph_pair_forces_conserve_linear_momentum_and_repulse() {
    let mut fluid = Fluid::new(FluidSettings {
        gravity: Vec3::ZERO,
        rest_density: 100.0,
        ..Default::default()
    })
    .unwrap();
    fluid.push(Vec3::X * -0.05, Vec3::Y);
    fluid.push(Vec3::X * 0.05, -Vec3::Y);
    fluid.step(0.001, None);
    let [a, b] = [fluid.velocities()[0], fluid.velocities()[1]];
    assert!((a + b).length() < 1e-12);
    assert!(a.x < 0.0 && b.x > 0.0);
    assert!(a.y < 1.0 && b.y > -1.0);
    assert_eq!(fluid.densities()[0], fluid.densities()[1]);
}
