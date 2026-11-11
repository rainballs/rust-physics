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

#[test]
fn soft_body_self_contact_separates_unlinked_nodes_and_respects_pins() {
    let mut body = SoftBody::default();
    let a = body.push_node(Vec3::ZERO, 0.0).unwrap();
    let b = body.push_node(Vec3::ZERO, 1.0).unwrap();
    let c = body.push_node(Vec3::X, 1.0).unwrap();
    let d = body.push_node(Vec3::X * 1.05, 1.0).unwrap();
    assert!(body.add_distance([c, d], 0.0));
    let settings = SoftSettings {
        gravity: Vec3::ZERO,
        self_collision_distance: Some(0.1),
        ..Default::default()
    };
    assert!(body.step(1.0 / 60.0, settings, None));
    assert_eq!(body.position(a), Some(Vec3::ZERO));
    assert!((body.position(b).unwrap().length() - 0.1).abs() < 1e-12);
    assert!(
        (body
            .position(c)
            .unwrap()
            .distance(body.position(d).unwrap())
            - 0.05)
            .abs()
            < 1e-12
    );
}

#[test]
fn pbf_projection_reduces_compression_without_center_drift() {
    use rust_physics::fluid::{PbfFluid, PbfSettings};
    let mut fluid = PbfFluid::new(PbfSettings {
        gravity: Vec3::ZERO,
        rest_density: 280.0,
        viscosity: 0.0,
        ..Default::default()
    })
    .unwrap();
    fluid.push(Vec3::X * -0.06, Vec3::ZERO);
    fluid.push(Vec3::X * 0.06, Vec3::ZERO);
    fluid.step(0.01, None);
    let [a, b] = [fluid.positions()[0], fluid.positions()[1]];
    assert!((a + b).length() < 1e-12);
    assert!(a.distance(b) > 0.12);
    assert!(fluid.densities().iter().all(|&rho| rho <= 280.1));
}

#[test]
fn pbf_floor_boundary_keeps_particles_above_surface() {
    use rust_physics::fluid::{PbfFluid, PbfSettings};
    let mut w = PhysicsWorld::new();
    w.add_body(RigidBodyBuilder::static_body().shape(Shape::Plane).build());
    let settings = PbfSettings::default();
    let mut fluid = PbfFluid::new(settings).unwrap();
    for x in 0..3 {
        for z in 0..3 {
            fluid.push(
                Vec3::new(f64::from(x) * 0.16, 0.3, f64::from(z) * 0.16),
                Vec3::ZERO,
            );
        }
    }
    for _ in 0..80 {
        fluid.step(0.01, Some((&w, QueryFilter::default())));
    }
    assert!(
        fluid
            .positions()
            .iter()
            .all(|p| p.is_finite() && p.y >= settings.collision_radius)
    );
}

#[test]
fn sph_rigid_boundary_exchanges_momentum_using_both_masses() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vec3::ZERO).unwrap();
    let id = world.add_body(RigidBodyBuilder::dynamic().mass(3.0).build());
    let mut fluid = Fluid::new(FluidSettings {
        gravity: Vec3::ZERO,
        particle_mass: 1.0,
        collision_radius: 0.1,
        restitution: 1.0,
        ..Default::default()
    })
    .unwrap();
    fluid.push(Vec3::Y * 0.7, -Vec3::Y * 2.0);
    assert!(fluid.step_coupled(0.1, &mut world, QueryFilter::default()));
    let particle = fluid.velocities()[0];
    let body = world.body(id).unwrap().velocity();
    assert!(particle.distance(Vec3::Y) < 1e-12);
    assert!(body.distance(-Vec3::Y) < 1e-12);
    assert!((particle + body * 3.0).distance(-Vec3::Y * 2.0) < 1e-12);
}

#[test]
fn equal_mass_particle_pair_exchanges_normal_velocities() {
    let mut particles = Particles::default();
    particles.push(Particle {
        position: -Vec3::X * 0.1,
        velocity: Vec3::X * 2.0,
        remaining: 1.0,
    });
    particles.push(Particle {
        position: Vec3::X * 0.1,
        velocity: -Vec3::X,
        remaining: 1.0,
    });
    assert!(particles.collide_pairs(0.15, 1.0));
    let states: Vec<_> = particles.iter().collect();
    assert_eq!(states[0].velocity, -Vec3::X);
    assert_eq!(states[1].velocity, Vec3::X * 2.0);
    assert!((states[0].position.distance(states[1].position) - 0.3).abs() < 1e-12);
}

#[test]
fn cloth_surface_self_collision_reaches_triangle_interior() {
    use rust_physics::softbody::SelfCollision;
    let mut body = SoftBody::default();
    for p in [-Vec3::X - Vec3::Z, Vec3::X - Vec3::Z, Vec3::Z] {
        body.push_node(p, 0.0).unwrap();
    }
    assert!(body.add_triangle([0, 1, 2]));
    let i = body.push_node(Vec3::Y * 0.04, 1.0).unwrap();
    let settings = SoftSettings {
        gravity: Vec3::ZERO,
        self_collision_distance: Some(0.1),
        self_collision: SelfCollision::Surface,
        ..Default::default()
    };
    assert!(body.step(1.0 / 60.0, settings, None));
    assert!(body.position(i).unwrap().distance(Vec3::Y * 0.1) < 1e-12);
}

#[test]
fn cloth_surface_self_collision_handles_crossing_interior_edges() {
    use rust_physics::softbody::SelfCollision;
    let mut body = SoftBody::default();
    for p in [Vec3::X * -2.0, Vec3::X * 2.0, Vec3::Z * -2.0] {
        body.push_node(p, 0.0).unwrap();
    }
    for p in [Vec3::Z * -3.0, Vec3::Z * 3.0, Vec3::new(3.0, 0.0, 3.0)] {
        body.push_node(p + Vec3::Y * 0.04, 1.0).unwrap();
    }
    assert!(body.add_triangle([0, 1, 2]));
    assert!(body.add_triangle([3, 4, 5]));
    let settings = SoftSettings {
        gravity: Vec3::ZERO,
        self_collision_distance: Some(0.1),
        self_collision: SelfCollision::Surface,
        ..Default::default()
    };
    body.step(1.0 / 60.0, settings, None);
    let middle = (body.position(3).unwrap() + body.position(4).unwrap()) * 0.5;
    assert!(middle.y >= 0.1 - 1e-8);
}

#[test]
fn coupled_soft_node_transfers_momentum_to_dynamic_collider() {
    let mut world = PhysicsWorld::new();
    world.set_gravity(Vec3::ZERO).unwrap();
    let sphere = world.add_body(RigidBodyBuilder::dynamic().mass(3.0).build());
    let mut soft = SoftBody::default();
    let index = soft.push_node(Vec3::Y * 0.65, 1.0).unwrap();
    let settings = SoftSettings {
        gravity: -Vec3::Y * 20.0,
        radius: 0.1,
        damping: 0.0,
        ..Default::default()
    };
    soft.step_coupled(0.1, settings, &mut world, QueryFilter::default());
    let velocity = soft.velocity(index).unwrap();
    let rigid = world.body(sphere).unwrap().velocity();
    assert!(velocity.distance(rigid) < 1e-12);
    assert!((velocity + rigid * 3.0).distance(-Vec3::Y * 2.0) < 1e-12);
    assert!((soft.position(index).unwrap().y - 0.6).abs() < 1e-12);
}

#[test]
fn color_field_surface_tension_pulls_symmetric_interface_inward() {
    let mut fluid = Fluid::new(FluidSettings {
        gravity: Vec3::ZERO,
        viscosity: 0.0,
        pressure_stiffness: 0.0,
        surface_tension: 1.0,
        ..Default::default()
    })
    .unwrap();
    fluid.push(-Vec3::X * 0.1, Vec3::ZERO);
    fluid.push(Vec3::X * 0.1, Vec3::ZERO);
    fluid.step(0.001, None);
    let [a, b] = [fluid.velocities()[0], fluid.velocities()[1]];
    assert!(a.x > 0.0 && b.x < 0.0);
    assert!((a + b).length() < 1e-12);
}

#[test]
fn fluid_advance_returns_time_left_when_budget_is_exhausted() {
    let mut fluid = Fluid::new(FluidSettings::default()).unwrap();
    fluid.push(Vec3::ZERO, Vec3::ZERO);
    let progress = fluid.advance(0.1, 2, None).unwrap();
    assert_eq!(progress.substeps, 2);
    assert!(progress.remaining > 0.0 && progress.remaining < 0.1);
    let rest = fluid.advance(progress.remaining, 64, None).unwrap();
    assert_eq!(rest.remaining, 0.0);
    assert!((fluid.velocities()[0].y + 0.981).abs() < 1e-12);
}
