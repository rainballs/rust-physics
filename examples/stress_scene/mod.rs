use rust_physics::{
    BroadPhase, Material, PhysicsWorld, Quat, RigidBodyBuilder, Settings, Shape, Vec3,
};

fn unit(seed: u64) -> f64 {
    let mut bits = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
    bits = (bits ^ (bits >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    bits = (bits ^ (bits >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    ((bits ^ (bits >> 31)) >> 11) as f64 / (1_u64 << 53) as f64
}

fn body(i: usize, seed: u64, height: f64) -> rust_physics::RigidBody {
    let t = (i as u64).wrapping_mul(17) + seed * 50_000;
    let r = 0.17 + 0.2 * unit(t);
    let shape = match i % 4 {
        0 => Shape::sphere(r),
        1 => Shape::cuboid(Vec3::new(r, r * 1.4, r * 0.8)),
        2 => Shape::capsule(r, r * 0.7),
        _ => Shape::cylinder(r, r * 0.6),
    };
    let spin = Vec3::new(unit(t + 1), unit(t + 2), unit(t + 3)) * 2.0;
    let velocity = Vec3::new(unit(t + 4) - 0.5, 0.0, unit(t + 5) - 0.5) * 3.0;
    let position = Vec3::new(
        (i % 4) as f64 * 0.75 - 1.2,
        height,
        ((i / 4) % 4) as f64 * 0.75 - 1.2,
    );
    RigidBodyBuilder::dynamic()
        .shape(shape)
        .position(position)
        .velocity(velocity)
        .rotation(Quat::from_scaled_axis(spin))
        .angular_velocity(spin)
        .mass(0.2 + unit(t + 6) * 6.0)
        .material(Material {
            restitution: unit(t + 7) * 0.4,
            ..Default::default()
        })
        .ccd(i.is_multiple_of(11))
        .build()
}

pub fn soak(seed: u64, frames: usize, trace: bool) -> (usize, usize) {
    let modes = [
        BroadPhase::SweepAndPrune,
        BroadPhase::Bvh,
        BroadPhase::UniformGrid,
        BroadPhase::DynamicTree,
    ];
    let mut world = PhysicsWorld::new();
    world
        .set_settings(Settings {
            broad_phase: modes[seed as usize % modes.len()],
            ccd_moving_spheres: seed.is_multiple_of(3),
            ccd_convex: seed.is_multiple_of(5),
            ..Default::default()
        })
        .unwrap();
    for (position, rotation) in [
        (Vec3::ZERO, Quat::IDENTITY),
        (
            Vec3::X * 6.0,
            Quat::from_rotation_z(std::f64::consts::FRAC_PI_2),
        ),
        (
            -Vec3::X * 6.0,
            Quat::from_rotation_z(-std::f64::consts::FRAC_PI_2),
        ),
        (
            Vec3::Z * 6.0,
            Quat::from_rotation_x(-std::f64::consts::FRAC_PI_2),
        ),
        (
            -Vec3::Z * 6.0,
            Quat::from_rotation_x(std::f64::consts::FRAC_PI_2),
        ),
    ] {
        world.add_body(
            RigidBodyBuilder::static_body()
                .shape(Shape::Plane)
                .position(position)
                .rotation(rotation)
                .build(),
        );
    }
    let mut handles: Vec<_> = (0..48)
        .map(|i| world.add_body(body(i, seed, 0.5 + (i / 16) as f64 * 0.9)))
        .collect();
    let mut collisions = 0;
    let mut impacts = 0;
    for frame in 0..frames {
        if frame.is_multiple_of(80) {
            let index = (frame / 80 * 17) % handles.len();
            let old = handles[index];
            world.remove_body(old).unwrap();
            handles[index] = world.add_body(body(index + frame, seed, 4.0));
            assert!(world.body(old).is_none());
        }
        if frame.is_multiple_of(27) {
            let index = (frame / 27 * 11) % handles.len();
            let handle = handles[index];
            let origin = world.body(handle).unwrap().position();
            let impulse = Vec3::new(
                unit(seed + frame as u64) - 0.5,
                1.0,
                unit(seed + index as u64) - 0.5,
            );
            world.apply_impulse(handle, impulse * 3.0, origin + Vec3::X * 0.1);
        }
        let stats = if trace {
            let profile = world.step_fixed_profiled().unwrap();
            if frame.is_multiple_of(100) || profile.timings.total.as_secs_f64() >= 0.05 {
                eprintln!(
                    "seed {seed} frame {frame} contacts {} ccd {} fallbacks {} {:?}",
                    profile.stats.contact_points,
                    profile.stats.ccd_impacts,
                    profile.stats.ccd_budget_fallbacks,
                    profile.timings
                );
            }
            profile.stats
        } else {
            world.step_fixed().unwrap()
        };
        collisions += stats.contact_points;
        impacts += stats.ccd_impacts;
        assert!(
            world
                .debug_contacts()
                .iter()
                .all(|(_, _, contact)| contact.normal.is_normalized()
                    && contact.point.is_finite()
                    && contact.depth.is_finite()
                    && contact.depth >= 0.0),
            "seed {seed} frame {frame}: invalid contact"
        );
        for (_, body) in world.bodies() {
            assert!(
                body.position().is_finite()
                    && body.position().abs().max_element() < 30.0
                    && body.rotation().is_normalized()
                    && body.velocity().is_finite()
                    && body.velocity().length() < 200.0
                    && body.angular_velocity().is_finite(),
                "seed {seed} frame {frame}: {body:?}"
            );
        }
    }
    (collisions, impacts)
}
