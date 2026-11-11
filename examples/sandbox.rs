use macroquad::prelude::*;
use rust_physics::debug::{self, DebugOptions, LineKind};
use rust_physics::fluid::{Fluid, FluidSettings};
use rust_physics::particles::{Emitter, Particle, ParticleSettings, Particles};
use rust_physics::softbody::{SoftBody, SoftSettings};
use rust_physics::{
    Joint, JointKind, Material, PhysicsWorld, Quat, RigidBodyBuilder, Shape, StepStats,
    Vec3 as DVec3,
};
use std::time::Instant;

const DT: f64 = 1.0 / 60.0;
const SCENES: [&str; 10] = [
    "Falling spheres",
    "Box stacks",
    "Dominoes",
    "Pendulum",
    "Jointed ragdoll",
    "Rotated cuboids",
    "Cloth",
    "Particle fountain",
    "SPH tank",
    "Stress test",
];
const KEYS: [KeyCode; 10] = [
    KeyCode::Key1,
    KeyCode::Key2,
    KeyCode::Key3,
    KeyCode::Key4,
    KeyCode::Key5,
    KeyCode::Key6,
    KeyCode::Key7,
    KeyCode::Key8,
    KeyCode::Key9,
    KeyCode::Key0,
];

struct Scene {
    world: PhysicsWorld,
    cloth: Option<SoftBody>,
    particles: Option<(Particles, Emitter)>,
    fluid: Option<Fluid>,
}

impl Scene {
    fn new(index: usize) -> Self {
        let mut scene = Self {
            world: PhysicsWorld::new(),
            cloth: None,
            particles: None,
            fluid: None,
        };
        scene
            .world
            .add_body(RigidBodyBuilder::static_body().shape(Shape::Plane).build());
        let w = &mut scene.world;
        match index {
            0 => {
                for i in 0..14 {
                    w.add_body(
                        RigidBodyBuilder::dynamic()
                            .position(DVec3::new(
                                (i % 5) as f64 * 1.1 - 2.2,
                                2.0 + i as f64 * 0.6,
                                (i % 3) as f64 - 1.0,
                            ))
                            .material(Material {
                                restitution: 0.5,
                                ..Default::default()
                            })
                            .build(),
                    );
                }
            }
            1 => {
                for x in -2..=2 {
                    for y in 0..8 {
                        w.add_body(
                            RigidBodyBuilder::dynamic()
                                .position(DVec3::new(x as f64 * 1.1, y as f64 * 0.8 + 0.4, 0.0))
                                .shape(Shape::cuboid(DVec3::splat(0.4)))
                                .build(),
                        );
                    }
                }
            }
            2 => {
                for i in 0..18 {
                    let tilt = if i == 0 { -0.22 } else { 0.0 };
                    w.add_body(
                        RigidBodyBuilder::dynamic()
                            .position(DVec3::new(i as f64 * 0.6 - 5.1, 0.72, 0.0))
                            .rotation(Quat::from_rotation_z(tilt))
                            .shape(Shape::cuboid(DVec3::new(0.14, 0.7, 0.43)))
                            .build(),
                    );
                }
            }
            3 => {
                let p = DVec3::Y * 4.5;
                let anchor = w.add_body(
                    RigidBodyBuilder::static_body()
                        .position(p)
                        .shape(Shape::sphere(0.16))
                        .build(),
                );
                let bob = w.add_body(
                    RigidBodyBuilder::dynamic()
                        .position(p + DVec3::new(2.0, -2.0, 0.0))
                        .build(),
                );
                w.add_joint(Joint::new(
                    [anchor, bob],
                    [DVec3::ZERO; 2],
                    JointKind::Distance {
                        length: 2.0_f64.sqrt() * 2.0,
                    },
                ))
                .unwrap();
            }
            4 => ragdoll(w),
            5 => {
                for i in 0..16 {
                    w.add_body(
                        RigidBodyBuilder::dynamic()
                            .position(DVec3::new((i % 4) as f64 - 1.5, 1.0 + 0.4 * i as f64, 0.0))
                            .rotation(Quat::from_euler(
                                ::glam::EulerRot::XYZ,
                                i as f64 * 0.17,
                                i as f64 * 0.13,
                                0.4,
                            ))
                            .angular_velocity(DVec3::new(1.0, -0.5, 0.4))
                            .shape(Shape::cuboid(DVec3::new(0.36, 0.62, 0.31)))
                            .build(),
                    );
                }
            }
            6 => {
                w.add_body(
                    RigidBodyBuilder::static_body()
                        .position(DVec3::Y * 0.7)
                        .shape(Shape::sphere(1.0))
                        .build(),
                );
                scene.cloth = SoftBody::cloth([21, 21], 0.15, DVec3::new(-1.5, 2.8, -1.5), 5e-7);
            }
            7 => scene.particles = Some((Particles::default(), Emitter::new(31))),
            8 => {
                for side in [-1.0, 1.0] {
                    for (center, half) in [
                        (DVec3::new(side * 1.5, 0.7, 0.0), DVec3::new(0.08, 0.7, 1.6)),
                        (DVec3::new(0.0, 0.7, side * 1.5), DVec3::new(1.6, 0.7, 0.08)),
                    ] {
                        w.add_body(
                            RigidBodyBuilder::static_body()
                                .position(center)
                                .shape(Shape::cuboid(half))
                                .build(),
                        );
                    }
                }
                let mut fluid = Fluid::new(FluidSettings::default()).unwrap();
                for x in 0..9 {
                    for y in 0..8 {
                        for z in 0..9 {
                            let p = DVec3::new(
                                x as f64 * 0.16 - 1.1,
                                y as f64 * 0.16 + 0.27,
                                z as f64 * 0.16 - 1.1,
                            );
                            fluid.push(p, DVec3::ZERO);
                        }
                    }
                }
                scene.fluid = Some(fluid);
            }
            _ => {
                for x in -20..20 {
                    for z in -20..20 {
                        w.add_body(
                            RigidBodyBuilder::dynamic()
                                .position(DVec3::new(
                                    x as f64 * 0.42,
                                    1.0 + ((x + z) % 4_i32).abs() as f64,
                                    z as f64 * 0.42,
                                ))
                                .shape(Shape::sphere(0.16))
                                .build(),
                        );
                    }
                }
            }
        }
        scene
    }

    fn step(&mut self) -> StepStats {
        let stats = self.world.step_fixed().unwrap();
        let filter = Default::default();
        if let Some(cloth) = &mut self.cloth {
            for _ in 0..2 {
                cloth.step(
                    DT * 0.5,
                    SoftSettings::default(),
                    Some((&self.world, filter)),
                );
            }
        }
        if let Some((particles, emitter)) = &mut self.particles {
            emitter.burst(
                particles,
                20,
                Particle {
                    position: DVec3::Y * 0.35,
                    velocity: DVec3::Y * 6.0,
                    remaining: 3.0,
                },
                DVec3::new(2.0, 0.8, 2.0),
            );
            let settings = ParticleSettings {
                drag: 0.15,
                ..Default::default()
            };
            particles.step(DT, settings, &[]);
            particles.collide(&self.world, filter, settings);
        }
        if let Some(fluid) = &mut self.fluid {
            for _ in 0..4 {
                fluid.step(DT / 4.0, Some((&self.world, filter)));
            }
        }
        stats
    }
}

fn ragdoll(world: &mut PhysicsWorld) {
    let parts = [
        (DVec3::new(0.0, 4.0, 0.0), DVec3::new(0.35, 0.55, 0.2)),
        (DVec3::new(0.0, 4.82, 0.0), DVec3::splat(0.23)),
        (DVec3::new(-0.64, 4.24, 0.0), DVec3::new(0.31, 0.14, 0.14)),
        (DVec3::new(0.64, 4.24, 0.0), DVec3::new(0.31, 0.14, 0.14)),
        (DVec3::new(-0.19, 2.98, 0.0), DVec3::new(0.16, 0.47, 0.16)),
        (DVec3::new(0.19, 2.98, 0.0), DVec3::new(0.16, 0.47, 0.16)),
    ];
    let ids = parts.map(|(p, h)| {
        world.add_body(
            RigidBodyBuilder::dynamic()
                .position(p)
                .shape(Shape::cuboid(h))
                .filter(2, 1)
                .velocity(DVec3::new(0.65, 0.0, p.y * 0.17))
                .build(),
        )
    });
    for (i, anchor) in [
        DVec3::new(0.0, 4.55, 0.0),
        DVec3::new(-0.35, 4.24, 0.0),
        DVec3::new(0.35, 4.24, 0.0),
        DVec3::new(-0.19, 3.45, 0.0),
        DVec3::new(0.19, 3.45, 0.0),
    ]
    .into_iter()
    .enumerate()
    {
        world
            .add_joint(Joint::new(
                [ids[0], ids[i + 1]],
                [anchor - parts[0].0, anchor - parts[i + 1].0],
                JointKind::Ball,
            ))
            .unwrap();
    }
}

fn render(scene: &Scene, options: DebugOptions, dense: bool) {
    let mut options = options;
    options.colliders &= !dense;
    debug::lines(&scene.world, options, |line| {
        let color = match line.kind {
            LineKind::Collider => SKYBLUE,
            LineKind::Sensor => VIOLET,
            LineKind::Sleeping => GRAY,
            LineKind::Bounds => DARKGREEN,
            LineKind::Contact => ORANGE,
            LineKind::Velocity => YELLOW,
            LineKind::Force => MAGENTA,
            LineKind::CenterOfMass => GREEN,
            LineKind::Joint => WHITE,
        };
        draw_line_3d(convert(line.from), convert(line.to), color);
    });
    if dense {
        for (_, b) in scene.world.bodies().skip(1) {
            draw_cube(convert(b.position()), Vec3::splat(0.2), None, SKYBLUE);
        }
    }
    if let Some(cloth) = &scene.cloth {
        let positions: Vec<_> = cloth.positions().map(convert).collect();
        for &[a, b, c] in cloth.triangles() {
            for (i, j) in [(a, b), (b, c), (c, a)] {
                draw_line_3d(positions[i], positions[j], ORANGE);
            }
        }
    }
    if let Some((particles, _)) = &scene.particles {
        for &p in particles.positions() {
            draw_cube(convert(p), Vec3::splat(0.036), None, YELLOW);
        }
    }
    if let Some(fluid) = &scene.fluid {
        for &p in fluid.positions() {
            draw_cube(convert(p), Vec3::splat(0.09), None, BLUE);
        }
    }
}

fn convert(v: DVec3) -> Vec3 {
    vec3(v.x as f32, v.y as f32, v.z as f32)
}

#[macroquad::main("Rust Physics Sandbox")]
async fn main() {
    let mut index = 0;
    let mut scene = Scene::new(index);
    let mut paused = false;
    let mut options = DebugOptions::default();
    let mut accumulator = 0.0;
    let mut yaw = 0.5_f32;
    let mut pitch = 0.44_f32;
    let mut distance = 13.0_f32;
    let mut stats = StepStats::default();
    let mut last_step = 0.0;
    let mut previous_mouse = mouse_position();
    let smoke = std::env::var_os("RUST_PHYSICS_SMOKE").is_some();
    let mut smoke_scene = 0;

    loop {
        if smoke {
            index = smoke_scene;
            scene = Scene::new(index);
            accumulator = DT;
        }
        if is_key_pressed(KeyCode::Escape) {
            break;
        }
        if let Some(selected) = KEYS.iter().position(|&key| is_key_pressed(key)) {
            index = selected;
            scene = Scene::new(index);
            accumulator = 0.0;
        }
        if is_key_pressed(KeyCode::R) {
            scene = Scene::new(index);
            accumulator = 0.0;
        }
        if is_key_pressed(KeyCode::Space) {
            paused = !paused;
        }
        if is_key_pressed(KeyCode::B) {
            options.bounds = !options.bounds;
        }
        if is_key_pressed(KeyCode::C) {
            options.contacts = !options.contacts;
        }
        if is_key_pressed(KeyCode::V) {
            options.velocities = !options.velocities;
        }
        if is_key_pressed(KeyCode::G) {
            let gravity = if scene.world.settings().gravity == DVec3::ZERO {
                -DVec3::Y * 9.81
            } else {
                DVec3::ZERO
            };
            scene.world.set_gravity(gravity).unwrap();
        }
        if is_key_pressed(KeyCode::S) {
            scene.world.add_body(
                RigidBodyBuilder::dynamic()
                    .position(DVec3::new(0.2, 6.0, 1.0))
                    .build(),
            );
        }
        if is_key_pressed(KeyCode::X) {
            scene.world.add_body(
                RigidBodyBuilder::dynamic()
                    .position(DVec3::new(0.2, 6.0, 1.0))
                    .shape(Shape::cuboid(DVec3::splat(0.35)))
                    .build(),
            );
        }
        if paused {
            accumulator = if is_key_pressed(KeyCode::N) { DT } else { 0.0 };
        } else {
            accumulator += f64::from(get_frame_time()).min(4.0 * DT);
        }
        while accumulator >= DT {
            let start = Instant::now();
            stats = scene.step();
            last_step = start.elapsed().as_secs_f64() * 1000.0;
            accumulator -= DT;
        }
        let mouse = mouse_position();
        if is_mouse_button_down(MouseButton::Right) {
            yaw -= (mouse.0 - previous_mouse.0) * 0.01;
            pitch = (pitch + (mouse.1 - previous_mouse.1) * 0.01).clamp(-0.15, 1.48);
        }
        previous_mouse = mouse;
        distance = (distance - mouse_wheel().1 * 0.75).clamp(3.0, 50.0);
        let target = vec3(0.0, 1.5, 0.0);
        let camera = target
            + vec3(
                yaw.sin() * pitch.cos(),
                pitch.sin(),
                yaw.cos() * pitch.cos(),
            ) * distance;
        clear_background(Color::from_rgba(16, 24, 32, 255));
        set_camera(&Camera3D {
            position: camera,
            target,
            up: Vec3::Y,
            ..Default::default()
        });
        render(&scene, options, index == 9);
        set_default_camera();
        draw_rectangle(
            0.0,
            0.0,
            screen_width(),
            120.0,
            Color::from_rgba(16, 24, 32, 232),
        );
        let state = if paused { "Paused" } else { "Running" };
        draw_text(
            format!("{}   |   {state}", SCENES[index]),
            16.0,
            30.0,
            26.0,
            WHITE,
        );
        draw_text(
            format!(
                "{} bodies   {} contacts   {} sleeping   {:.2} ms/step",
                scene.world.bodies().len(),
                stats.contact_points,
                stats.sleeping_bodies,
                last_step
            ),
            16.0,
            52.0,
            18.0,
            SKYBLUE,
        );
        draw_text(
            "1–0 scenes   Space pause   N step   R reset   S sphere   X box   G gravity",
            16.0,
            76.0,
            17.0,
            WHITE,
        );
        draw_text(
            "Right-drag orbit   Wheel zoom   B bounds   C contacts   V velocity   Esc exit",
            16.0,
            98.0,
            17.0,
            WHITE,
        );
        next_frame().await;
        if smoke {
            assert!(scene.world.bodies().all(|(_, b)| b.position().is_finite()));
            println!("rendered {}", SCENES[index]);
            smoke_scene += 1;
            if smoke_scene == SCENES.len() {
                break;
            }
        }
    }
}
