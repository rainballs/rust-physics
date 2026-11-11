//! Renderer-independent collider, contact, and velocity line geometry.
use crate::spatial::Aabb;
use crate::{PhysicsWorld, RigidBody, Shape, Vec3};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LineKind {
    Collider,
    Sensor,
    Sleeping,
    Bounds,
    Contact,
    Velocity,
    Force,
    CenterOfMass,
    Joint,
}

#[derive(Clone, Copy, Debug)]
pub struct Line {
    pub from: Vec3,
    pub to: Vec3,
    pub kind: LineKind,
}

#[derive(Clone, Copy, Debug)]
pub struct DebugOptions {
    pub colliders: bool,
    pub bounds: bool,
    pub contacts: bool,
    pub velocities: bool,
    pub forces: bool,
    pub centers: bool,
    pub joints: bool,
    /// Width and depth of the finite marker used to draw an infinite plane.
    pub plane_extent: f64,
}

impl Default for DebugOptions {
    fn default() -> Self {
        Self {
            colliders: true,
            bounds: false,
            contacts: true,
            velocities: false,
            forces: false,
            centers: false,
            joints: true,
            plane_extent: 8.0,
        }
    }
}

/// Streams line segments in current body-storage order.
pub fn lines(world: &PhysicsWorld, options: DebugOptions, mut emit: impl FnMut(Line)) {
    let mut line = |from, to, kind| emit(Line { from, to, kind });
    for (_, b) in world.bodies() {
        let center = b.position();
        if options.colliders {
            let kind = if b.is_sensor() {
                LineKind::Sensor
            } else if b.is_sleeping() {
                LineKind::Sleeping
            } else {
                LineKind::Collider
            };
            wireframe(b, options.plane_extent, &mut |a, b| line(a, b, kind));
        }
        if options.bounds
            && let Some(aabb) = Aabb::of(b)
        {
            box_edges((aabb.max - aabb.min) * 0.5, |a, b| {
                let center = (aabb.max + aabb.min) * 0.5;
                line(center + a, center + b, LineKind::Bounds);
            });
        }
        if options.velocities && b.velocity() != Vec3::ZERO {
            line(center, center + b.velocity() * 0.2, LineKind::Velocity);
        }
        if options.forces && b.force != Vec3::ZERO {
            line(center, center + b.force * 0.1, LineKind::Force);
        }
        if options.centers {
            for axis in Vec3::AXES {
                line(
                    center - axis * 0.08,
                    center + axis * 0.08,
                    LineKind::CenterOfMass,
                );
            }
        }
    }
    if options.contacts {
        for (_, _, c) in world.debug_contacts() {
            line(
                c.point,
                c.point + c.normal * (0.15 + c.depth),
                LineKind::Contact,
            );
        }
    }
    if options.joints {
        for (_, joint) in world.joints() {
            let points = [0, 1].map(|i| {
                let b = world.body(joint.bodies[i]).expect("live joint");
                b.position() + b.rotation() * joint.anchors[i]
            });
            line(points[0], points[1], LineKind::Joint);
        }
    }
}

fn wireframe(b: &RigidBody, plane_extent: f64, line: &mut dyn FnMut(Vec3, Vec3)) {
    let mut local = |p, q| {
        line(
            b.position() + b.rotation() * p,
            b.position() + b.rotation() * q,
        )
    };
    match b.shape {
        Shape::Sphere { radius } => {
            for (u, v) in [(Vec3::X, Vec3::Y), (Vec3::Y, Vec3::Z), (Vec3::Z, Vec3::X)] {
                arc(
                    Vec3::ZERO,
                    u * radius,
                    v * radius,
                    std::f64::consts::TAU,
                    20,
                    &mut local,
                );
            }
        }
        Shape::Cuboid { half_extents } => box_edges(half_extents, local),
        Shape::Capsule {
            radius,
            half_segment,
        } => {
            for sign in [-1.0, 1.0] {
                let end = Vec3::Y * (sign * half_segment);
                arc(
                    end,
                    Vec3::X * radius,
                    Vec3::Z * radius,
                    std::f64::consts::TAU,
                    20,
                    &mut local,
                );
                for side in [Vec3::X, Vec3::Z] {
                    arc(
                        end,
                        side * radius,
                        Vec3::Y * (sign * radius),
                        std::f64::consts::PI,
                        12,
                        &mut local,
                    );
                    local(-end + side * radius, end + side * radius);
                }
            }
        }
        Shape::Cylinder {
            radius,
            half_height,
        } => {
            let end = Vec3::Y * half_height;
            for side in [Vec3::X, -Vec3::X, Vec3::Z, -Vec3::Z] {
                local(side * radius - end, side * radius + end);
            }
            for center in [-end, end] {
                arc(
                    center,
                    Vec3::X * radius,
                    Vec3::Z * radius,
                    std::f64::consts::TAU,
                    20,
                    &mut local,
                );
            }
        }
        Shape::Compound(ref shape) => {
            for part in shape.parts() {
                wireframe(&b.collider_part(part), plane_extent, line);
            }
        }
        Shape::Mesh(ref mesh) => mesh_wireframe(mesh, local),
        Shape::Hull(ref hull) => mesh_wireframe(hull.surface(), local),
        Shape::Plane => {
            let e = plane_extent;
            for i in -8..=8 {
                let s = f64::from(i) * e / 8.0;
                local(Vec3::new(-e, 0.0, s), Vec3::new(e, 0.0, s));
                local(Vec3::new(s, 0.0, -e), Vec3::new(s, 0.0, e));
            }
        }
    }
}

fn mesh_wireframe(mesh: &crate::terrain::TriangleMesh, mut line: impl FnMut(Vec3, Vec3)) {
    for &[a, b, c] in mesh.triangles() {
        let [a, b, c] = [a, b, c].map(|i| mesh.vertices()[i]);
        line(a, b);
        line(b, c);
        line(c, a);
    }
}

fn arc(
    center: Vec3,
    u: Vec3,
    v: Vec3,
    angle: f64,
    segments: usize,
    mut line: impl FnMut(Vec3, Vec3),
) {
    let mut previous = center + u;
    for i in 1..=segments {
        let t = angle * i as f64 / segments as f64;
        let point = center + u * t.cos() + v * t.sin();
        line(previous, point);
        previous = point;
    }
}

fn box_edges(h: Vec3, mut line: impl FnMut(Vec3, Vec3)) {
    for axis in 0..3 {
        for u in [-1.0, 1.0] {
            for v in [-1.0, 1.0] {
                let mut start = -h;
                start[(axis + 1) % 3] *= u;
                start[(axis + 2) % 3] *= v;
                let mut end = start;
                end[axis] = h[axis];
                line(start, end);
            }
        }
    }
}
