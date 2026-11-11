//! Optional environmental forces for rigid simulations.
use crate::{BodyHandle, BodyKind, PhysicsWorld, Shape, Vec3};
use std::collections::BTreeSet;
use std::f64::consts::PI;

#[derive(Clone, Copy, Debug)]
pub struct Atmosphere {
    pub density: f64,
    pub drag_coefficient: f64,
    pub wind: Vec3,
}

impl Atmosphere {
    /// Adds quadratic drag at the center of mass of a convex solid.
    /// The model uses the collider's projected area and a prescribed coefficient.
    pub fn apply(&self, world: &mut PhysicsWorld, handle: BodyHandle) -> Option<Vec3> {
        if !self.density.is_finite()
            || self.density < 0.0
            || !self.drag_coefficient.is_finite()
            || self.drag_coefficient < 0.0
            || !self.wind.is_finite()
        {
            return None;
        }
        let body = world.body(handle)?;
        if body.kind() != BodyKind::Dynamic || body.is_sensor() || !body.is_enabled() {
            return None;
        }
        let relative = body.velocity() - self.wind;
        if relative == Vec3::ZERO {
            return Some(Vec3::ZERO);
        }
        let speed = relative.length();
        let direction = body.rotation().conjugate() * (relative / speed);
        let area = match *body.shape() {
            Shape::Sphere { radius } => PI * radius * radius,
            Shape::Cuboid { half_extents: h } => {
                4.0 * direction
                    .abs()
                    .dot(Vec3::new(h.y * h.z, h.z * h.x, h.x * h.y))
            }
            Shape::Capsule {
                radius,
                half_segment,
            } => {
                let side = Vec3::new(direction.x, 0.0, direction.z).length();
                PI * radius * radius + 4.0 * radius * half_segment * side
            }
            Shape::Cylinder {
                radius,
                half_height,
            } => {
                let side = Vec3::new(direction.x, 0.0, direction.z).length();
                PI * radius * radius * direction.y.abs() + 4.0 * radius * half_height * side
            }
            Shape::Hull(ref hull) => {
                let surface = hull.surface();
                surface
                    .triangles()
                    .iter()
                    .map(|triangle| {
                        let [a, b, c] = triangle.map(|i| surface.vertices()[i]);
                        0.5 * (b - a).cross(c - a).dot(direction).max(0.0)
                    })
                    .sum()
            }
            _ => return None,
        };
        let force = relative * (-0.5 * self.density * self.drag_coefficient * area * speed);
        let center = body.position();
        if !force.is_finite() {
            return None;
        }
        if force != Vec3::ZERO && !world.add_force_at_point(handle, force, center) {
            return None;
        }
        Some(force)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct MutualGravity {
    pub gravitational_constant: f64,
    /// Plummer softening length. Zero gives Newtonian point masses.
    pub softening: f64,
}

impl MutualGravity {
    /// Accumulates pairwise forces for selected live dynamic bodies and returns
    /// their potential energy. Use zero world gravity for an isolated system.
    /// Evaluate at each integration interval. One world substep per call
    /// gives standard semi-implicit Euler for the changing force. Cost is quadratic.
    pub fn apply(&self, world: &mut PhysicsWorld, handles: &[BodyHandle]) -> Option<f64> {
        if !self.gravitational_constant.is_finite()
            || self.gravitational_constant < 0.0
            || !self.softening.is_finite()
            || self.softening < 0.0
        {
            return None;
        }
        let mut seen = BTreeSet::new();
        let bodies: Vec<_> = handles
            .iter()
            .map(|&id| {
                let body = world.body(id)?;
                (body.kind() == BodyKind::Dynamic
                    && body.is_enabled()
                    && !body.is_sensor()
                    && seen.insert(id))
                .then_some(body)
            })
            .collect::<Option<_>>()?;
        let mut forces = vec![Vec3::ZERO; bodies.len()];
        let mut potential = 0.0;
        for (i, first) in bodies.iter().enumerate() {
            for (j, second) in bodies.iter().enumerate().skip(i + 1) {
                let offset = second.position() - first.position();
                let distance_squared = offset.length_squared() + self.softening * self.softening;
                if distance_squared == 0.0 {
                    return None;
                }
                let strength = self.gravitational_constant
                    / first.inverse_mass()
                    / second.inverse_mass()
                    / distance_squared.sqrt();
                potential -= strength;
                let force = offset * (strength / distance_squared);
                forces[i] += force;
                forces[j] -= force;
            }
        }
        if !potential.is_finite()
            || forces
                .iter()
                .zip(&bodies)
                .any(|(&force, body)| !(body.force + force).is_finite())
        {
            return None;
        }
        let centers: Vec<_> = bodies.into_iter().map(|body| body.position()).collect();
        for ((&id, force), center) in handles.iter().zip(forces).zip(centers) {
            if force != Vec3::ZERO {
                world.add_force_at_point(id, force, center);
            }
        }
        Some(potential)
    }
}
