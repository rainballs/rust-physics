//! Analytical spherical-cap buoyancy in a planar fluid region.
use crate::{BodyHandle, BodyKind, PhysicsWorld, Shape, Vec3};

#[derive(Clone, Copy, Debug)]
pub struct BuoyancyPlane {
    /// A point on the free surface.
    pub point: Vec3,
    /// Unit normal pointing out of the fluid.
    pub up: Vec3,
    pub density: f64,
    /// Linear drag coefficient per displaced unit volume.
    pub drag: f64,
    pub fluid_velocity: Vec3,
}

#[derive(Clone, Copy, Debug)]
pub struct BuoyancyLoad {
    pub displaced_volume: f64,
    pub center: Vec3,
    pub force: Vec3,
}

impl BuoyancyPlane {
    /// Computes and accumulates buoyancy and drag for a live dynamic sphere.
    /// Buoyancy is opposite world gravity. Drag is a linear approximation
    /// evaluated at the center of the submerged spherical cap.
    pub fn apply_sphere(
        &self,
        world: &mut PhysicsWorld,
        handle: BodyHandle,
    ) -> Option<BuoyancyLoad> {
        if !self.point.is_finite()
            || !self.up.is_normalized()
            || !self.fluid_velocity.is_finite()
            || !self.density.is_finite()
            || self.density < 0.0
            || !self.drag.is_finite()
            || self.drag < 0.0
        {
            return None;
        }
        let body = world.body(handle)?;
        if body.kind() != BodyKind::Dynamic || body.is_sensor() || !body.is_enabled() {
            return None;
        }
        let Shape::Sphere { radius } = *body.shape() else {
            return None;
        };
        let signed = (body.position() - self.point).dot(self.up);
        let cap = (radius - signed).clamp(0.0, 2.0 * radius);
        if cap == 0.0 {
            return Some(BuoyancyLoad {
                displaced_volume: 0.0,
                center: body.position(),
                force: Vec3::ZERO,
            });
        }
        let volume = std::f64::consts::PI * cap * cap * (radius - cap / 3.0);
        // The cap-height factors cancel when dividing its first moment by volume.
        let offset = -0.25 * (2.0 * radius - cap).powi(2) / (radius - cap / 3.0);
        let center = body.position() + self.up * offset;
        let relative_velocity = body.point_velocity(self.up * offset) - self.fluid_velocity;
        let force =
            -volume * (world.settings().gravity * self.density + relative_velocity * self.drag);
        if !volume.is_finite() || !center.is_finite() || !force.is_finite() {
            return None;
        }
        if force != Vec3::ZERO {
            world.add_force_at_point(handle, force, center);
        }
        Some(BuoyancyLoad {
            displaced_volume: volume,
            center,
            force,
        })
    }
}
