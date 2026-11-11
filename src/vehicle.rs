//! Raycast suspension and a load-limited tire-force approximation.
use crate::{BodyHandle, BodyKind, PhysicsWorld, Quat, QueryFilter, RayHit, Vec3};

#[derive(Clone, Copy, Debug)]
pub struct Wheel {
    pub anchor: Vec3,
    pub radius: f64,
    pub rest_length: f64,
    pub stiffness: f64,
    pub damping: f64,
    pub friction: f64,
    pub lateral_stiffness: f64,
    /// Steering is applied about local up when this wheel is steerable.
    pub steerable: bool,
    pub driven: bool,
}

impl Wheel {
    fn valid(self) -> bool {
        self.anchor.is_finite()
            && self.radius.is_finite()
            && self.radius > 0.0
            && [
                self.rest_length,
                self.stiffness,
                self.damping,
                self.friction,
                self.lateral_stiffness,
            ]
            .iter()
            .all(|&v| v.is_finite() && v >= 0.0)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct VehicleInput {
    pub steering_radians: f64,
    /// Requested longitudinal force per driven wheel.
    pub drive_force: f64,
    /// Linear braking coefficient in force per unit speed.
    pub brake: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct WheelContact {
    pub wheel: usize,
    pub hit: RayHit,
    pub compression: f64,
    pub normal_load: f64,
    pub force: Vec3,
}

pub struct RaycastVehicle {
    pub chassis: BodyHandle,
    pub wheels: Vec<Wheel>,
}

impl RaycastVehicle {
    /// Applies forces for the next rigid-world step. Local +Y is chassis up
    /// and local +Z is forward. Dynamic surfaces receive the opposite force.
    /// Wheels are probes and have no separate rigid bodies or angular state.
    pub fn apply(
        &self,
        world: &mut PhysicsWorld,
        input: VehicleInput,
        mut filter: QueryFilter,
    ) -> Option<Vec<WheelContact>> {
        if !input.steering_radians.is_finite()
            || !input.drive_force.is_finite()
            || !input.brake.is_finite()
            || input.brake < 0.0
            || !self.wheels.iter().all(|w| w.valid())
        {
            return None;
        }
        let chassis = world.body(self.chassis)?;
        if chassis.kind() != BodyKind::Dynamic || chassis.is_sensor() || !chassis.is_enabled() {
            return None;
        }
        let rotation = chassis.rotation();
        let up = rotation * Vec3::Y;
        filter.exclude = Some(self.chassis);
        filter.include_sensors = false;
        let mut contacts = Vec::new();
        for (index, wheel) in self.wheels.iter().enumerate() {
            let origin = chassis.position() + rotation * wheel.anchor;
            let Some(hit) = world.raycast(origin, -up, wheel.rest_length + wheel.radius, filter)
            else {
                continue;
            };
            let alignment = hit.normal.dot(up);
            if alignment < 0.2 {
                continue;
            }
            let compression =
                (wheel.rest_length + wheel.radius - hit.distance).clamp(0.0, wheel.rest_length);
            let other = world.body(hit.body).expect("live raycast body");
            let velocity = chassis.point_velocity(hit.position - chassis.position())
                - other.point_velocity(hit.position - other.position());
            // Relative motion along the ray is the normal speed divided by
            // the surface alignment with the suspension axis.
            let load = (wheel.stiffness * compression
                - wheel.damping * velocity.dot(hit.normal) / alignment)
                .max(0.0);
            let steering = if wheel.steerable {
                input.steering_radians
            } else {
                0.0
            };
            let nominal = rotation * (Quat::from_rotation_y(steering) * Vec3::Z);
            let Some(forward) = (nominal - hit.normal * nominal.dot(hit.normal)).try_normalize()
            else {
                continue;
            };
            let lateral = hit.normal.cross(forward);
            let drive = if wheel.driven { input.drive_force } else { 0.0 };
            let friction = forward * (drive - input.brake * velocity.dot(forward))
                - lateral * (wheel.lateral_stiffness * velocity.dot(lateral));
            let force = hit.normal * load + friction.clamp_length_max(wheel.friction * load);
            if force.is_finite() {
                contacts.push(WheelContact {
                    wheel: index,
                    hit,
                    compression,
                    normal_load: load,
                    force,
                });
            }
        }
        for contact in &contacts {
            if contact.force == Vec3::ZERO {
                continue;
            }
            world.add_force_at_point(self.chassis, contact.force, contact.hit.position);
            world.add_force_at_point(contact.hit.body, -contact.force, contact.hit.position);
        }
        Some(contacts)
    }
}
