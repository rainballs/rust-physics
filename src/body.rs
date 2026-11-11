//! Single-collider rigid bodies with a center of mass at the collider origin.
use crate::{Mat3, Quat, Vec3};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BodyKind {
    Static,
    Dynamic,
    Kinematic,
}

/// Local geometry centered on the body origin. A plane bounds the solid half-space
/// below its local +Y normal.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Shape {
    Sphere { radius: f64 },
    Cuboid { half_extents: Vec3 },
    Plane,
}

impl Shape {
    pub fn sphere(radius: f64) -> Self {
        Self::Sphere { radius }
    }

    pub fn cuboid(half_extents: Vec3) -> Self {
        Self::Cuboid { half_extents }
    }

    fn valid(self) -> bool {
        match self {
            Self::Sphere { radius } => radius.is_finite() && radius > 0.0,
            Self::Cuboid { half_extents } => {
                half_extents.is_finite() && half_extents.min_element() > 0.0
            }
            Self::Plane => true,
        }
    }

    fn inertia(self, mass: f64) -> Vec3 {
        match self {
            Self::Sphere { radius } => Vec3::splat(0.4 * mass * radius * radius),
            Self::Cuboid { half_extents: h } => {
                let sq = h * h;
                (sq.yzx() + sq.zxy()) * (mass / 3.0)
            }
            Self::Plane => Vec3::ZERO,
        }
    }
}

use glam::Vec3Swizzles;

/// Contact properties. Friction coefficients combine by geometric mean,
/// restitution by maximum.
#[derive(Clone, Copy, Debug)]
pub struct Material {
    pub static_friction: f64,
    pub dynamic_friction: f64,
    pub restitution: f64,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            static_friction: 0.6,
            dynamic_friction: 0.45,
            restitution: 0.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct RigidBody {
    pub(crate) kind: BodyKind,
    pub(crate) shape: Shape,
    pub(crate) position: Vec3,
    pub(crate) rotation: Quat,
    pub(crate) velocity: Vec3,
    pub(crate) angular_momentum: Vec3,
    pub(crate) kinematic_spin: Vec3,
    pub(crate) force: Vec3,
    pub(crate) torque: Vec3,
    pub(crate) inverse_mass: f64,
    pub(crate) inverse_inertia: Vec3,
    pub(crate) material: Material,
    pub(crate) damping: [f64; 2],
    pub(crate) gravity_scale: f64,
    pub(crate) layer: u32,
    pub(crate) mask: u32,
    pub(crate) sensor: bool,
    pub(crate) sleep_time: f64,
    pub(crate) asleep: bool,
    pub(crate) ccd: bool,
}

impl RigidBody {
    pub fn kind(&self) -> BodyKind {
        self.kind
    }
    pub fn shape(&self) -> Shape {
        self.shape
    }
    pub fn position(&self) -> Vec3 {
        self.position
    }
    pub fn rotation(&self) -> Quat {
        self.rotation
    }
    pub fn velocity(&self) -> Vec3 {
        self.velocity
    }
    pub fn angular_momentum(&self) -> Vec3 {
        self.angular_momentum
    }
    pub fn inverse_mass(&self) -> f64 {
        self.inverse_mass
    }
    pub fn is_sleeping(&self) -> bool {
        self.asleep
    }
    pub fn is_sensor(&self) -> bool {
        self.sensor
    }
    pub fn layer(&self) -> u32 {
        self.layer
    }

    pub fn angular_velocity(&self) -> Vec3 {
        match self.kind {
            BodyKind::Static => Vec3::ZERO,
            BodyKind::Kinematic => self.kinematic_spin,
            BodyKind::Dynamic if self.angular_momentum == Vec3::ZERO => Vec3::ZERO,
            BodyKind::Dynamic => self.apply_inverse_inertia(self.angular_momentum),
        }
    }

    pub fn world_inverse_inertia(&self) -> Mat3 {
        let r = Mat3::from_quat(self.rotation);
        r * Mat3::from_diagonal(self.inverse_inertia) * r.transpose()
    }

    pub(crate) fn apply_inverse_inertia(&self, v: Vec3) -> Vec3 {
        self.rotation * (self.inverse_inertia * (self.rotation.conjugate() * v))
    }

    pub fn wake(&mut self) {
        self.asleep = false;
        self.sleep_time = 0.0;
    }

    pub fn set_velocity(&mut self, velocity: Vec3) {
        if self.kind != BodyKind::Static && velocity.is_finite() {
            self.velocity = velocity;
            self.wake();
        }
    }

    pub fn set_angular_velocity(&mut self, velocity: Vec3) {
        if !velocity.is_finite() {
            return;
        }
        match self.kind {
            BodyKind::Static => return,
            BodyKind::Kinematic => self.kinematic_spin = velocity,
            BodyKind::Dynamic => {
                let local = self.rotation.conjugate() * velocity;
                self.angular_momentum = self.rotation * (local / self.inverse_inertia);
            }
        }
        self.wake();
    }

    /// Accumulates force through all substeps of the next public step.
    pub fn add_force(&mut self, force: Vec3) {
        self.add_force_at_point(force, self.position);
    }

    pub fn add_force_at_point(&mut self, force: Vec3, point: Vec3) {
        if self.kind == BodyKind::Dynamic && force.is_finite() && point.is_finite() {
            self.force += force;
            self.torque += (point - self.position).cross(force);
            self.wake();
        }
    }

    pub fn add_torque(&mut self, torque: Vec3) {
        if self.kind == BodyKind::Dynamic && torque.is_finite() {
            self.torque += torque;
            self.wake();
        }
    }

    pub fn apply_impulse(&mut self, impulse: Vec3, point: Vec3) {
        if self.kind == BodyKind::Dynamic && impulse.is_finite() && point.is_finite() {
            self.wake();
            self.solver_impulse(impulse, point - self.position);
        }
    }

    pub(crate) fn solver_impulse(&mut self, impulse: Vec3, offset: Vec3) {
        if self.inverse_mass > 0.0 && !self.asleep {
            self.velocity += impulse * self.inverse_mass;
            self.angular_momentum += offset.cross(impulse);
        }
    }

    pub(crate) fn point_velocity(&self, offset: Vec3) -> Vec3 {
        self.velocity + self.angular_velocity().cross(offset)
    }

    pub(crate) fn integrate_pose(&mut self, dt: f64) {
        if self.kind == BodyKind::Static || self.asleep {
            return;
        }
        self.position += self.velocity * dt;
        let spin = self.angular_velocity();
        if spin == Vec3::ZERO {
            return;
        }
        let original = self.rotation;
        // Midpoint angular velocity for anisotropic inertia and fixed world momentum.
        self.rotation = (Quat::from_scaled_axis(spin * (0.5 * dt)) * original).normalize();
        self.rotation =
            (Quat::from_scaled_axis(self.angular_velocity() * dt) * original).normalize();
    }
}

#[derive(Clone, Debug)]
pub struct RigidBodyBuilder {
    body: RigidBody,
    mass: f64,
    spin: Vec3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InvalidBody;

impl std::fmt::Display for InvalidBody {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("invalid body geometry, mass, pose, or material")
    }
}
impl std::error::Error for InvalidBody {}

impl RigidBodyBuilder {
    fn new(kind: BodyKind) -> Self {
        Self {
            body: RigidBody {
                kind,
                shape: Shape::sphere(0.5),
                position: Vec3::ZERO,
                rotation: Quat::IDENTITY,
                velocity: Vec3::ZERO,
                angular_momentum: Vec3::ZERO,
                kinematic_spin: Vec3::ZERO,
                force: Vec3::ZERO,
                torque: Vec3::ZERO,
                inverse_mass: 0.0,
                inverse_inertia: Vec3::ZERO,
                material: Material::default(),
                damping: [0.0, 0.0],
                gravity_scale: 1.0,
                layer: 1,
                mask: u32::MAX,
                sensor: false,
                sleep_time: 0.0,
                asleep: false,
                ccd: false,
            },
            mass: 1.0,
            spin: Vec3::ZERO,
        }
    }

    pub fn dynamic() -> Self {
        Self::new(BodyKind::Dynamic)
    }
    pub fn static_body() -> Self {
        Self::new(BodyKind::Static)
    }
    pub fn kinematic() -> Self {
        Self::new(BodyKind::Kinematic)
    }

    pub fn position(mut self, value: Vec3) -> Self {
        self.body.position = value;
        self
    }
    pub fn rotation(mut self, value: Quat) -> Self {
        self.body.rotation = value;
        self
    }
    pub fn velocity(mut self, value: Vec3) -> Self {
        self.body.velocity = value;
        self
    }
    pub fn angular_velocity(mut self, value: Vec3) -> Self {
        self.spin = value;
        self
    }
    pub fn shape(mut self, value: Shape) -> Self {
        self.body.shape = value;
        self
    }
    pub fn mass(mut self, value: f64) -> Self {
        self.mass = value;
        self
    }
    pub fn material(mut self, value: Material) -> Self {
        self.body.material = value;
        self
    }
    pub fn gravity_scale(mut self, value: f64) -> Self {
        self.body.gravity_scale = value;
        self
    }
    pub fn damping(mut self, linear: f64, angular: f64) -> Self {
        self.body.damping = [linear, angular];
        self
    }
    pub fn filter(mut self, layer: u32, mask: u32) -> Self {
        self.body.layer = layer;
        self.body.mask = mask;
        self
    }
    pub fn sensor(mut self, enabled: bool) -> Self {
        self.body.sensor = enabled;
        self
    }
    pub fn ccd(mut self, enabled: bool) -> Self {
        self.body.ccd = enabled;
        self
    }

    /// Validates and normalizes the initial state.
    pub fn try_build(self) -> Result<RigidBody, InvalidBody> {
        let mut b = self.body;
        let m = b.material;
        let rotation_length = b.rotation.length();
        if !b.shape.valid()
            || !b.position.is_finite()
            || !b.velocity.is_finite()
            || !self.spin.is_finite()
            || !rotation_length.is_finite()
            || rotation_length <= f64::MIN_POSITIVE
            || !b.gravity_scale.is_finite()
            || !b.damping.iter().all(|d| d.is_finite() && *d >= 0.0)
            || !m.static_friction.is_finite()
            || m.static_friction < 0.0
            || !(0.0..=m.static_friction).contains(&m.dynamic_friction)
            || !(0.0..=1.0).contains(&m.restitution)
        {
            return Err(InvalidBody);
        }
        b.rotation /= rotation_length;
        if b.kind == BodyKind::Dynamic {
            let inverse_inertia = b.shape.inertia(self.mass).recip();
            if !self.mass.is_finite()
                || self.mass <= 0.0
                || !self.mass.recip().is_finite()
                || !inverse_inertia.is_finite()
                || inverse_inertia.min_element() <= 0.0
            {
                return Err(InvalidBody);
            }
            b.inverse_mass = self.mass.recip();
            b.inverse_inertia = inverse_inertia;
        } else if b.kind == BodyKind::Static {
            b.velocity = Vec3::ZERO;
        }
        b.set_angular_velocity(self.spin);
        if !b.angular_momentum.is_finite() {
            return Err(InvalidBody);
        }
        Ok(b)
    }

    /// Builds a body, panicking on invalid configuration. Use [Self::try_build]
    /// for geometry or settings supplied at runtime.
    pub fn build(self) -> RigidBody {
        self.try_build().expect("invalid rigid body")
    }
}
