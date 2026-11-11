//! Single-collider rigid bodies with a center of mass at the collider origin.
use crate::compound::{ColliderPart, Compound};
use crate::hull::RigidHull;
use crate::terrain::TriangleMesh;
use crate::{Mat3, Quat, Vec3};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_GEOMETRY: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BodyKind {
    Static,
    Dynamic,
    Kinematic,
}

/// Local geometry centered on the body origin. A plane bounds the solid half-space
/// below its local +Y normal.
#[derive(Clone, Debug, PartialEq)]
pub enum Shape {
    Sphere { radius: f64 },
    Cuboid { half_extents: Vec3 },
    Capsule { radius: f64, half_segment: f64 },
    Cylinder { radius: f64, half_height: f64 },
    Mesh(Arc<TriangleMesh>),
    Hull(Arc<RigidHull>),
    Compound(Arc<Compound>),
    Plane,
}

impl Shape {
    pub fn sphere(radius: f64) -> Self {
        Self::Sphere { radius }
    }

    pub fn cuboid(half_extents: Vec3) -> Self {
        Self::Cuboid { half_extents }
    }

    /// A local-Y line segment of length twice `half_segment`, swept by a sphere.
    pub fn capsule(radius: f64, half_segment: f64) -> Self {
        Self::Capsule {
            radius,
            half_segment,
        }
    }

    /// A solid cylinder centered on the origin with its axis along local Y.
    pub fn cylinder(radius: f64, half_height: f64) -> Self {
        Self::Cylinder {
            radius,
            half_height,
        }
    }

    pub(crate) fn valid(&self) -> bool {
        match *self {
            Self::Sphere { radius } => radius.is_finite() && radius > 0.0,
            Self::Cuboid { half_extents } => {
                half_extents.is_finite() && half_extents.min_element() > 0.0
            }
            Self::Capsule {
                radius,
                half_segment,
            } => {
                radius.is_finite()
                    && radius > 0.0
                    && half_segment.is_finite()
                    && half_segment >= 0.0
            }
            Self::Cylinder {
                radius,
                half_height,
            } => radius.is_finite() && radius > 0.0 && half_height.is_finite() && half_height > 0.0,
            Self::Hull(_) | Self::Compound(_) | Self::Mesh(_) | Self::Plane => true,
        }
    }

    pub(crate) fn inertia(&self, mass: f64) -> Mat3 {
        let diagonal = match *self {
            Self::Sphere { radius } => Vec3::splat(0.4 * mass * radius * radius),
            Self::Cuboid { half_extents: h } => {
                let sq = h * h;
                (sq.yzx() + sq.zxy()) * (mass / 3.0)
            }
            Self::Capsule {
                radius: r,
                half_segment: h,
            } => {
                let cylinder = mass * (h / (h + 2.0 * r / 3.0));
                let caps = mass - cylinder;
                let r2 = r * r;
                let axial = (0.5 * cylinder + 0.4 * caps) * r2;
                let transverse =
                    cylinder * (0.25 * r2 + h * h / 3.0) + caps * (0.4 * r2 + h * h + 0.75 * h * r);
                Vec3::new(transverse, axial, transverse)
            }
            Self::Cylinder {
                radius: r,
                half_height: h,
            } => {
                let transverse = mass * (r * r / 4.0 + h * h / 3.0);
                Vec3::new(transverse, 0.5 * mass * r * r, transverse)
            }
            Self::Hull(ref hull) => return hull.inertia(mass),
            Self::Compound(ref compound) => return compound.inertia(mass),
            Self::Mesh(_) | Self::Plane => Vec3::ZERO,
        };
        Mat3::from_diagonal(diagonal)
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
    pub(crate) geometry_id: u64,
    pub(crate) geometry_dirty: bool,
    pub(crate) shape: Shape,
    pub(crate) position: Vec3,
    pub(crate) rotation: Quat,
    pub(crate) velocity: Vec3,
    pub(crate) angular_momentum: Vec3,
    pub(crate) kinematic_spin: Vec3,
    pub(crate) force: Vec3,
    pub(crate) torque: Vec3,
    pub(crate) inverse_mass: f64,
    inertia: Mat3,
    pub(crate) custom_inertia: bool,
    inverse_inertia: Mat3,
    constrained_inverse_inertia: Mat3,
    pub(crate) linear_axes: Vec3,
    pub(crate) angular_axes: Vec3,
    pub(crate) material: Material,
    pub(crate) damping: [f64; 2],
    pub(crate) gravity_scale: f64,
    pub(crate) layer: u32,
    pub(crate) mask: u32,
    pub(crate) sensor: bool,
    pub(crate) sleep_time: f64,
    pub(crate) asleep: bool,
    pub(crate) enabled: bool,
    pub(crate) ccd: bool,
}

/// Instantaneous acceleration from the current external loads and damping.
#[derive(Clone, Copy, Debug, Default)]
pub struct Acceleration {
    pub linear: Vec3,
    pub angular: Vec3,
}

impl RigidBody {
    pub fn kind(&self) -> BodyKind {
        self.kind
    }
    pub fn shape(&self) -> &Shape {
        &self.shape
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

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
    pub fn material(&self) -> Material {
        self.material
    }
    pub fn collision_mask(&self) -> u32 {
        self.mask
    }

    /// Returns external acceleration before contact and joint impulses.
    pub fn acceleration(&self, gravity: Vec3) -> Acceleration {
        if self.kind != BodyKind::Dynamic || !self.enabled || self.asleep {
            return Acceleration::default();
        }
        let rotation = Mat3::from_quat(self.rotation);
        let omega = self.angular_velocity();
        let full_momentum = rotation * (self.inertia * (rotation.transpose() * omega));
        Acceleration {
            linear: gravity * self.gravity_scale * self.linear_axes
                + self.apply_inverse_mass(self.force)
                - self.velocity * self.damping[0],
            angular: self.apply_inverse_inertia(
                self.torque - omega.cross(full_momentum) - self.angular_momentum * self.damping[1],
            ),
        }
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
        if self.angular_axes != Vec3::ONE {
            return self.constrained_inverse_inertia;
        }
        let r = Mat3::from_quat(self.rotation);
        r * self.inverse_inertia * r.transpose()
    }

    pub(crate) fn apply_inverse_inertia(&self, v: Vec3) -> Vec3 {
        if self.angular_axes != Vec3::ONE {
            return self.constrained_inverse_inertia * v;
        }
        self.rotation * (self.inverse_inertia * (self.rotation.conjugate() * v))
    }

    pub(crate) fn apply_inverse_mass(&self, impulse: Vec3) -> Vec3 {
        if self.linear_axes == Vec3::ONE {
            impulse * self.inverse_mass
        } else {
            (impulse * self.linear_axes) * self.inverse_mass
        }
    }

    fn refresh_constrained_inertia(&mut self) {
        if self.kind != BodyKind::Dynamic || self.angular_axes == Vec3::ONE {
            return;
        }
        let r = Mat3::from_quat(self.rotation);
        let inertia = r * self.inertia * r.transpose();
        let scale = inertia.x_axis.x.max(inertia.y_axis.y).max(inertia.z_axis.z);
        let p = Mat3::from_diagonal(self.angular_axes);
        let inactive = Mat3::IDENTITY - p;
        // Invert the inertia restricted to the active subspace. The identity
        // on the complement makes the matrix invertible without coupling it.
        self.constrained_inverse_inertia =
            p * (p * (inertia * scale.recip()) * p + inactive).inverse() * p * scale.recip();
    }

    pub fn wake(&mut self) {
        self.asleep = false;
        self.sleep_time = 0.0;
    }

    /// Teleports the body to a finite pose and preserves world-space momenta.
    /// Velocity limits still apply to subsequent integration.
    pub fn set_pose(&mut self, position: Vec3, rotation: Quat) -> bool {
        let length = rotation.length();
        if !position.is_finite() || !length.is_finite() || length <= f64::MIN_POSITIVE {
            return false;
        }
        self.position = position;
        self.rotation = rotation / length;
        self.geometry_dirty = true;
        self.refresh_constrained_inertia();
        self.wake();
        true
    }

    pub fn set_velocity(&mut self, velocity: Vec3) {
        if self.kind != BodyKind::Static && velocity.is_finite() {
            self.velocity = velocity * self.linear_axes;
            self.wake();
        }
    }

    pub fn set_angular_velocity(&mut self, velocity: Vec3) {
        if !velocity.is_finite() {
            return;
        }
        let velocity = velocity * self.angular_axes;
        match self.kind {
            BodyKind::Static => return,
            BodyKind::Kinematic => self.kinematic_spin = velocity,
            BodyKind::Dynamic => {
                let local = self.rotation.conjugate() * velocity;
                self.angular_momentum =
                    (self.rotation * (self.inertia * local)) * self.angular_axes;
            }
        }
        self.wake();
    }

    /// Accumulates force through all substeps of the next public step.
    pub fn add_force(&mut self, force: Vec3) -> bool {
        self.add_force_at_point(force, self.position)
    }

    pub fn add_force_at_point(&mut self, force: Vec3, point: Vec3) -> bool {
        if self.kind != BodyKind::Dynamic
            || !self.enabled
            || !force.is_finite()
            || !point.is_finite()
        {
            return false;
        }
        let accumulated_force = self.force + force;
        let accumulated_torque = self.torque + (point - self.position).cross(force);
        if !accumulated_force.is_finite() || !accumulated_torque.is_finite() {
            return false;
        }
        self.force = accumulated_force;
        self.torque = accumulated_torque;
        self.wake();
        true
    }

    pub fn add_torque(&mut self, torque: Vec3) -> bool {
        let accumulated = self.torque + torque;
        if self.kind != BodyKind::Dynamic || !self.enabled || !accumulated.is_finite() {
            return false;
        }
        self.torque = accumulated;
        self.wake();
        true
    }

    pub fn apply_impulse(&mut self, impulse: Vec3, point: Vec3) -> bool {
        if self.kind != BodyKind::Dynamic
            || !self.enabled
            || !impulse.is_finite()
            || !point.is_finite()
        {
            return false;
        }
        let velocity = self.velocity + self.apply_inverse_mass(impulse);
        let momentum =
            self.angular_momentum + (point - self.position).cross(impulse) * self.angular_axes;
        if !velocity.is_finite() || !momentum.is_finite() {
            return false;
        }
        self.wake();
        self.velocity = velocity;
        self.angular_momentum = momentum;
        true
    }

    pub(crate) fn solver_impulse(&mut self, impulse: Vec3, offset: Vec3) {
        if self.inverse_mass > 0.0 && !self.asleep && self.enabled {
            self.velocity += self.apply_inverse_mass(impulse);
            self.angular_momentum += offset.cross(impulse) * self.angular_axes;
        }
    }

    pub(crate) fn point_velocity(&self, offset: Vec3) -> Vec3 {
        self.velocity + self.angular_velocity().cross(offset)
    }

    /// Geometry-only child transform for contact and query dispatch.
    pub(crate) fn collider_part(&self, part: &ColliderPart) -> Self {
        Self {
            shape: part.shape.clone(),
            position: self.position + self.rotation * part.position,
            rotation: self.rotation * part.rotation,
            ..self.clone()
        }
    }

    /// Pose along the frozen angular-velocity path used by convex CCD.
    pub(crate) fn integrate_swept_pose(&mut self, dt: f64) {
        let spin = self.angular_velocity();
        self.geometry_dirty |= dt != 0.0 && (self.velocity != Vec3::ZERO || spin != Vec3::ZERO);
        self.position += self.velocity * dt;
        if spin != Vec3::ZERO {
            self.rotation = (Quat::from_scaled_axis(spin * dt) * self.rotation).normalize();
            self.refresh_constrained_inertia();
        }
    }

    pub(crate) fn integrate_pose(&mut self, dt: f64) {
        if self.kind == BodyKind::Static || self.asleep || !self.enabled {
            return;
        }
        self.position += self.velocity * dt;
        let spin = self.angular_velocity();
        self.geometry_dirty |= dt != 0.0 && (self.velocity != Vec3::ZERO || spin != Vec3::ZERO);
        if spin == Vec3::ZERO {
            return;
        }
        let original = self.rotation;
        // Midpoint angular velocity for anisotropic inertia and fixed world momentum.
        self.rotation = (Quat::from_scaled_axis(spin * (0.5 * dt)) * original).normalize();
        self.refresh_constrained_inertia();
        self.rotation =
            (Quat::from_scaled_axis(self.angular_velocity() * dt) * original).normalize();
        self.refresh_constrained_inertia();
    }
}

#[derive(Clone, Debug)]
pub struct RigidBodyBuilder {
    body: RigidBody,
    mass: f64,
    spin: Vec3,
    inertia: Option<Mat3>,
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
                geometry_id: 0,
                geometry_dirty: true,
                shape: Shape::sphere(0.5),
                position: Vec3::ZERO,
                rotation: Quat::IDENTITY,
                velocity: Vec3::ZERO,
                angular_momentum: Vec3::ZERO,
                kinematic_spin: Vec3::ZERO,
                force: Vec3::ZERO,
                torque: Vec3::ZERO,
                inverse_mass: 0.0,
                inertia: Mat3::ZERO,
                custom_inertia: false,
                inverse_inertia: Mat3::ZERO,
                constrained_inverse_inertia: Mat3::ZERO,
                linear_axes: Vec3::ONE,
                angular_axes: Vec3::ONE,
                material: Material::default(),
                damping: [0.0, 0.0],
                gravity_scale: 1.0,
                layer: 1,
                mask: u32::MAX,
                sensor: false,
                sleep_time: 0.0,
                asleep: false,
                enabled: true,
                ccd: false,
            },
            mass: 1.0,
            spin: Vec3::ZERO,
            inertia: None,
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
    /// Overrides the local center-of-mass tensor of a dynamic body.
    /// The tensor must be symmetric positive definite and include the mass.
    pub fn local_inertia(mut self, value: Mat3) -> Self {
        self.inertia = Some(value);
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

    /// Selects world-space axes for linear motion. False locks the axis.
    pub fn translation_axes(mut self, enabled: [bool; 3]) -> Self {
        self.body.linear_axes = Vec3::from_array(enabled.map(|v| f64::from(u8::from(v))));
        self
    }

    /// Selects world-space axes for angular motion. False locks the axis.
    pub fn rotation_axes(mut self, enabled: [bool; 3]) -> Self {
        self.body.angular_axes = Vec3::from_array(enabled.map(|v| f64::from(u8::from(v))));
        self
    }

    /// Restricts motion to XY translation and rotation about world Z.
    pub fn planar_xy(self) -> Self {
        self.translation_axes([true, true, false])
            .rotation_axes([false, false, true])
    }

    /// Validates and normalizes the initial state.
    pub fn try_build(self) -> Result<RigidBody, InvalidBody> {
        let mut b = self.body;
        let m = b.material;
        let rotation_length = b.rotation.length();
        if (matches!(b.shape, Shape::Mesh(_)) && b.kind != BodyKind::Static)
            || !b.shape.valid()
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
            if matches!(b.shape, Shape::Plane) {
                return Err(InvalidBody);
            }
            let inertia = self.inertia.unwrap_or_else(|| b.shape.inertia(self.mass));
            let scale = inertia.x_axis.x.max(inertia.y_axis.y).max(inertia.z_axis.z);
            if !self.mass.is_finite()
                || self.mass <= 0.0
                || !self.mass.recip().is_finite()
                || !inertia.is_finite()
                || scale <= 0.0
            {
                return Err(InvalidBody);
            }
            let normalized = inertia * scale.recip();
            let asymmetry = normalized - normalized.transpose();
            if asymmetry
                .x_axis
                .abs()
                .max(asymmetry.y_axis.abs())
                .max(asymmetry.z_axis.abs())
                .max_element()
                > 1e-12
                || normalized.x_axis.x <= 0.0
                || normalized.x_axis.x * normalized.y_axis.y - normalized.x_axis.y.powi(2) <= 0.0
                || normalized.determinant() <= 0.0
            {
                return Err(InvalidBody);
            }
            let inverse = normalized.inverse() * scale.recip();
            if !inverse.is_finite() {
                return Err(InvalidBody);
            }
            b.inverse_mass = self.mass.recip();
            b.inertia = inertia;
            b.custom_inertia = self.inertia.is_some();
            b.inverse_inertia = inverse;
        } else if b.kind == BodyKind::Static {
            b.velocity = Vec3::ZERO;
        }
        b.velocity *= b.linear_axes;
        b.refresh_constrained_inertia();
        b.set_angular_velocity(self.spin);
        if !b.angular_momentum.is_finite() || !b.constrained_inverse_inertia.is_finite() {
            return Err(InvalidBody);
        }
        b.geometry_id = NEXT_GEOMETRY
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .expect("collider identifiers exhausted");
        Ok(b)
    }

    /// Builds a body, panicking on invalid configuration. Use [Self::try_build]
    /// for geometry or settings supplied at runtime.
    pub fn build(self) -> RigidBody {
        self.try_build().expect("invalid rigid body")
    }
}
