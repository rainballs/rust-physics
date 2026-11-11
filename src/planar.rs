//! XY rigid-body facade using planar contacts and world-axis constraints.
use crate::body::InvalidBody;
use crate::{
    BodyHandle, Mat3, Material, PhysicsWorld, Quat, QueryFilter, RigidBody, RigidBodyBuilder,
    Shape, StepError, StepStats, Vec2, Vec3,
};

#[derive(Clone, Copy, Debug)]
pub enum PlanarShape {
    Circle {
        radius: f64,
    },
    Rectangle {
        half_extents: Vec2,
    },
    /// Local +Y points out of the solid half-plane.
    HalfPlane,
}

pub struct PlanarBodyBuilder {
    inner: RigidBodyBuilder,
    shape: PlanarShape,
    mass: f64,
}

impl PlanarBodyBuilder {
    fn new(inner: RigidBodyBuilder, shape: PlanarShape) -> Self {
        Self {
            inner: inner.planar_xy(),
            shape,
            mass: 1.0,
        }
    }

    pub fn dynamic(shape: PlanarShape) -> Self {
        Self::new(RigidBodyBuilder::dynamic(), shape)
    }
    pub fn static_body(shape: PlanarShape) -> Self {
        Self::new(RigidBodyBuilder::static_body(), shape)
    }
    pub fn kinematic(shape: PlanarShape) -> Self {
        Self::new(RigidBodyBuilder::kinematic(), shape)
    }

    pub fn position(mut self, position: Vec2) -> Self {
        self.inner = self.inner.position(position.extend(0.0));
        self
    }

    pub fn angle(mut self, radians: f64) -> Self {
        self.inner = self.inner.rotation(Quat::from_rotation_z(radians));
        self
    }

    pub fn velocity(mut self, velocity: Vec2) -> Self {
        self.inner = self.inner.velocity(velocity.extend(0.0));
        self
    }

    pub fn spin(mut self, radians_per_second: f64) -> Self {
        self.inner = self.inner.angular_velocity(Vec3::Z * radians_per_second);
        self
    }

    pub fn mass(mut self, mass: f64) -> Self {
        self.mass = mass;
        self.inner = self.inner.mass(mass);
        self
    }

    pub fn material(mut self, material: Material) -> Self {
        self.inner = self.inner.material(material);
        self
    }

    pub fn filter(mut self, layer: u32, mask: u32) -> Self {
        self.inner = self.inner.filter(layer, mask);
        self
    }

    pub fn sensor(mut self, sensor: bool) -> Self {
        self.inner = self.inner.sensor(sensor);
        self
    }

    fn build(self) -> Result<RigidBody, InvalidBody> {
        match self.shape {
            PlanarShape::Circle { radius } => {
                let quarter = 0.25 * self.mass * radius * radius;
                self.inner
                    .shape(Shape::sphere(radius))
                    .local_inertia(Mat3::from_diagonal(Vec3::new(
                        quarter,
                        quarter,
                        2.0 * quarter,
                    )))
                    .try_build()
            }
            PlanarShape::Rectangle { half_extents } => {
                let [x, y] = half_extents.to_array();
                let thickness = x.max(y);
                self.inner
                    .shape(Shape::cuboid(half_extents.extend(thickness)))
                    .try_build()
            }
            PlanarShape::HalfPlane => self.inner.shape(Shape::Plane).try_build(),
        }
    }
}

#[derive(Default)]
pub struct PlanarWorld {
    inner: PhysicsWorld,
}

#[derive(Clone, Copy, Debug)]
pub struct PlanarPose {
    pub position: Vec2,
    pub radians: f64,
    pub velocity: Vec2,
    pub spin: f64,
    pub sleeping: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct PlanarHit {
    pub body: BodyHandle,
    pub point: Vec2,
    pub normal: Vec2,
    pub distance: f64,
}

impl PlanarWorld {
    pub fn new() -> Self {
        Self::default()
    }

    /// Access to shared debug drawing and query APIs.
    pub fn rigid_world(&self) -> &PhysicsWorld {
        &self.inner
    }

    pub fn add_body(&mut self, builder: PlanarBodyBuilder) -> Result<BodyHandle, InvalidBody> {
        Ok(self.inner.add_body(builder.build()?))
    }

    pub fn remove_body(&mut self, handle: BodyHandle) -> bool {
        self.inner.remove_body(handle).is_some()
    }

    pub fn pose(&self, handle: BodyHandle) -> Option<PlanarPose> {
        let body = self.inner.body(handle)?;
        let axis = body.rotation() * Vec3::X;
        Some(PlanarPose {
            position: body.position().truncate(),
            radians: axis.y.atan2(axis.x),
            velocity: body.velocity().truncate(),
            spin: body.angular_velocity().z,
            sleeping: body.is_sleeping(),
        })
    }

    pub fn set_gravity(&mut self, gravity: Vec2) -> Result<(), StepError> {
        self.inner.set_gravity(gravity.extend(0.0))
    }

    pub fn settings(&self) -> &crate::Settings {
        self.inner.settings()
    }

    pub fn set_settings(&mut self, settings: crate::Settings) -> Result<(), StepError> {
        if settings.gravity.z != 0.0 {
            return Err(StepError::InvalidSettings);
        }
        self.inner.set_settings(settings)
    }

    pub fn set_pose(&mut self, handle: BodyHandle, position: Vec2, angle: f64) -> bool {
        self.inner
            .body_mut(handle)
            .is_some_and(|body| body.set_pose(position.extend(0.0), Quat::from_rotation_z(angle)))
    }

    pub fn step(&mut self, dt: f64) -> Result<StepStats, StepError> {
        self.inner.step(dt)
    }
    pub fn step_fixed(&mut self) -> Result<StepStats, StepError> {
        self.inner.step_fixed()
    }

    pub fn add_joint(
        &mut self,
        joint: crate::Joint,
    ) -> Result<crate::JointHandle, crate::world::InvalidJoint> {
        self.inner.add_joint(joint)
    }

    pub fn apply_impulse(&mut self, body: BodyHandle, impulse: Vec2, point: Vec2) -> bool {
        self.inner
            .apply_impulse(body, impulse.extend(0.0), point.extend(0.0))
    }

    pub fn add_force_at_point(&mut self, body: BodyHandle, force: Vec2, point: Vec2) -> bool {
        self.inner
            .add_force_at_point(body, force.extend(0.0), point.extend(0.0))
    }

    pub fn raycast(
        &self,
        origin: Vec2,
        direction: Vec2,
        max: f64,
        filter: QueryFilter,
    ) -> Option<PlanarHit> {
        let hit = self
            .inner
            .raycast(origin.extend(0.0), direction.extend(0.0), max, filter)?;
        Some(PlanarHit {
            body: hit.body,
            point: hit.position.truncate(),
            normal: hit.normal.truncate(),
            distance: hit.distance,
        })
    }
}
