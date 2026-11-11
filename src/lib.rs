//! Double-precision rigid-body dynamics and experimental particle-based simulation.
#![forbid(unsafe_code)]

pub mod body;
pub mod buoyancy;
pub mod character;
pub mod collider;
pub mod collision;
pub mod compound;
pub mod convex;
pub mod debug;
mod dynamic_tree;
pub mod fields;
pub mod fluid;
pub mod grid;
pub mod hull;
pub mod integrators;
pub mod joints;
mod motion;
pub mod particles;
pub mod planar;
pub mod softbody;
pub mod spatial;
pub mod terrain;
pub mod timestep;
pub mod vehicle;
pub mod world;

pub use body::{Acceleration, BodyKind, Material, RigidBody, RigidBodyBuilder, Shape};
pub use collider::{ColliderHandle, ColliderRayHit, ColliderView};
pub use glam::{DMat2 as Mat2, DMat3 as Mat3, DMat4 as Mat4, DQuat as Quat};
pub use glam::{DVec2 as Vec2, DVec3 as Vec3, DVec4 as Vec4};
pub use joints::{HingeLimits, Joint, JointBreakLimit, JointHandle, JointKind, Motor};
pub use world::{
    BodyHandle, BroadPhase, CcdImpact, JointBreakEvent, NumericalStage, PhysicsWorld, ProfiledStep,
    QueryFilter, Ray, RayHit, Settings, StepError, StepStats, StepTimings, WorldSnapshot,
};
