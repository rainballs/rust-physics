//! Compound rigid geometry with child transforms and analytical mass properties.
use crate::spatial::Aabb;
use crate::{Mat3, Quat, RigidBodyBuilder, Shape, Vec3};

/// A primitive child and its mass contribution, before recentering.
#[derive(Clone, Debug, PartialEq)]
pub struct ColliderPart {
    pub shape: Shape,
    pub position: Vec3,
    pub rotation: Quat,
    pub weight: f64,
}

#[derive(Debug, PartialEq)]
pub struct Compound {
    parts: Box<[ColliderPart]>,
    bounds: Aabb,
    inertia: Mat3,
    input_center: Vec3,
}

impl Compound {
    /// Builds compound geometry from finite primitives. Part weights determine
    /// mass fractions. The constructor recenters geometry at its center of mass.
    /// If parts overlap, their mass contributions still add.
    pub fn new(mut parts: Vec<ColliderPart>) -> Option<Self> {
        if parts.is_empty() {
            return None;
        }
        let mut total_weight = 0.0;
        let mut center = Vec3::ZERO;
        for part in &mut parts {
            let norm = part.rotation.length();
            if !matches!(
                part.shape,
                Shape::Sphere { .. }
                    | Shape::Cuboid { .. }
                    | Shape::Capsule { .. }
                    | Shape::Cylinder { .. }
                    | Shape::Hull(_)
            ) || !part.shape.valid()
                || !part.position.is_finite()
                || !part.weight.is_finite()
                || part.weight <= 0.0
                || !norm.is_finite()
                || norm <= f64::MIN_POSITIVE
            {
                return None;
            }
            part.rotation /= norm;
            total_weight += part.weight;
            center += part.position * part.weight;
        }
        center /= total_weight;
        if !center.is_finite() || !total_weight.is_finite() {
            return None;
        }
        let mut bounds: Option<Aabb> = None;
        let mut inertia = Mat3::ZERO;
        for part in &mut parts {
            part.position -= center;
            let mass = part.weight / total_weight;
            let r = Mat3::from_quat(part.rotation);
            let d = part.position;
            let outer = Mat3::from_cols(d * d.x, d * d.y, d * d.z);
            inertia += r * part.shape.inertia(mass) * r.transpose()
                + (Mat3::IDENTITY * d.length_squared() - outer) * mass;
            let body = RigidBodyBuilder::static_body()
                .shape(part.shape.clone())
                .position(part.position)
                .rotation(part.rotation)
                .try_build()
                .ok()?;
            let local = Aabb::of(&body)?;
            bounds = Some(bounds.map_or(local, |previous| previous.union(local)));
        }
        if !inertia.is_finite() {
            return None;
        }
        Some(Self {
            parts: parts.into_boxed_slice(),
            bounds: bounds?,
            inertia,
            input_center: center,
        })
    }

    /// Recentered parts in input order.
    pub fn parts(&self) -> &[ColliderPart] {
        &self.parts
    }

    /// Center of mass in the original input frame.
    pub fn input_center(&self) -> Vec3 {
        self.input_center
    }

    pub fn bounds(&self) -> Aabb {
        self.bounds
    }

    pub(crate) fn inertia(&self, mass: f64) -> Mat3 {
        self.inertia * mass
    }
}
