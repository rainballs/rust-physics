//! World query and event accessors.
use super::{
    BodyHandle, CcdImpact, CollisionEvent, Contact, ContactImpulse, JointBreakEvent, JointHandle,
    JointReaction, PhysicsWorld, PointHit, QueryFilter, Ray, RayHit, Shape, Vec3,
};
use crate::RigidBody;
use crate::spatial::{Aabb, SpatialIndex, cast_convex_body, project, ray_body, sphere_hit};

impl PhysicsWorld {
    /// Transitions observed during the last successful step, in substep order.
    pub fn events(&self) -> &[CollisionEvent] {
        &self.events
    }

    /// Joint failures observed during the last successful step.
    pub fn joint_break_events(&self) -> &[JointBreakEvent] {
        &self.joint_breaks
    }

    /// Solved contact responses collected when reporting is enabled.
    pub fn contact_impulses(&self) -> &[ContactImpulse] {
        &self.contact_impulses
    }

    /// Returns the measured reaction from the last completed joint solve.
    pub fn joint_reaction(&self, handle: JointHandle) -> Option<JointReaction> {
        let h = self.previous_h?;
        let rows = self.joint_cache.get(&handle)?;
        if rows.is_empty() {
            return None;
        }
        let (linear, angular) = rows
            .iter()
            .map(|row| row.reaction())
            .fold((Vec3::ZERO, Vec3::ZERO), |(a, b), (c, d)| (a + c, b + d));
        Some(JointReaction {
            force: linear / h,
            torque: angular / h,
        })
    }

    /// Swept impacts sorted by time within the last successful step.
    pub fn ccd_impacts(&self) -> &[CcdImpact] {
        &self.impacts
    }

    /// World-space contacts sampled before the final substep's penetration correction.
    pub fn debug_contacts(&self) -> &[(BodyHandle, BodyHandle, Contact)] {
        &self.debug_contacts
    }

    pub fn spatial_index(&self) -> SpatialIndex<'_> {
        SpatialIndex::new(self)
    }

    pub fn raycast(
        &self,
        origin: Vec3,
        direction: Vec3,
        max_distance: f64,
        filter: QueryFilter,
    ) -> Option<RayHit> {
        let direction = direction.try_normalize()?;
        if !origin.is_finite() || max_distance.is_nan() || max_distance < 0.0 {
            return None;
        }
        self.bodies()
            .filter(|&(id, b)| filter.accepts(id, b))
            .filter_map(|(body, b)| {
                let (distance, normal) = ray_body(b, origin, direction)?;
                (distance.is_finite() && distance <= max_distance).then_some(RayHit {
                    body,
                    normal,
                    distance,
                    position: origin + direction * distance,
                })
            })
            .min_by(|a, b| a.distance.total_cmp(&b.distance).then(a.body.cmp(&b.body)))
    }

    pub fn raycast_batch<'a>(
        &'a self,
        rays: &'a [Ray],
        filter: QueryFilter,
    ) -> impl ExactSizeIterator<Item = Option<RayHit>> + 'a {
        rays.iter()
            .map(move |ray| self.raycast(ray.origin, ray.direction, ray.max_distance, filter))
    }

    /// Finds the nearest collider surface, including the surface of a collider
    /// containing the point. Selection uses absolute surface distance.
    pub fn nearest_surface(&self, point: Vec3, filter: QueryFilter) -> Option<PointHit> {
        if !point.is_finite() {
            return None;
        }
        self.bodies()
            .filter(|&(id, b)| filter.accepts(id, b))
            .map(|(body, b)| {
                let (position, normal, distance) = project(b, point);
                PointHit {
                    body,
                    position,
                    normal,
                    distance,
                }
            })
            .min_by(|a, b| {
                a.distance
                    .abs()
                    .total_cmp(&b.distance.abs())
                    .then(a.body.cmp(&b.body))
            })
    }

    /// Exact sphere overlap against the supported collider shapes.
    pub fn overlap_sphere(
        &self,
        center: Vec3,
        radius: f64,
        filter: QueryFilter,
    ) -> impl Iterator<Item = BodyHandle> + '_ {
        self.bodies().filter_map(move |(id, b)| {
            (radius.is_finite()
                && radius >= 0.0
                && center.is_finite()
                && filter.accepts(id, b)
                && project(b, center).2 <= radius)
                .then_some(id)
        })
    }

    /// Casts a sphere and reports the first target surface hit. Starting
    /// overlaps are returned at zero distance.
    pub fn cast_sphere(
        &self,
        origin: Vec3,
        radius: f64,
        direction: Vec3,
        max_distance: f64,
        filter: QueryFilter,
    ) -> Option<RayHit> {
        let direction = direction.try_normalize()?;
        if !origin.is_finite()
            || !radius.is_finite()
            || radius < 0.0
            || max_distance.is_nan()
            || max_distance < 0.0
        {
            return None;
        }
        if radius == 0.0 {
            return self.raycast(origin, direction, max_distance, filter);
        }
        self.bodies()
            .filter(|&(id, b)| filter.accepts(id, b))
            .filter_map(|(body, b)| sphere_hit(body, b, origin, direction, radius, max_distance))
            .min_by(|a, b| a.distance.total_cmp(&b.distance).then(a.body.cmp(&b.body)))
    }

    /// Translates a finite convex collider from its current pose against
    /// other convex colliders, compounds, planes, and static meshes. Moving
    /// spheres use the specialized sphere-cast paths.
    pub fn cast_shape(
        &self,
        moving: &RigidBody,
        direction: Vec3,
        max_distance: f64,
        filter: QueryFilter,
    ) -> Option<RayHit> {
        let direction = direction.try_normalize()?;
        if max_distance.is_nan() || max_distance < 0.0 {
            return None;
        }
        if let Shape::Sphere { radius } = *moving.shape() {
            return self.cast_sphere(moving.position(), radius, direction, max_distance, filter);
        }
        self.bodies()
            .filter(|&(id, b)| filter.accepts(id, b))
            .filter_map(|(body, target)| {
                let (distance, normal, position) =
                    cast_convex_body(moving, target, direction, max_distance)?;
                Some(RayHit {
                    body,
                    position,
                    normal,
                    distance,
                })
            })
            .min_by(|a, b| a.distance.total_cmp(&b.distance).then(a.body.cmp(&b.body)))
    }

    /// Finite colliders whose bounding boxes intersect the given region.
    pub fn query_aabb(
        &self,
        bounds: Aabb,
        filter: QueryFilter,
    ) -> impl Iterator<Item = BodyHandle> + '_ {
        self.bodies().filter_map(move |(id, b)| {
            (filter.accepts(id, b) && Aabb::of(b).is_some_and(|a| a.intersects(bounds)))
                .then_some(id)
        })
    }
}
