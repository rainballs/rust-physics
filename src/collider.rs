//! Stable read views for rigid colliders and compound children.
use crate::spatial::ray_body;
use crate::{BodyHandle, Material, PhysicsWorld, Quat, QueryFilter, Shape, Vec3};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ColliderHandle {
    body: BodyHandle,
    geometry: u64,
    part: Option<usize>,
}

impl ColliderHandle {
    pub fn body(self) -> BodyHandle {
        self.body
    }
    pub fn part(self) -> Option<usize> {
        self.part
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ColliderView<'a> {
    pub handle: ColliderHandle,
    pub shape: &'a Shape,
    pub position: Vec3,
    pub rotation: Quat,
    pub material: Material,
    pub layer: u32,
    pub mask: u32,
    pub sensor: bool,
    pub enabled: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct ColliderRayHit {
    pub collider: ColliderHandle,
    pub position: Vec3,
    pub normal: Vec3,
    pub distance: f64,
}

impl PhysicsWorld {
    /// Returns the root collider and any compound children in input order.
    pub fn colliders(&self, body: BodyHandle) -> impl Iterator<Item = ColliderHandle> + '_ {
        self.body(body).into_iter().flat_map(move |b| {
            let count = match b.shape() {
                Shape::Compound(parts) => parts.parts().len(),
                _ => 0,
            };
            std::iter::once(None)
                .chain((0..count).map(Some))
                .map(move |part| ColliderHandle {
                    body,
                    geometry: b.geometry_id,
                    part,
                })
        })
    }

    /// Resolves a collider only while its body and geometry are still live.
    pub fn collider(&self, handle: ColliderHandle) -> Option<ColliderView<'_>> {
        let body = self.body(handle.body)?;
        if body.geometry_id != handle.geometry {
            return None;
        }
        let (shape, position, rotation) = if let Some(i) = handle.part {
            let Shape::Compound(parts) = body.shape() else {
                return None;
            };
            let part = parts.parts().get(i)?;
            (
                &part.shape,
                body.position() + body.rotation() * part.position,
                body.rotation() * part.rotation,
            )
        } else {
            (body.shape(), body.position(), body.rotation())
        };
        Some(ColliderView {
            handle,
            shape,
            position,
            rotation,
            material: body.material(),
            layer: body.layer(),
            mask: body.collision_mask(),
            sensor: body.is_sensor(),
            enabled: body.is_enabled(),
        })
    }

    /// Raycasts individual child surfaces. For a compound union surface
    /// with merged inside intervals, use [Self::raycast].
    pub fn raycast_collider(
        &self,
        origin: Vec3,
        direction: Vec3,
        max_distance: f64,
        filter: QueryFilter,
    ) -> Option<ColliderRayHit> {
        let direction = direction.try_normalize()?;
        if !origin.is_finite() || max_distance.is_nan() || max_distance < 0.0 {
            return None;
        }
        let mut best: Option<ColliderRayHit> = None;
        let mut consider = |collider: ColliderHandle, geometry: &crate::RigidBody| {
            if let Some((distance, normal)) = ray_body(geometry, origin, direction)
                && distance.is_finite()
                && distance <= max_distance
                && best.is_none_or(|b| {
                    distance
                        .total_cmp(&b.distance)
                        .then(collider.cmp(&b.collider))
                        .is_lt()
                })
            {
                best = Some(ColliderRayHit {
                    collider,
                    position: origin + direction * distance,
                    normal,
                    distance,
                });
            }
        };
        for (body, b) in self.bodies().filter(|&(id, b)| filter.accepts(id, b)) {
            let geometry = b.geometry_id;
            match b.shape() {
                Shape::Compound(parts) => {
                    for (i, part) in parts.parts().iter().enumerate() {
                        consider(
                            ColliderHandle {
                                body,
                                geometry,
                                part: Some(i),
                            },
                            &b.collider_part(part),
                        );
                    }
                }
                _ => consider(
                    ColliderHandle {
                        body,
                        geometry,
                        part: None,
                    },
                    b,
                ),
            }
        }
        best
    }
}
