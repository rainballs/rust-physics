//! Experimental conservative advancement for moving convex rigid geometry.
use crate::convex::{self, ConvexDistance, PrimitiveSupport, SupportMap, TriangleSupport};
use crate::spatial::Aabb;
use crate::{Quat, RigidBody, Shape, Vec3};

pub(crate) struct MotionHit {
    pub time: f64,
    pub point: Vec3,
    pub normal: Vec3,
    pub separation: f64,
}

struct Motion<'a> {
    body: &'a RigidBody,
    max_time: f64,
    radius: f64,
    spin: Vec3,
    tolerance: f64,
}

#[derive(Clone, Copy)]
struct Gap {
    distance: f64,
    normal: Vec3,
    point: Vec3,
}

pub(crate) fn cast(body: &RigidBody, target: &RigidBody, max_time: f64) -> Option<MotionHit> {
    PrimitiveSupport::new(body)?;
    let bounds = Aabb::of(body)?;
    let radius = (bounds.min - body.position)
        .abs()
        .max((bounds.max - body.position).abs())
        .length();
    let spin = body.angular_velocity();
    let angular_motion = radius * spin.length();
    let smallest = (bounds.max - bounds.min).min_element();
    if (body.velocity.length() + angular_motion) * max_time <= smallest * 0.05 {
        return None;
    }
    Motion {
        body,
        max_time,
        radius,
        spin,
        tolerance: radius * 1e-6,
    }
    .cast_target(target)
}

impl Motion<'_> {
    fn cast_target(&self, target: &RigidBody) -> Option<MotionHit> {
        match target.shape {
            Shape::Plane => {
                let normal = target.rotation * Vec3::Y;
                self.advance(|moving| {
                    let point = moving.feature_center(-normal);
                    let distance = normal.dot(point - target.position);
                    Some(Gap {
                        distance,
                        normal,
                        point: point - normal * distance,
                    })
                })
            }
            Shape::Compound(ref compound) => compound
                .parts()
                .iter()
                .filter_map(|part| self.cast_target(&target.collider_part(part)))
                .min_by(|a, b| a.time.total_cmp(&b.time)),
            Shape::Mesh(ref mesh) => {
                let inverse = target.rotation.conjugate();
                let origin = inverse * (self.body.position - target.position);
                let end = origin + inverse * (self.body.velocity * self.max_time);
                let bounds = Aabb {
                    min: origin.min(end),
                    max: origin.max(end),
                }
                .expanded(self.radius);
                let mut best = None;
                mesh.visit_triangles(bounds, |triangle| {
                    let triangle =
                        TriangleSupport(triangle.map(|p| target.position + target.rotation * p));
                    if let Some(hit) = self.cast_support(&triangle)
                        && best
                            .as_ref()
                            .is_none_or(|previous: &MotionHit| hit.time < previous.time)
                    {
                        best = Some(hit);
                    }
                });
                best
            }
            _ => {
                let end = self.body.position + self.body.velocity * self.max_time;
                let swept = Aabb {
                    min: self.body.position.min(end),
                    max: self.body.position.max(end),
                }
                .expanded(self.radius);
                if !Aabb::of(target)?.intersects(swept) {
                    return None;
                }
                self.cast_support(&PrimitiveSupport::new(target)?)
            }
        }
    }

    fn cast_support(&self, target: &impl SupportMap) -> Option<MotionHit> {
        self.advance(
            |moving| match convex::distance(target, moving, self.tolerance * 0.1) {
                ConvexDistance::Separated {
                    distance,
                    normal,
                    point_a,
                    ..
                } => Some(Gap {
                    distance,
                    normal,
                    point: point_a,
                }),
                ConvexDistance::Intersecting => Some(Gap {
                    distance: f64::NEG_INFINITY,
                    normal: Vec3::ZERO,
                    point: Vec3::ZERO,
                }),
                ConvexDistance::Indeterminate => None,
            },
        )
    }

    fn advance(
        &self,
        mut measure: impl FnMut(&PrimitiveSupport<'_>) -> Option<Gap>,
    ) -> Option<MotionHit> {
        let mut time = 0.0;
        let mut previous = None;
        let support = PrimitiveSupport::new(self.body)?;
        let velocity = self.body.velocity;
        for _ in 0..64 {
            let position = self.body.position + velocity * time;
            let rotation = if self.spin == Vec3::ZERO {
                self.body.rotation
            } else {
                (Quat::from_scaled_axis(self.spin * time) * self.body.rotation).normalize()
            };
            let gap = measure(&support.at_pose(position, rotation))?;
            if time == 0.0 && gap.distance < -self.tolerance {
                return None; // Existing overlap belongs to the discrete solver.
            }
            if gap.distance <= self.tolerance {
                let Gap { normal, point, .. } = if gap.normal == Vec3::ZERO {
                    previous?
                } else {
                    gap
                };
                let approach = -(velocity + self.spin.cross(point - position)).dot(normal);
                if approach <= 0.0 {
                    let angular_bound = self.radius * self.spin.cross(normal).length();
                    if angular_bound == 0.0 {
                        return None;
                    }
                    time += (4.0 * self.tolerance / (angular_bound + velocity.length()))
                        .max(self.max_time / 64.0);
                    if time > self.max_time {
                        return None;
                    }
                    continue;
                }
                return Some(MotionHit {
                    time,
                    normal,
                    point,
                    separation: self.tolerance * 0.1,
                });
            }
            let angular_bound = self.radius * self.spin.cross(gap.normal).length();
            let speed_bound = -velocity.dot(gap.normal) + angular_bound;
            if speed_bound <= 0.0 {
                return None;
            }
            let advance = (gap.distance - self.tolerance * 0.25) / speed_bound;
            let next = time + advance;
            if !next.is_finite() || next > self.max_time || next == time {
                return None;
            }
            previous = Some(gap);
            time = next;
        }
        None
    }
}
