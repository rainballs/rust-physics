//! Experimental conservative advancement along linear and frozen-spin paths.
use crate::convex::{self, ConvexDistance, PrimitiveSupport, SupportMap, TriangleSupport};
use crate::spatial::Aabb;
use crate::{BodyKind, Quat, RigidBody, Shape, Vec3};

pub(crate) struct MotionHit {
    pub time: f64,
    pub point: Vec3,
    pub normal: Vec3,
    pub separation: f64,
}

#[derive(Clone, Copy)]
struct Motion<'a> {
    support: PrimitiveSupport<'a>,
    position: Vec3,
    rotation: Quat,
    velocity: Vec3,
    spin: Vec3,
    radius: f64,
    smallest_extent: f64,
}

impl<'a> Motion<'a> {
    fn new(body: &'a RigidBody) -> Option<Self> {
        let bounds = Aabb::of(body)?;
        Some(Self {
            support: PrimitiveSupport::new(body)?,
            position: body.position,
            rotation: body.rotation,
            velocity: body.velocity,
            spin: body.angular_velocity(),
            radius: (bounds.min - body.position)
                .abs()
                .max((bounds.max - body.position).abs())
                .length(),
            smallest_extent: (bounds.max - bounds.min).min_element(),
        })
    }

    fn center(self, time: f64) -> Vec3 {
        self.position + self.velocity * time
    }

    fn sample(self, time: f64) -> PrimitiveSupport<'a> {
        let rotation = if self.spin == Vec3::ZERO {
            self.rotation
        } else {
            (Quat::from_scaled_axis(self.spin * time) * self.rotation).normalize()
        };
        self.support.at_pose(self.center(time), rotation)
    }

    fn swept(self, time: f64) -> Aabb {
        let end = self.center(time);
        Aabb {
            min: self.position.min(end),
            max: self.position.max(end),
        }
        .expanded(self.radius)
    }

    fn angular_bound(self, normal: Vec3) -> f64 {
        self.radius * self.spin.cross(normal).length()
    }

    fn velocity_at(self, time: f64, point: Vec3) -> Vec3 {
        self.velocity + self.spin.cross(point - self.center(time))
    }
}

struct Advancement<'a> {
    moving: Motion<'a>,
    max_time: f64,
    tolerance: f64,
}

#[derive(Clone, Copy)]
struct Gap {
    distance: f64,
    normal: Vec3,
    point: Vec3,
}

pub(crate) fn cast(body: &RigidBody, target: &RigidBody, max_time: f64) -> Option<MotionHit> {
    let moving = Motion::new(body)?;
    let other = if target.kind == BodyKind::Static {
        None
    } else {
        Some(Motion::new(target)?)
    };
    let linear = moving.velocity - other.map_or(Vec3::ZERO, |m| m.velocity);
    let angular =
        moving.radius * moving.spin.length() + other.map_or(0.0, |m| m.radius * m.spin.length());
    if (linear.length() + angular) * max_time <= moving.smallest_extent * 0.05 {
        return None;
    }

    let advancement = Advancement {
        moving,
        max_time,
        tolerance: moving.radius * 1e-6,
    };
    if let Some(other) = other {
        if !moving.swept(max_time).intersects(other.swept(max_time)) {
            return None;
        }
        advancement.advance(Some(other), |time, moving| {
            centered_gap(&other.sample(time), moving, advancement.tolerance)
        })
    } else {
        advancement.cast_static(target)
    }
}

impl Advancement<'_> {
    fn cast_static(&self, target: &RigidBody) -> Option<MotionHit> {
        match target.shape {
            Shape::Plane => {
                let normal = target.rotation * Vec3::Y;
                self.advance(None, |_, moving| {
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
                .filter_map(|part| self.cast_static(&target.collider_part(part)))
                .min_by(|a, b| a.time.total_cmp(&b.time)),
            Shape::Mesh(ref mesh) => {
                let inverse = target.rotation.conjugate();
                let origin = inverse * (self.moving.position - target.position);
                let end = origin + inverse * (self.moving.velocity * self.max_time);
                let bounds = Aabb {
                    min: origin.min(end),
                    max: origin.max(end),
                }
                .expanded(self.moving.radius);
                let mut best = None;
                mesh.visit_triangles(bounds, |triangle| {
                    let triangle =
                        TriangleSupport(triangle.map(|p| target.position + target.rotation * p));
                    if let Some(hit) = self.advance(None, |_, moving| {
                        measure_gap(&triangle, moving, self.tolerance)
                    }) && best
                        .as_ref()
                        .is_none_or(|previous: &MotionHit| hit.time < previous.time)
                    {
                        best = Some(hit);
                    }
                });
                best
            }
            _ => {
                if !Aabb::of(target)?.intersects(self.moving.swept(self.max_time)) {
                    return None;
                }
                let support = PrimitiveSupport::new(target)?;
                self.advance(None, |_, moving| {
                    centered_gap(&support, moving, self.tolerance)
                })
            }
        }
    }

    fn advance(
        &self,
        other: Option<Motion<'_>>,
        mut measure: impl FnMut(f64, &PrimitiveSupport<'_>) -> Option<Gap>,
    ) -> Option<MotionHit> {
        let mut time = 0.0;
        let mut previous = None;
        let velocity = self.moving.velocity - other.map_or(Vec3::ZERO, |m| m.velocity);
        let angular_bound = |normal| {
            self.moving.angular_bound(normal) + other.map_or(0.0, |m| m.angular_bound(normal))
        };
        for _ in 0..64 {
            let gap = measure(time, &self.moving.sample(time))?;
            if time == 0.0 && gap.distance < -self.tolerance {
                return None; // Existing overlap belongs to the discrete solver.
            }
            if gap.distance <= self.tolerance {
                let Gap { normal, point, .. } = if gap.normal == Vec3::ZERO {
                    previous?
                } else {
                    gap
                };
                let relative = self.moving.velocity_at(time, point)
                    - other.map_or(Vec3::ZERO, |m| m.velocity_at(time, point));
                if relative.dot(normal) >= 0.0 {
                    let bound = angular_bound(normal);
                    if bound == 0.0 {
                        return None;
                    }
                    time += (4.0 * self.tolerance / (bound + velocity.length()))
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
            let speed_bound = -velocity.dot(gap.normal) + angular_bound(gap.normal);
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

fn centered_gap(
    target: &PrimitiveSupport<'_>,
    moving: &PrimitiveSupport<'_>,
    tolerance: f64,
) -> Option<Gap> {
    let mut gap = measure_gap(target, moving, tolerance)?;
    if gap.normal != Vec3::ZERO {
        let a = target.feature_center(gap.normal);
        let b = moving.feature_center(-gap.normal);
        let offset = b - a;
        let lateral = offset - gap.normal * offset.dot(gap.normal);
        if lateral.length_squared() <= 16.0 * tolerance * tolerance {
            gap.point = a;
        }
    }
    Some(gap)
}

fn measure_gap(target: &impl SupportMap, moving: &impl SupportMap, tolerance: f64) -> Option<Gap> {
    match convex::distance(target, moving, tolerance * 0.1) {
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
    }
}
