//! Bounding boxes and ray intersection primitives.
use crate::{RigidBody, Shape, Vec3};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    /// Infinite planes have no finite AABB.
    pub fn of(body: &RigidBody) -> Option<Self> {
        let extent = match body.shape {
            Shape::Sphere { radius } => Vec3::splat(radius),
            Shape::Cuboid { half_extents: h } => {
                (body.rotation * Vec3::X).abs() * h.x
                    + (body.rotation * Vec3::Y).abs() * h.y
                    + (body.rotation * Vec3::Z).abs() * h.z
            }
            Shape::Plane => return None,
        };
        Some(Self {
            min: body.position - extent,
            max: body.position + extent,
        })
    }

    pub fn intersects(self, other: Self) -> bool {
        self.min.cmple(other.max).all() && other.min.cmple(self.max).all()
    }
}

pub(crate) fn ray_body(body: &RigidBody, origin: Vec3, direction: Vec3) -> Option<(f64, Vec3)> {
    let relative = origin - body.position;
    match body.shape {
        Shape::Sphere { radius } => {
            let b = relative.dot(direction);
            let c = relative.length_squared() - radius * radius;
            let discriminant = b * b - c;
            if discriminant < 0.0 {
                return None;
            }
            let near = -b - discriminant.sqrt();
            let far = -b + discriminant.sqrt();
            let t = if near >= 0.0 { near } else { far };
            (t >= 0.0).then(|| (t, (relative + direction * t).normalize_or_zero()))
        }
        Shape::Plane => {
            let n = body.rotation * Vec3::Y;
            let denominator = n.dot(direction);
            if denominator == 0.0 {
                return None;
            }
            let t = -relative.dot(n) / denominator;
            (t >= 0.0).then_some((t, n))
        }
        Shape::Cuboid { half_extents: h } => {
            let o = body.rotation.conjugate() * relative;
            let d = body.rotation.conjugate() * direction;
            let mut near = (f64::NEG_INFINITY, Vec3::ZERO);
            let mut far = (f64::INFINITY, Vec3::ZERO);
            for i in 0..3 {
                if d[i] == 0.0 {
                    if o[i].abs() > h[i] {
                        return None;
                    }
                    continue;
                }
                let sign = d[i].signum();
                let n = Vec3::AXES[i] * sign;
                let enter = (-sign * h[i] - o[i]) / d[i];
                let exit = (sign * h[i] - o[i]) / d[i];
                if enter > near.0 {
                    near = (enter, -n);
                }
                if exit < far.0 {
                    far = (exit, n);
                }
            }
            if far.0 < 0.0 || near.0 > far.0 {
                return None;
            }
            let (t, n) = if near.0 >= 0.0 { near } else { far };
            Some((t, body.rotation * n))
        }
    }
}

/// Surface projection, outward normal, and signed distance (negative inside).
pub(crate) fn project(body: &RigidBody, point: Vec3) -> (Vec3, Vec3, f64) {
    let offset = point - body.position;
    match body.shape {
        Shape::Sphere { radius } => {
            let normal = offset.try_normalize().unwrap_or(Vec3::X);
            (
                body.position + normal * radius,
                normal,
                offset.length() - radius,
            )
        }
        Shape::Plane => {
            let normal = body.rotation * Vec3::Y;
            let distance = offset.dot(normal);
            (point - normal * distance, normal, distance)
        }
        Shape::Cuboid { half_extents } => {
            let local = body.rotation.conjugate() * offset;
            let mut closest = local.clamp(-half_extents, half_extents);
            let delta = local - closest;
            let (normal, distance) = if let Some(normal) = delta.try_normalize() {
                (normal, delta.length())
            } else {
                let clearance = half_extents - local.abs();
                let axis = (0..3)
                    .min_by(|&a, &b| clearance[a].total_cmp(&clearance[b]))
                    .unwrap();
                let normal = Vec3::AXES[axis] * local[axis].signum();
                closest[axis] = normal[axis] * half_extents[axis];
                (normal, -clearance[axis])
            };
            (
                body.position + body.rotation * closest,
                body.rotation * normal,
                distance,
            )
        }
    }
}

/// First entering impact along a linear path. The caller handles initial
/// overlaps and supplies a normalized direction.
pub(crate) fn sweep_sphere(
    body: &RigidBody,
    origin: Vec3,
    direction: Vec3,
    radius: f64,
    max_distance: f64,
) -> Option<(f64, Vec3)> {
    let offset = origin - body.position;
    let (distance, normal) = match body.shape {
        Shape::Plane => {
            let n = body.rotation * Vec3::Y;
            let speed = direction.dot(n);
            let gap = offset.dot(n) - radius;
            if speed >= 0.0 || gap < -1e-8 {
                return None;
            }
            (-gap.max(0.0) / speed, n)
        }
        Shape::Sphere { radius: other } => {
            let sum = radius + other;
            let t = sphere_root(offset, direction, sum)?;
            (t, (offset + t * direction).try_normalize()?)
        }
        Shape::Cuboid { half_extents } => {
            let inverse = body.rotation.conjugate();
            let (t, normal) = rounded_box(
                inverse * offset,
                inverse * direction,
                half_extents,
                radius,
                max_distance,
            )?;
            (t, body.rotation * normal)
        }
    };
    (distance.is_finite() && distance <= max_distance).then_some((distance, normal))
}

fn sphere_root(offset: Vec3, direction: Vec3, radius: f64) -> Option<f64> {
    let a = direction.length_squared();
    let b = offset.dot(direction);
    let c = offset.length_squared() - radius * radius;
    if c < -1e-8 * radius * radius || b >= 0.0 {
        return None;
    }
    // The cross product retains small impact parameters for distant starts.
    let d = a * radius * radius - offset.cross(direction).length_squared();
    (d >= 0.0).then(|| c.max(0.0) / (-b + d.sqrt()))
}

fn rounded_box(
    origin: Vec3,
    direction: Vec3,
    h: Vec3,
    radius: f64,
    max_distance: f64,
) -> Option<(f64, Vec3)> {
    // Between crossings of the six box planes, the nearest feature is fixed.
    // On each interval, squared distance to that feature is a quadratic.
    let mut boundaries = [0.0; 8];
    boundaries[1] = max_distance;
    let mut count = 2;
    for axis in 0..3 {
        if direction[axis] == 0.0 {
            continue;
        }
        for sign in [-1.0, 1.0] {
            let t = (sign * h[axis] - origin[axis]) / direction[axis];
            if t > 0.0 && t < max_distance {
                boundaries[count] = t;
                count += 1;
            }
        }
    }
    boundaries[..count].sort_unstable_by(f64::total_cmp);
    for interval in boundaries[..count].windows(2) {
        let [start, end] = [interval[0], interval[1]];
        if start == end {
            continue;
        }
        let initial = origin + start * direction;
        let sample = if end.is_finite() {
            origin + (start * 0.5 + end * 0.5) * direction
        } else {
            // No more plane crossings remain. Moving components are on their
            // outgoing side and constant components retain their initial value.
            Vec3::select(
                direction.cmpgt(Vec3::ZERO),
                Vec3::splat(f64::INFINITY),
                Vec3::select(
                    direction.cmplt(Vec3::ZERO),
                    Vec3::splat(f64::NEG_INFINITY),
                    initial,
                ),
            )
        };
        let fixed = sample.abs().cmpgt(h);
        let feature = sample.clamp(-h, h);
        let offset = Vec3::select(fixed, initial - feature, Vec3::ZERO);
        let approach = Vec3::select(fixed, direction, Vec3::ZERO);
        if let Some(local_t) = sphere_root(offset, approach, radius)
            && local_t <= end - start
        {
            let t = start + local_t;
            let point = origin + t * direction;
            let normal = (point - point.clamp(-h, h)).try_normalize()?;
            return Some((t, normal));
        }
    }
    None
}
