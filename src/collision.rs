//! Narrow phase for spheres, oriented boxes, and infinite half-spaces.
use crate::{RigidBody, Shape, Vec3};

/// A world-space contact. The normal points in the direction that separates
/// the second body from the first.
#[derive(Clone, Copy, Debug)]
pub struct Contact {
    pub point: Vec3,
    pub normal: Vec3,
    pub depth: f64,
}

/// Emits the current overlap manifold into a caller-owned vector.
pub fn contacts(a: &RigidBody, b: &RigidBody, out: &mut Vec<Contact>) {
    let start = out.len();
    match (a.shape, b.shape) {
        (Shape::Sphere { radius: ar }, Shape::Sphere { radius: br }) => {
            let delta = b.position - a.position;
            let d = delta.length();
            let depth = ar + br - d;
            if depth >= 0.0 {
                let normal = delta.try_normalize().unwrap_or(Vec3::X);
                out.push(Contact {
                    point: a.position + normal * (ar - depth * 0.5),
                    normal,
                    depth,
                });
            }
        }
        (Shape::Plane, Shape::Plane) => {}
        (Shape::Plane, _) => plane_shape(a, b, out),
        (_, Shape::Plane) => {
            plane_shape(b, a, out);
            flip(&mut out[start..]);
        }
        (Shape::Cuboid { half_extents }, Shape::Sphere { radius }) => {
            box_sphere(a, half_extents, b, radius, out);
        }
        (Shape::Sphere { radius }, Shape::Cuboid { half_extents }) => {
            box_sphere(b, half_extents, a, radius, out);
            flip(&mut out[start..]);
        }
        (Shape::Cuboid { half_extents: ah }, Shape::Cuboid { half_extents: bh }) => {
            box_box(Obb::new(a, ah), Obb::new(b, bh), out);
        }
    }
}

fn flip(contacts: &mut [Contact]) {
    for contact in contacts {
        contact.normal = -contact.normal;
    }
}

fn plane_shape(plane: &RigidBody, body: &RigidBody, out: &mut Vec<Contact>) {
    let normal = plane.rotation * Vec3::Y;
    let mut add = |point: Vec3| {
        let depth = (plane.position - point).dot(normal);
        if depth >= 0.0 {
            out.push(Contact {
                point: point + normal * depth * 0.5,
                normal,
                depth,
            });
        }
    };
    match body.shape {
        Shape::Sphere { radius } => add(body.position - normal * radius),
        Shape::Cuboid { half_extents } => {
            let obb = Obb::new(body, half_extents);
            // Test the four corners of the most anti-parallel face.
            let axis = obb.face_axis(normal);
            for p in obb.face(axis, -normal) {
                add(p);
            }
        }
        Shape::Plane => {}
    }
}

fn box_sphere(cube: &RigidBody, h: Vec3, sphere: &RigidBody, radius: f64, out: &mut Vec<Contact>) {
    let center = cube.rotation.conjugate() * (sphere.position - cube.position);
    let mut closest = center.clamp(-h, h);
    let delta = center - closest;
    let (local_normal, depth) = if let Some(normal) = delta.try_normalize() {
        (normal, radius - delta.length())
    } else {
        let distances = h - center.abs();
        let axis = min_axis(distances);
        let normal = Vec3::AXES[axis] * center[axis].signum();
        closest[axis] = normal[axis] * h[axis];
        (normal, radius + distances[axis])
    };
    if depth >= 0.0 {
        out.push(Contact {
            point: cube.position + cube.rotation * (closest - local_normal * depth * 0.5),
            normal: cube.rotation * local_normal,
            depth,
        });
    }
}

fn min_axis(v: Vec3) -> usize {
    if v.x <= v.y && v.x <= v.z {
        0
    } else if v.y <= v.z {
        1
    } else {
        2
    }
}

#[derive(Clone, Copy)]
struct Obb {
    center: Vec3,
    axes: [Vec3; 3],
    h: Vec3,
}

impl Obb {
    fn new(body: &RigidBody, h: Vec3) -> Self {
        Self {
            center: body.position,
            axes: Vec3::AXES.map(|axis| body.rotation * axis),
            h,
        }
    }

    fn projected_radius(self, n: Vec3) -> f64 {
        self.axes
            .iter()
            .zip(self.h.to_array())
            .map(|(axis, h)| axis.dot(n).abs() * h)
            .sum()
    }

    fn face_axis(self, n: Vec3) -> usize {
        min_axis(Vec3::from_array(self.axes.map(|axis| -axis.dot(n).abs())))
    }

    fn face(self, axis: usize, n: Vec3) -> [Vec3; 4] {
        let center =
            self.center + self.axes[axis] * (self.axes[axis].dot(n).signum() * self.h[axis]);
        let j = (axis + 1) % 3;
        let k = (axis + 2) % 3;
        let u = self.axes[j] * self.h[j];
        let v = self.axes[k] * self.h[k];
        [
            center - u - v,
            center + u - v,
            center + u + v,
            center - u + v,
        ]
    }

    fn edge(self, axis: usize, n: Vec3) -> (Vec3, Vec3) {
        let mut center = self.center;
        for i in 0..3 {
            if i != axis {
                center += self.axes[i] * (self.axes[i].dot(n).signum() * self.h[i]);
            }
        }
        let half = self.axes[axis] * self.h[axis];
        (center - half, center + half)
    }
}

#[derive(Clone, Copy)]
enum Feature {
    FaceA(usize),
    FaceB(usize),
    Edges(usize, usize),
}

fn box_box(a: Obb, b: Obb, out: &mut Vec<Contact>) {
    let delta = b.center - a.center;
    let mut best = (f64::INFINITY, Vec3::X, Feature::FaceA(0));
    let mut test = |axis: Vec3, feature: Feature| {
        if axis.length_squared() < 1e-16 {
            return true;
        }
        let axis = axis.normalize();
        let distance = axis.dot(delta);
        let depth = a.projected_radius(axis) + b.projected_radius(axis) - distance.abs();
        if depth < 0.0 {
            return false;
        }
        // Favor a face when axes are numerically equivalent.
        if depth < best.0 - 1e-8 * a.h.max(b.h).max_element() {
            best = (depth, axis * distance.signum(), feature);
        }
        true
    };
    for i in 0..3 {
        if !test(a.axes[i], Feature::FaceA(i)) || !test(b.axes[i], Feature::FaceB(i)) {
            return;
        }
    }
    for i in 0..3 {
        for j in 0..3 {
            if !test(a.axes[i].cross(b.axes[j]), Feature::Edges(i, j)) {
                return;
            }
        }
    }
    let (depth, normal, feature) = best;
    match feature {
        Feature::FaceA(axis) => clip_faces(a, b, axis, normal, normal, out),
        Feature::FaceB(axis) => clip_faces(b, a, axis, -normal, normal, out),
        Feature::Edges(i, j) => {
            let (a0, a1) = a.edge(i, normal);
            let (b0, b1) = b.edge(j, -normal);
            let u = a1 - a0;
            let v = b1 - b0;
            let w = a0 - b0;
            let uv = u.dot(v);
            let uu = u.length_squared();
            let vv = v.length_squared();
            let denominator = u.cross(v).length_squared();
            let s = ((uv * v.dot(w) - vv * u.dot(w)) / denominator).clamp(0.0, 1.0);
            let t = ((uv * s + v.dot(w)) / vv).clamp(0.0, 1.0);
            let s = ((uv * t - u.dot(w)) / uu).clamp(0.0, 1.0);
            out.push(Contact {
                point: (a0 + s * u + b0 + t * v) * 0.5,
                normal,
                depth,
            });
        }
    }
}

fn clip_faces(
    reference: Obb,
    incident: Obb,
    axis: usize,
    n: Vec3,
    normal: Vec3,
    out: &mut Vec<Contact>,
) {
    // A quad clipped against four planes has at most eight vertices.
    let mut polygon = [Vec3::ZERO; 8];
    let mut scratch = [Vec3::ZERO; 8];
    polygon[..4].copy_from_slice(&incident.face(incident.face_axis(n), -n));
    let mut count = 4;
    for side in [(axis + 1) % 3, (axis + 2) % 3] {
        for sign in [-1.0, 1.0] {
            if count == 0 {
                return;
            }
            let plane_normal = reference.axes[side] * sign;
            let distance = |p: Vec3| (p - reference.center).dot(plane_normal) - reference.h[side];
            let mut previous = polygon[count - 1];
            let mut last = distance(previous);
            let mut written = 0;
            for &point in &polygon[..count] {
                let next = distance(point);
                if (last < 0.0 && next > 0.0) || (last > 0.0 && next < 0.0) {
                    scratch[written] = previous.lerp(point, last / (last - next));
                    written += 1;
                }
                if next <= 0.0 {
                    scratch[written] = point;
                    written += 1;
                }
                previous = point;
                last = next;
            }
            count = written;
            std::mem::swap(&mut polygon, &mut scratch);
        }
    }
    for &point in &polygon[..count] {
        let depth = reference.h[axis] - (point - reference.center).dot(n);
        if depth >= 0.0 {
            out.push(Contact {
                point: point + n * (depth * 0.5),
                normal,
                depth,
            });
        }
    }
}
