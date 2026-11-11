//! Rigid convex polyhedra from closed, outward-oriented triangle surfaces.
use crate::terrain::{TriangleMesh, triangle_normal};
use crate::{Mat3, Vec3};
use std::collections::BTreeMap;

#[derive(Debug, PartialEq)]
pub struct RigidHull {
    surface: TriangleMesh,
    planes: Box<[(Vec3, f64)]>,
    input_center: Vec3,
    inertia: Mat3,
}

impl RigidHull {
    /// Validates a closed convex triangle surface, computes its uniform
    /// solid mass properties, and recenters it at its volume centroid.
    pub fn new(vertices: Vec<Vec3>, triangles: Vec<[usize; 3]>) -> Option<Self> {
        if vertices.len() < 4 || triangles.len() < 4 || vertices.iter().any(|p| !p.is_finite()) {
            return None;
        }
        let reference = vertices
            .iter()
            .map(|p| p / vertices.len() as f64)
            .sum::<Vec3>();
        let scale = vertices
            .iter()
            .map(|p| p.distance(reference))
            .fold(0.0, f64::max);
        if !scale.is_finite() || scale <= 0.0 {
            return None;
        }

        // Normalize around an interior reference point while accumulating
        // integrals. Volume and second moment use different length powers.
        let points: Vec<_> = vertices.iter().map(|p| (p - reference) / scale).collect();
        let mut topology: BTreeMap<(usize, usize), (u8, i8)> = BTreeMap::new();
        let mut volume = 0.0;
        let mut first = Vec3::ZERO;
        let mut second = Mat3::ZERO;
        let mut planes = Vec::with_capacity(triangles.len());
        for &[a, b, c] in &triangles {
            let [x, y, z] = [*points.get(a)?, *points.get(b)?, *points.get(c)?];
            let n = (y - x).cross(z - x).try_normalize()?;
            let offset = n.dot(x);
            if offset <= 1e-10 || points.iter().any(|&p| n.dot(p) > offset + 1e-9) {
                return None;
            }
            for (u, v) in [(a, b), (b, c), (c, a)] {
                let (edge, sign) = if u < v { ((u, v), 1) } else { ((v, u), -1) };
                let (count, winding) = topology.entry(edge).or_default();
                if *count == 2 {
                    return None;
                }
                *count += 1;
                *winding += sign;
            }
            let dv = x.dot(y.cross(z)) / 6.0;
            if dv <= 0.0 {
                return None;
            }
            let sum = x + y + z;
            volume += dv;
            first += sum * (dv / 4.0);
            second += (outer(sum) + outer(x) + outer(y) + outer(z)) * (dv / 20.0);
        }
        if topology
            .values()
            .any(|&(count, winding)| count != 2 || winding != 0)
        {
            return None;
        }
        let center = first / volume;
        let covariance = second * volume.recip() - outer(center);
        let trace = covariance.x_axis.x + covariance.y_axis.y + covariance.z_axis.z;
        let inertia = (Mat3::IDENTITY * trace - covariance) * (scale * scale);
        let input_center = reference + center * scale;
        if !inertia.is_finite() || !input_center.is_finite() {
            return None;
        }
        let vertices: Vec<_> = points.iter().map(|p| (p - center) * scale).collect();
        for face in &triangles {
            let triangle = face.map(|i| vertices[i]);
            let normal = triangle_normal(triangle);
            planes.push((normal, normal.dot(triangle[0])));
        }
        Some(Self {
            surface: TriangleMesh::new(vertices, triangles)?,
            planes: planes.into_boxed_slice(),
            input_center,
            inertia,
        })
    }

    /// Volume centroid before recentering.
    pub fn input_center(&self) -> Vec3 {
        self.input_center
    }

    pub fn surface(&self) -> &TriangleMesh {
        &self.surface
    }

    pub(crate) fn inertia(&self, mass: f64) -> Mat3 {
        self.inertia * mass
    }

    pub(crate) fn project(&self, point: Vec3) -> (Vec3, Vec3, f64) {
        let (surface, normal, distance) = self.surface.project(point);
        if distance > 0.0 && self.planes.iter().all(|&(n, d)| n.dot(point) <= d) {
            (surface, -normal, -distance)
        } else {
            (surface, normal, distance)
        }
    }

    pub(crate) fn ray(&self, origin: Vec3, direction: Vec3) -> Option<(f64, Vec3)> {
        let mut near = (f64::NEG_INFINITY, Vec3::ZERO);
        let mut far = (f64::INFINITY, Vec3::ZERO);
        for &(normal, offset) in &self.planes {
            let numerator = offset - normal.dot(origin);
            let denominator = normal.dot(direction);
            if denominator == 0.0 {
                if numerator < 0.0 {
                    return None;
                }
                continue;
            }
            let distance = numerator / denominator;
            if denominator < 0.0 && distance > near.0 {
                near = (distance, normal);
            }
            if denominator > 0.0 && distance < far.0 {
                far = (distance, normal);
            }
            if near.0 > far.0 {
                return None;
            }
        }
        if near.0 >= 0.0 {
            Some(near)
        } else if far.0 >= 0.0 {
            Some(far)
        } else {
            None
        }
    }
}

fn outer(v: Vec3) -> Mat3 {
    Mat3::from_cols(v * v.x, v * v.y, v * v.z)
}
