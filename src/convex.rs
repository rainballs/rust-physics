//! Experimental GJK intersection and EPA penetration for finite support maps.
use crate::{Quat, RigidBody, Shape, Vec3};

/// A finite convex set that can return an extreme point in any unit direction.
pub trait SupportMap {
    fn support(&self, direction: Vec3) -> Vec3;
}

/// Convex hull represented by its input points. Interior points are accepted
/// and contribute to search cost. No face or edge mesh is generated.
#[derive(Clone, Debug)]
pub struct ConvexHull {
    points: Box<[Vec3]>,
}

impl ConvexHull {
    /// Rejects nonfinite points and clouds without resolvable 3D volume.
    pub fn new(points: impl Into<Box<[Vec3]>>) -> Option<Self> {
        let points = points.into();
        if points.len() < 4 || points.iter().any(|v| !v.is_finite()) {
            return None;
        }
        let origin = points[0];
        let farthest = |measure: &dyn Fn(Vec3) -> f64| {
            points
                .iter()
                .map(|&p| p - origin)
                .max_by(|&a, &b| measure(a).total_cmp(&measure(b)))
                .expect("nonempty point cloud")
        };
        let axis = farthest(&Vec3::length_squared);
        let scale = axis.length();
        let axis = axis.try_normalize()?;
        let normal = axis.cross(farthest(&|p| axis.cross(p).length_squared()));
        if normal.length() <= scale * 1e-10 {
            return None;
        }
        let normal = normal.try_normalize()?;
        if farthest(&|p| normal.dot(p).abs()).dot(normal).abs() <= scale * 1e-10 {
            return None;
        }
        Some(Self { points })
    }

    pub fn points(&self) -> &[Vec3] {
        &self.points
    }
}

impl SupportMap for ConvexHull {
    fn support(&self, direction: Vec3) -> Vec3 {
        self.points
            .iter()
            .copied()
            .max_by(|a, b| a.dot(direction).total_cmp(&b.dot(direction)))
            .expect("validated point cloud")
    }
}

/// A rigid transform of an existing support map. Rotation must be a unit quaternion.
pub struct Transformed<'a, T> {
    pub shape: &'a T,
    pub position: Vec3,
    pub rotation: Quat,
}

impl<T: SupportMap> SupportMap for Transformed<'_, T> {
    fn support(&self, direction: Vec3) -> Vec3 {
        self.position + self.rotation * self.shape.support(self.rotation.conjugate() * direction)
    }
}

/// Support-map view of any finite built-in rigid collider.
pub struct PrimitiveSupport<'a>(&'a RigidBody);

impl<'a> PrimitiveSupport<'a> {
    pub fn new(body: &'a RigidBody) -> Option<Self> {
        (body.shape() != Shape::Plane).then_some(Self(body))
    }
}

impl SupportMap for PrimitiveSupport<'_> {
    fn support(&self, direction: Vec3) -> Vec3 {
        let b = self.0;
        let d = b.rotation().conjugate() * direction;
        let p = match b.shape() {
            Shape::Sphere { radius } => d * radius,
            Shape::Capsule {
                radius,
                half_segment,
            } => d * radius + Vec3::Y * (d.y.signum() * half_segment),
            Shape::Cuboid { half_extents } => d.signum() * half_extents,
            Shape::Cylinder {
                radius,
                half_height,
            } => {
                let radial = Vec3::new(d.x, 0.0, d.z).normalize_or_zero();
                radial * radius + Vec3::Y * (d.y.signum() * half_height)
            }
            Shape::Plane => unreachable!("validated finite collider"),
        };
        b.position() + b.rotation() * p
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ConvexContact {
    Separated,
    /// The origin is within tolerance of the Minkowski boundary and no
    /// enclosing tetrahedron was found.
    Touching,
    Penetrating {
        /// Unit direction that separates B from A.
        normal: Vec3,
        depth: f64,
        point_a: Vec3,
        point_b: Vec3,
    },
    /// Invalid arithmetic or failure to converge within the iteration budget.
    Indeterminate,
}

#[derive(Clone, Copy, Default)]
struct Vertex {
    a: Vec3,
    b: Vec3,
}

impl Vertex {
    fn delta(self) -> Vec3 {
        self.a - self.b
    }
}

#[derive(Clone, Copy)]
struct Closest {
    point: Vec3,
    weights: [f64; 4],
}

/// Runs GJK, then EPA if a tetrahedron enclosing the origin can be constructed.
/// The tolerance is a positive world-space distance. A bounded solver can
/// return [ConvexContact::Indeterminate] for poorly conditioned support maps.
pub fn query(a: &impl SupportMap, b: &impl SupportMap, tolerance: f64) -> ConvexContact {
    use ConvexContact::*;
    if !tolerance.is_finite() || tolerance <= 0.0 {
        return Indeterminate;
    }
    let support = |direction: Vec3| {
        let vertex = Vertex {
            a: a.support(direction),
            b: b.support(-direction),
        };
        vertex.delta().is_finite().then_some(vertex)
    };

    let Some(first) = support(Vec3::X) else {
        return Indeterminate;
    };
    let mut simplex = [Vertex::default(); 4];
    simplex[0] = first;
    let mut count = 1;
    for _ in 0..48 {
        let closest = closest_simplex(&simplex[..count]);
        let distance = closest.point.length();
        if distance <= tolerance {
            let seed = if count == 4 && strictly_contains(simplex) {
                Some(simplex)
            } else {
                seed_tetrahedron(&simplex[..count], &support, tolerance)
            };
            return match seed {
                Some(seed) => epa(seed, support, tolerance),
                None => Touching,
            };
        }
        let Some(direction) = (-closest.point).try_normalize() else {
            return Indeterminate;
        };
        let Some(next) = support(direction) else {
            return Indeterminate;
        };
        if next.delta().dot(direction) < -tolerance {
            return Separated;
        }

        // Retain only vertices incident to the nearest simplex feature.
        let old = simplex;
        count = 0;
        for (index, &weight) in closest.weights.iter().enumerate() {
            if weight > 0.0 {
                simplex[count] = old[index];
                count += 1;
            }
        }
        if count == 4
            || simplex[..count]
                .iter()
                .any(|p| p.delta().distance(next.delta()) <= tolerance * 0.01)
        {
            return Indeterminate;
        }
        simplex[count] = next;
        count += 1;
    }
    Indeterminate
}

fn closest_simplex(vertices: &[Vertex]) -> Closest {
    let mut closest = Closest {
        point: vertices[0].delta(),
        weights: [1.0, 0.0, 0.0, 0.0],
    };
    let mut consider = |indices: &[usize], weights: &[f64]| {
        if weights.iter().any(|&w| !w.is_finite() || w < 0.0) {
            return;
        }
        let mut candidate = Closest {
            point: Vec3::ZERO,
            weights: [0.0; 4],
        };
        for (&index, &weight) in indices.iter().zip(weights) {
            candidate.point += vertices[index].delta() * weight;
            candidate.weights[index] = weight;
        }
        if candidate.point.length_squared() < closest.point.length_squared() {
            closest = candidate;
        }
    };
    for i in 0..vertices.len() {
        let a = vertices[i].delta();
        consider(&[i], &[1.0]);
        for j in i + 1..vertices.len() {
            let b = vertices[j].delta();
            let ab = b - a;
            let t = (-a.dot(ab) / ab.length_squared()).clamp(0.0, 1.0);
            consider(&[i, j], &[1.0 - t, t]);
            for (k, vertex) in vertices.iter().enumerate().skip(j + 1) {
                let c = vertex.delta();
                if let Some(weights) = triangle_projection(a, b, c) {
                    consider(&[i, j, k], &weights);
                }
            }
        }
    }
    if vertices.len() == 4 {
        let points = std::array::from_fn(|i| vertices[i].delta());
        if let Some(weights) = tetra_weights(points)
            && weights.iter().all(|&w| w >= 0.0)
        {
            return Closest {
                point: Vec3::ZERO,
                weights,
            };
        }
    }
    closest
}

fn triangle_projection(a: Vec3, b: Vec3, c: Vec3) -> Option<[f64; 3]> {
    let u = b - a;
    let v = c - a;
    let normal = u.cross(v);
    let denominator = normal.length_squared();
    if denominator <= 1e-24 * u.length_squared() * v.length_squared() {
        return None;
    }
    let y = (-a).cross(v).dot(normal) / denominator;
    let z = u.cross(-a).dot(normal) / denominator;
    Some([1.0 - y - z, y, z])
}

fn tetra_weights([a, b, c, d]: [Vec3; 4]) -> Option<[f64; 4]> {
    let [u, v, w] = [b - a, c - a, d - a];
    let det = u.dot(v.cross(w));
    if det.abs() <= 1e-12 * u.length() * v.length() * w.length() {
        return None;
    }
    let y = (-a).dot(v.cross(w)) / det;
    let z = u.dot((-a).cross(w)) / det;
    let t = u.dot(v.cross(-a)) / det;
    Some([1.0 - y - z - t, y, z, t])
}

fn strictly_contains(points: [Vertex; 4]) -> bool {
    tetra_weights(points.map(Vertex::delta))
        .is_some_and(|weights| weights.iter().all(|&w| w > 1e-9))
}

fn seed_tetrahedron(
    simplex: &[Vertex],
    support: &impl Fn(Vec3) -> Option<Vertex>,
    tolerance: f64,
) -> Option<[Vertex; 4]> {
    let mut candidates = Vec::from(simplex);
    let basis = if simplex.len() >= 2 {
        let u = (simplex[1].delta() - simplex[0].delta())
            .try_normalize()
            .unwrap_or(Vec3::X);
        let (v, w) = u.any_orthonormal_pair();
        [u, v, w]
    } else {
        Vec3::AXES
    };
    // Try complete tetrahedra as each new extreme point is discovered.
    for x in -1..=1 {
        for y in -1..=1 {
            for z in -1..=1 {
                let direction =
                    basis[0] * f64::from(x) + basis[1] * f64::from(y) + basis[2] * f64::from(z);
                let Some(unit) = direction.try_normalize() else {
                    continue;
                };
                let vertex = support(unit)?;
                if candidates
                    .iter()
                    .any(|p| p.delta().distance(vertex.delta()) <= tolerance * 0.01)
                {
                    continue;
                }
                for i in 0..candidates.len() {
                    for j in i + 1..candidates.len() {
                        for k in j + 1..candidates.len() {
                            let tetrahedron = [candidates[i], candidates[j], candidates[k], vertex];
                            if strictly_contains(tetrahedron) {
                                return Some(tetrahedron);
                            }
                        }
                    }
                }
                candidates.push(vertex);
            }
        }
    }
    None
}

#[derive(Clone, Copy)]
struct Face {
    vertices: [usize; 3],
    normal: Vec3,
    distance: f64,
}

impl Face {
    fn new(points: &[Vertex], mut vertices: [usize; 3]) -> Option<Self> {
        let [a, b, c] = vertices.map(|i| points[i].delta());
        let mut normal = (b - a).cross(c - a).try_normalize()?;
        let mut distance = normal.dot(a);
        if distance < 0.0 {
            vertices.swap(1, 2);
            normal = -normal;
            distance = -distance;
        }
        Some(Self {
            vertices,
            normal,
            distance,
        })
    }

    fn contact(self, points: &[Vertex], faces: &[Face], tolerance: f64) -> ConvexContact {
        // Triangulated coplanar facets may have multiple triangles at the
        // same plane distance. Select the triangle closest to the origin.
        let target = self.normal * self.distance;
        let (vertices, closest) = faces
            .iter()
            .filter(|face| {
                face.normal.dot(self.normal) >= 1.0 - 1e-10
                    && (face.distance - self.distance).abs() <= tolerance
            })
            .map(|face| {
                let vertices = face.vertices.map(|i| points[i]);
                let closest = closest_simplex(&vertices);
                (vertices, closest)
            })
            .min_by(|(_, a), (_, b)| {
                a.point
                    .distance_squared(target)
                    .total_cmp(&b.point.distance_squared(target))
            })
            .expect("selected face is present");
        let (a, b) = vertices
            .iter()
            .zip(closest.weights)
            .fold((Vec3::ZERO, Vec3::ZERO), |(a, b), (point, weight)| {
                (a + point.a * weight, b + point.b * weight)
            });
        ConvexContact::Penetrating {
            normal: self.normal,
            depth: self.distance,
            point_a: a,
            point_b: b,
        }
    }
}

fn epa(
    seed: [Vertex; 4],
    support: impl Fn(Vec3) -> Option<Vertex>,
    tolerance: f64,
) -> ConvexContact {
    let mut points = Vec::from(seed);
    let mut faces: Vec<_> = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
        .into_iter()
        .filter_map(|triangle| Face::new(&points, triangle))
        .collect();
    let mut border = Vec::new();
    for _ in 0..96 {
        let Some(closest) = faces
            .iter()
            .copied()
            .min_by(|a, b| a.distance.total_cmp(&b.distance))
        else {
            break;
        };
        let Some(vertex) = support(closest.normal) else {
            break;
        };
        let gap = closest.normal.dot(vertex.delta()) - closest.distance;
        if gap <= tolerance {
            return closest.contact(&points, &faces, tolerance);
        }
        if points
            .iter()
            .any(|p| p.delta().distance(vertex.delta()) <= tolerance * 0.01)
        {
            break;
        }
        border.clear();
        faces.retain(|face| {
            if face.normal.dot(vertex.delta()) - face.distance <= tolerance * 0.01 {
                return true;
            }
            let [a, b, c] = face.vertices;
            for edge in [(a, b), (b, c), (c, a)] {
                if let Some(i) = border.iter().position(|&(u, v)| (v, u) == edge) {
                    border.swap_remove(i);
                } else {
                    border.push(edge);
                }
            }
            false
        });
        if border.is_empty() {
            break;
        }
        let next = points.len();
        points.push(vertex);
        faces.extend(
            border
                .iter()
                .filter_map(|&(a, b)| Face::new(&points, [a, b, next])),
        );
    }
    ConvexContact::Indeterminate
}
