//! Experimental GJK intersection and EPA penetration for finite support maps.
use crate::{Quat, RigidBody, Shape, Vec3};

/// A finite convex set that can return an extreme point in any unit direction.
pub trait SupportMap {
    fn support(&self, direction: Vec3) -> Vec3;
}

pub(crate) struct TriangleSupport(pub [Vec3; 3]);

impl SupportMap for TriangleSupport {
    fn support(&self, direction: Vec3) -> Vec3 {
        self.0
            .into_iter()
            .max_by(|a, b| a.dot(direction).total_cmp(&b.dot(direction)))
            .expect("three vertices")
    }
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
#[derive(Clone, Copy)]
pub struct PrimitiveSupport<'a> {
    shape: &'a Shape,
    position: Vec3,
    rotation: Quat,
}

impl<'a> PrimitiveSupport<'a> {
    pub fn new(body: &'a RigidBody) -> Option<Self> {
        (!matches!(
            body.shape(),
            Shape::Plane | Shape::Mesh(_) | Shape::Compound(_)
        ))
        .then_some(Self {
            shape: body.shape(),
            position: body.position(),
            rotation: body.rotation(),
        })
    }

    pub(crate) fn at_pose(self, position: Vec3, rotation: Quat) -> Self {
        Self {
            position,
            rotation,
            ..self
        }
    }
}

impl PrimitiveSupport<'_> {
    pub(crate) fn feature_center(&self, direction: Vec3) -> Vec3 {
        self.support_point(direction, true)
    }

    fn support_point(&self, direction: Vec3, center: bool) -> Vec3 {
        let d = self.rotation.conjugate() * direction;
        let side = |v: f64| {
            if center && v.abs() <= 1e-12 {
                0.0
            } else {
                v.signum()
            }
        };
        let p = match *self.shape {
            Shape::Sphere { radius } => d * radius,
            Shape::Capsule {
                radius,
                half_segment,
            } => d * radius + Vec3::Y * (side(d.y) * half_segment),
            Shape::Cuboid { half_extents } => {
                Vec3::from_array(d.to_array().map(side)) * half_extents
            }
            Shape::Cylinder {
                radius,
                half_height,
            } => {
                let radial = Vec3::new(d.x, 0.0, d.z);
                let radial = if center && radial.length_squared() <= 1e-24 {
                    Vec3::ZERO
                } else {
                    radial.normalize_or_zero()
                };
                radial * radius + Vec3::Y * (side(d.y) * half_height)
            }
            Shape::Hull(ref hull) => {
                let vertices = hull.surface().vertices();
                let extreme = vertices
                    .iter()
                    .copied()
                    .max_by(|a, b| a.dot(d).total_cmp(&b.dot(d)))
                    .expect("validated nonempty hull");
                if center {
                    let bounds = hull.surface().bounds();
                    let tolerance = (bounds.max - bounds.min).max_element() * 1e-12;
                    let maximum = extreme.dot(d);
                    let (sum, count) = vertices
                        .iter()
                        .copied()
                        .filter(|&p| maximum - p.dot(d) <= tolerance)
                        .fold((Vec3::ZERO, 0), |(sum, count), p| (sum + p, count + 1));
                    sum / f64::from(count)
                } else {
                    extreme
                }
            }
            Shape::Plane | Shape::Mesh(_) | Shape::Compound(_) => {
                unreachable!("validated convex collider")
            }
        };
        self.position + self.rotation * p
    }
}

impl SupportMap for PrimitiveSupport<'_> {
    fn support(&self, direction: Vec3) -> Vec3 {
        self.support_point(direction, false)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ConvexContact {
    Separated,
    /// The origin is in the tolerance band around the Minkowski boundary.
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
    let mut last_projection = first.delta().dot(Vec3::X);
    for _ in 0..48 {
        let closest = closest_simplex(&simplex[..count]);
        let distance = closest.point.length();
        if distance <= tolerance {
            if last_projection <= tolerance {
                return Touching;
            }
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
        last_projection = next.delta().dot(direction);
        if last_projection < -tolerance {
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
    if simplex.len() >= 3
        && let Some(normal) = (simplex[1].delta() - simplex[0].delta())
            .cross(simplex[2].delta() - simplex[0].delta())
            .try_normalize()
    {
        for direction in [normal, -normal] {
            if support(direction)?.delta().dot(direction) <= tolerance {
                return None;
            }
        }
    }
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
                if vertex.delta().dot(unit) <= tolerance {
                    return None;
                }
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
        let (a, b) = witnesses(&vertices, closest.weights);
        ConvexContact::Penetrating {
            normal: self.normal,
            depth: self.distance,
            point_a: a,
            point_b: b,
        }
    }
}

const EPA_ITERATIONS: usize = 96;

fn closed_horizon(edges: &[(usize, usize)], vertices: usize) -> bool {
    if edges.len() < 3 {
        return false;
    }
    let mut next = [usize::MAX; EPA_ITERATIONS + 4];
    for &(a, b) in edges {
        if a >= vertices || b >= vertices || next[a] != usize::MAX {
            return false;
        }
        next[a] = b;
    }
    let origin = edges[0].0;
    let mut current = origin;
    for i in 0..edges.len() {
        current = next[current];
        if current == usize::MAX {
            return false;
        }
        if current == origin {
            return i + 1 == edges.len();
        }
    }
    false
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
    if faces.len() != 4 {
        return ConvexContact::Indeterminate;
    }
    let mut border = Vec::new();
    for _ in 0..EPA_ITERATIONS {
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
        // A convex visible patch has exactly one oriented horizon. A
        // triangulated convex hull with V vertices has at most 2V - 4
        // faces. Reject degenerate topology before it can multiply faces.
        let next = points.len();
        if !closed_horizon(&border, next) || faces.len() + border.len() > 2 * (next + 1) - 4 {
            break;
        }
        points.push(vertex);
        for &(a, b) in &border {
            let vertices = [a, b, next];
            let Some(face) = Face::new(&points, vertices) else {
                return ConvexContact::Indeterminate;
            };
            if face.vertices != vertices {
                return ConvexContact::Indeterminate;
            }
            faces.push(face);
        }
    }
    ConvexContact::Indeterminate
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ConvexDistance {
    Separated {
        distance: f64,
        /// Unit direction from the nearest witness on A to the witness on B.
        normal: Vec3,
        point_a: Vec3,
        point_b: Vec3,
    },
    Intersecting,
    Indeterminate,
}

/// Measures separation between support maps using closest-simplex GJK.
/// Distances within tolerance are classified as intersecting.
pub fn distance(a: &impl SupportMap, b: &impl SupportMap, tolerance: f64) -> ConvexDistance {
    use ConvexDistance::*;
    if !tolerance.is_finite() || tolerance <= 0.0 {
        return Indeterminate;
    }
    let support = |direction| Vertex {
        a: a.support(direction),
        b: b.support(-direction),
    };
    let mut simplex = [Vertex::default(); 4];
    simplex[0] = support(Vec3::X);
    let mut count = 1;
    for _ in 0..64 {
        let closest = closest_simplex(&simplex[..count]);
        let distance = closest.point.length();
        if distance <= tolerance {
            return Intersecting;
        }
        if !distance.is_finite() {
            return Indeterminate;
        }
        let normal = -closest.point / distance;
        let next = support(normal);
        let value = next.delta().dot(normal);
        if !value.is_finite() {
            return Indeterminate;
        }
        if value + distance <= tolerance
            || simplex[..count]
                .iter()
                .any(|p| p.delta().distance_squared(next.delta()) <= tolerance * tolerance)
        {
            let (point_a, point_b) = witnesses(&simplex[..count], closest.weights);
            return Separated {
                distance,
                normal,
                point_a,
                point_b,
            };
        }
        let old = simplex;
        count = 0;
        for (i, weight) in closest.weights.into_iter().enumerate() {
            if weight > 0.0 {
                simplex[count] = old[i];
                count += 1;
            }
        }
        if count == 4 {
            return Indeterminate;
        }
        simplex[count] = next;
        count += 1;
    }
    Indeterminate
}

fn witnesses(vertices: &[Vertex], weights: [f64; 4]) -> (Vec3, Vec3) {
    vertices
        .iter()
        .zip(weights)
        .fold((Vec3::ZERO, Vec3::ZERO), |(a, b), (point, weight)| {
            (a + point.a * weight, b + point.b * weight)
        })
}

/// Conservative advancement under translation of B relative to A. Rotation
/// and relative acceleration remain fixed for the duration of the cast.
pub(crate) fn cast_translation(
    a: &impl SupportMap,
    b: &impl SupportMap,
    direction: Vec3,
    max: f64,
    tolerance: f64,
) -> Option<(f64, Vec3, Vec3)> {
    let mut travel = 0.0;
    let mut last = None;
    for _ in 0..64 {
        let translated = Transformed {
            shape: b,
            position: direction * travel,
            rotation: Quat::IDENTITY,
        };
        match distance(a, &translated, tolerance * 0.1) {
            ConvexDistance::Separated {
                distance,
                normal,
                point_a,
                ..
            } => {
                if distance <= tolerance {
                    return Some((travel, normal, point_a));
                }
                let speed = -normal.dot(direction);
                if speed <= 0.0 {
                    return None;
                }
                let step = (distance - tolerance * 0.25) / speed;
                let next = travel + step;
                if !next.is_finite() || next > max {
                    return None;
                }
                last = Some((normal, point_a));
                if next == travel {
                    return None;
                }
                travel = next;
            }
            ConvexDistance::Intersecting => {
                if let Some((normal, point)) = last {
                    return Some((travel, normal, point));
                }
                return match query(a, &translated, tolerance) {
                    ConvexContact::Penetrating {
                        normal, point_a, ..
                    } => Some((0.0, normal, point_a)),
                    ConvexContact::Touching => {
                        let normal = -direction;
                        Some((0.0, normal, a.support(normal)))
                    }
                    _ => None,
                };
            }
            ConvexDistance::Indeterminate => return None,
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::closed_horizon;

    #[test]
    fn horizon_requires_one_closed_cycle_without_duplicate_outgoing_edges() {
        assert!(closed_horizon(&[(2, 0), (0, 1), (1, 2)], 3));
        assert!(!closed_horizon(&[(0, 1), (0, 2), (2, 0)], 3));
        assert!(!closed_horizon(
            &[(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)],
            6
        ));
    }
}
