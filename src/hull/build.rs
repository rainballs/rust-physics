//! Deterministic incremental convex hull construction for point clouds.
use super::RigidHull;
use crate::Vec3;
use std::collections::{BTreeMap, BTreeSet};

const TOLERANCE: f64 = 1e-9;

#[derive(Clone, Copy)]
struct Face {
    indices: [usize; 3],
    normal: Vec3,
    offset: f64,
}

impl Face {
    fn new(points: &[Vec3], mut indices: [usize; 3], inside: Vec3) -> Option<Self> {
        let [a, b, c] = indices.map(|i| points[i]);
        let mut normal = (b - a).cross(c - a).try_normalize()?;
        if normal.dot(inside - a) > 0.0 {
            normal = -normal;
            indices.swap(1, 2);
        }
        Some(Self {
            indices,
            normal,
            offset: normal.dot(a),
        })
    }

    fn visible(self, point: Vec3) -> bool {
        self.normal.dot(point) - self.offset > TOLERANCE
    }
}

impl RigidHull {
    /// Builds a convex solid from a point cloud. Exact duplicates and interior
    /// points are removed. Features below roughly 1e-9 of the cloud extent
    /// are treated as coplanar during hull construction.
    pub fn from_points(points: impl IntoIterator<Item = Vec3>) -> Option<Self> {
        let mut original: Vec<_> = points
            .into_iter()
            .map(|point| Vec3::from_array(point.to_array().map(|v| if v == 0.0 { 0.0 } else { v })))
            .collect();
        if original.iter().any(|p| !p.is_finite()) || original.len() < 4 {
            return None;
        }
        original.sort_unstable_by(|a, b| {
            a.x.total_cmp(&b.x)
                .then(a.y.total_cmp(&b.y))
                .then(a.z.total_cmp(&b.z))
        });
        original.dedup();
        if original.len() < 4 {
            return None;
        }

        let lo = original.iter().copied().reduce(Vec3::min)?;
        let hi = original.iter().copied().reduce(Vec3::max)?;
        let center = lo * 0.5 + hi * 0.5;
        let scale = (hi - lo).max_element();
        if !scale.is_finite() || scale <= 0.0 {
            return None;
        }
        let points: Vec<_> = original.iter().map(|&p| (p - center) / scale).collect();
        let seed = initial_tetrahedron(&points)?;
        let [a, b, c, d] = seed;
        let inside = seed.into_iter().map(|i| points[i]).sum::<Vec3>() * 0.25;
        let mut faces: Vec<_> = [[a, b, c], [a, b, d], [a, c, d], [b, c, d]]
            .into_iter()
            .map(|indices| Face::new(&points, indices, inside))
            .collect::<Option<_>>()?;
        let mut horizon = BTreeSet::new();
        for (index, &point) in points.iter().enumerate() {
            if seed.contains(&index) {
                continue;
            }
            horizon.clear();
            let mut valid = true;
            faces.retain(|face| {
                if !face.visible(point) {
                    return true;
                }
                let [a, b, c] = face.indices;
                for edge in [(a, b), (b, c), (c, a)] {
                    if !horizon.remove(&(edge.1, edge.0)) && !horizon.insert(edge) {
                        valid = false;
                    }
                }
                false
            });
            if !valid {
                return None;
            }
            if horizon.is_empty() {
                continue;
            }
            if !single_cycle(&horizon) {
                return None;
            }
            for &(a, b) in &horizon {
                let indices = [a, b, index];
                let face = Face::new(&points, indices, inside)?;
                if face.indices != indices {
                    return None;
                }
                faces.push(face);
            }
        }
        let mut mapping = BTreeMap::new();
        for index in faces.iter().flat_map(|f| f.indices) {
            mapping.entry(index).or_insert(0);
        }
        let mut vertices = Vec::with_capacity(mapping.len());
        for (&old, index) in &mut mapping {
            *index = vertices.len();
            vertices.push(original[old]);
        }
        let triangles = faces
            .into_iter()
            .map(|face| face.indices.map(|i| mapping[&i]))
            .collect();
        Self::new(vertices, triangles)
    }
}

fn initial_tetrahedron(points: &[Vec3]) -> Option<[usize; 4]> {
    let farthest = |score: &dyn Fn(Vec3) -> f64| {
        points
            .iter()
            .enumerate()
            .map(|(i, &p)| (i, score(p)))
            .max_by(|a, b| a.1.total_cmp(&b.1))
    };
    let a = points[0];
    let (b, distance) = farthest(&|p| p.distance_squared(a))?;
    if distance <= TOLERANCE * TOLERANCE {
        return None;
    }
    let direction = (points[b] - a).normalize();
    let (c, distance) = farthest(&|p| direction.cross(p - a).length_squared())?;
    if distance <= TOLERANCE * TOLERANCE {
        return None;
    }
    let normal = direction.cross(points[c] - a).normalize();
    let (d, distance) = farthest(&|p| normal.dot(p - a).abs())?;
    (distance > TOLERANCE).then_some([0, b, c, d])
}

fn single_cycle(horizon: &BTreeSet<(usize, usize)>) -> bool {
    if horizon.len() < 3 {
        return false;
    }
    let links: BTreeMap<_, _> = horizon.iter().copied().collect();
    if links.len() != horizon.len() {
        return false;
    }
    let start = *links.first_key_value().expect("nonempty horizon").0;
    let mut current = start;
    for i in 0..links.len() {
        let Some(&next) = links.get(&current) else {
            return false;
        };
        current = next;
        if current == start {
            return i + 1 == links.len();
        }
    }
    false
}
