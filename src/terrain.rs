//! Static triangle-mesh and heightfield surfaces with local BVH acceleration.
use crate::convex::{self, PrimitiveSupport, TriangleSupport};
use crate::spatial::{Aabb, Bvh, Proxy, ray_capsule};
use crate::{RigidBody, Vec3};
use std::collections::BTreeMap;

#[derive(Debug)]
pub struct TriangleMesh {
    vertices: Box<[Vec3]>,
    triangles: Box<[[usize; 3]]>,
    normals: Box<[Vec3]>,
    active_edges: Box<[[bool; 3]]>,
    bvh: Bvh,
    bounds: Aabb,
}

struct EdgeUse {
    triangle: usize,
    edge: usize,
    opposite: usize,
}

impl PartialEq for TriangleMesh {
    fn eq(&self, other: &Self) -> bool {
        self.vertices == other.vertices && self.triangles == other.triangles
    }
}

impl TriangleMesh {
    /// Builds a two-sided triangle surface. Faces must have distinct, valid
    /// vertex indices and non-degenerate geometry.
    pub fn new(
        vertices: impl Into<Box<[Vec3]>>,
        triangles: impl Into<Box<[[usize; 3]]>>,
    ) -> Option<Self> {
        let vertices = vertices.into();
        let triangles = triangles.into();
        if vertices.iter().any(|p| !p.is_finite()) || triangles.is_empty() {
            return None;
        }
        let mut normals = Vec::with_capacity(triangles.len());
        let mut neighbors = BTreeMap::<_, Vec<EdgeUse>>::new();
        let mut proxies = Vec::with_capacity(triangles.len());
        for (index, &[i, j, k]) in triangles.iter().enumerate() {
            let [a, b, c] = [*vertices.get(i)?, *vertices.get(j)?, *vertices.get(k)?];
            let normal = (b - a).cross(c - a).try_normalize()?;
            if !normal.is_finite() {
                return None;
            }
            normals.push(normal);
            for (edge, (u, v, w)) in [(i, j, k), (j, k, i), (k, i, j)].into_iter().enumerate() {
                neighbors
                    .entry((u.min(v), u.max(v)))
                    .or_default()
                    .push(EdgeUse {
                        triangle: index,
                        edge,
                        opposite: w,
                    });
            }
            proxies.push(Proxy {
                index,
                bounds: Aabb {
                    min: a.min(b).min(c),
                    max: a.max(b).max(c),
                },
            });
        }
        let mut active_edges = vec![[true; 3]; triangles.len()];
        for ((u, v), linked) in neighbors {
            let [a, b] = linked.as_slice() else {
                continue;
            };
            let n = normals[a.triangle];
            let edge = vertices[v] - vertices[u];
            let side = |w| edge.cross(vertices[w] - vertices[u]).dot(n);
            if n.dot(normals[b.triangle]).abs() > 1.0 - 1e-10
                && side(a.opposite).signum() != side(b.opposite).signum()
            {
                active_edges[a.triangle][a.edge] = false;
                active_edges[b.triangle][b.edge] = false;
            }
        }
        let bounds = proxies.iter().map(|p| p.bounds).reduce(Aabb::union)?;
        let mut bvh = Bvh::default();
        bvh.rebuild(&mut proxies);
        Some(Self {
            vertices,
            triangles,
            normals: normals.into_boxed_slice(),
            active_edges: active_edges.into_boxed_slice(),
            bvh,
            bounds,
        })
    }

    /// Triangulates row-major heights in a regular XZ grid, starting at
    /// the local origin. Normals follow +Y for a flat field.
    pub fn heightfield(size: [usize; 2], spacing: [f64; 2], heights: &[f64]) -> Option<Self> {
        let [width, depth] = size;
        let [dx, dz] = spacing;
        if width < 2
            || depth < 2
            || width.checked_mul(depth)? != heights.len()
            || !spacing.iter().all(|&v| v.is_finite() && v > 0.0)
        {
            return None;
        }
        let vertices = heights
            .iter()
            .enumerate()
            .map(|(i, &h)| Vec3::new((i % width) as f64 * dx, h, (i / width) as f64 * dz))
            .collect::<Vec<_>>();
        let mut triangles = Vec::with_capacity((width - 1).checked_mul(depth - 1)?.checked_mul(2)?);
        for z in 0..depth - 1 {
            for x in 0..width - 1 {
                let a = z * width + x;
                let b = a + 1;
                let c = a + width;
                let d = c + 1;
                triangles.extend([[a, c, b], [b, c, d]]);
            }
        }
        Self::new(vertices, triangles)
    }

    pub fn vertices(&self) -> &[Vec3] {
        &self.vertices
    }
    pub fn triangles(&self) -> &[[usize; 3]] {
        &self.triangles
    }
    pub fn bounds(&self) -> Aabb {
        self.bounds
    }

    fn triangle(&self, index: usize) -> [Vec3; 3] {
        self.triangles[index].map(|v| self.vertices[v])
    }

    pub(crate) fn project(&self, point: Vec3) -> (Vec3, Vec3, f64) {
        let (distance, _, (surface, normal)) = self
            .bvh
            .nearest(point, |i| {
                let triangle = self.triangle(i);
                let surface = closest_feature(point, triangle, self.normals[i]).0;
                let delta = point - surface;
                let normal = delta.try_normalize().unwrap_or(self.normals[i]);
                (delta.length_squared(), (surface, normal))
            })
            .expect("validated nonempty mesh");
        (surface, normal, distance.sqrt())
    }

    pub(crate) fn ray(&self, origin: Vec3, direction: Vec3) -> Option<(f64, Vec3)> {
        let mut best = None;
        self.bvh.visit_ray(
            origin,
            direction,
            Vec3::ZERO,
            f64::INFINITY,
            |index, far| {
                let (t, n) =
                    ray_triangle(origin, direction, self.triangle(index), self.normals[index])?;
                if t > far {
                    return None;
                }
                if best.is_none_or(|(distance, i, _)| t < distance || (t == distance && index < i))
                {
                    best = Some((t, index, n));
                }
                Some(t)
            },
        );
        best.map(|(t, _, n)| (t, n))
    }

    pub(crate) fn visit_triangles(&self, bounds: Aabb, mut visit: impl FnMut([Vec3; 3])) {
        self.bvh.visit(
            |tri| tri.intersects(bounds),
            |index| visit(self.triangle(index)),
        );
    }

    pub(crate) fn sphere_contacts(
        &self,
        center: Vec3,
        radius: f64,
        mut emit: impl FnMut(Vec3, Vec3, f64),
    ) {
        let bounds = Aabb {
            min: center - Vec3::splat(radius),
            max: center + Vec3::splat(radius),
        };
        self.bvh.visit(
            |triangle| triangle.intersects(bounds),
            |index| {
                let triangle = self.triangle(index);
                let (closest, feature) = closest_feature(center, triangle, self.normals[index]);
                if let Some((edge, t)) = feature {
                    let active = self.active_edges[index];
                    if !active[edge]
                        && !(t == 0.0 && active[(edge + 2) % 3])
                        && !(t == 1.0 && active[(edge + 1) % 3])
                    {
                        return;
                    }
                }
                let delta = center - closest;
                let depth = radius - delta.length();
                if depth >= 0.0 {
                    let normal = delta.try_normalize().unwrap_or(self.normals[index]);
                    emit(closest, normal, depth);
                }
            },
        );
    }

    pub(crate) fn cast_convex(
        &self,
        moving: &RigidBody,
        direction: Vec3,
        max: f64,
    ) -> Option<(f64, Vec3, Vec3)> {
        let bounds = Aabb::of(moving)?;
        let extent = (bounds.max - bounds.min) * 0.5;
        let center = (bounds.max + bounds.min) * 0.5;
        let tolerance = extent.max_element() * 1e-6;
        let support = PrimitiveSupport::new(moving)?;
        let mut best = None;
        self.bvh
            .visit_ray(center, direction, extent, max, |index, far| {
                let triangle = TriangleSupport(self.triangle(index));
                let (t, n, p) =
                    convex::cast_translation(&triangle, &support, direction, far, tolerance)?;
                if best.is_none_or(|(old, _, _, j)| t < old || (t == old && index < j)) {
                    best = Some((t, n, p, index));
                }
                Some(t)
            });
        best.map(|(t, n, p, _)| (t, n, p))
    }

    pub(crate) fn cast_sphere(
        &self,
        origin: Vec3,
        direction: Vec3,
        radius: f64,
        max: f64,
    ) -> Option<(f64, Vec3)> {
        let mut best = None;
        self.bvh
            .visit_ray(origin, direction, Vec3::splat(radius), max, |index, far| {
                let (t, n) = cast_triangle(
                    origin,
                    direction,
                    radius,
                    self.triangle(index),
                    self.normals[index],
                    self.active_edges[index],
                )?;
                if t > far {
                    return None;
                }
                if best.is_none_or(|(distance, i, _)| t < distance || (t == distance && index < i))
                {
                    best = Some((t, index, n));
                }
                Some(t)
            });
        best.map(|(t, _, n)| (t, n))
    }
}

pub(crate) fn triangle_normal([a, b, c]: [Vec3; 3]) -> Vec3 {
    (b - a).cross(c - a).normalize()
}

pub(crate) fn inside_triangle(p: Vec3, [a, b, c]: [Vec3; 3], n: Vec3) -> bool {
    [(a, b), (b, c), (c, a)]
        .into_iter()
        .all(|(a, b)| (b - a).cross(p - a).dot(n) >= -1e-12 * (b - a).length_squared())
}

pub(crate) fn closest_triangle(point: Vec3, triangle: [Vec3; 3]) -> Vec3 {
    closest_feature(point, triangle, triangle_normal(triangle)).0
}

fn closest_feature(
    point: Vec3,
    triangle @ [a, b, c]: [Vec3; 3],
    normal: Vec3,
) -> (Vec3, Option<(usize, f64)>) {
    let projected = point - normal * (point - a).dot(normal);
    if inside_triangle(projected, triangle, normal) {
        return (projected, None);
    }
    let (nearest, edge, t) = [(a, b), (b, c), (c, a)]
        .into_iter()
        .enumerate()
        .map(|(i, (a, b))| {
            let edge = b - a;
            let t = ((point - a).dot(edge) / edge.length_squared()).clamp(0.0, 1.0);
            (a + edge * t, i, t)
        })
        .min_by(|(a, _, _), (b, _, _)| {
            a.distance_squared(point)
                .total_cmp(&b.distance_squared(point))
        })
        .expect("three edges");
    (nearest, Some((edge, t)))
}

fn ray_triangle(
    origin: Vec3,
    direction: Vec3,
    triangle: [Vec3; 3],
    normal: Vec3,
) -> Option<(f64, Vec3)> {
    let denominator = normal.dot(direction);
    if denominator == 0.0 {
        return None;
    }
    let t = normal.dot(triangle[0] - origin) / denominator;
    (t >= 0.0 && inside_triangle(origin + direction * t, triangle, normal))
        .then_some((t, -normal * denominator.signum()))
}

fn cast_triangle(
    origin: Vec3,
    direction: Vec3,
    radius: f64,
    triangle: [Vec3; 3],
    normal: Vec3,
    active_edges: [bool; 3],
) -> Option<(f64, Vec3)> {
    let start_distance = origin.distance(closest_feature(origin, triangle, normal).0);
    if start_distance < radius * (1.0 - 1e-8) {
        return None;
    }

    let denominator = normal.dot(direction);
    let mut best = None;
    if denominator != 0.0 {
        let n = -normal * denominator.signum();
        let start_gap = n.dot(origin - triangle[0]) - radius;
        let t = start_gap / denominator.abs();
        let touch = origin + direction * t - n * radius;
        if t >= 0.0 && inside_triangle(touch, triangle, normal) {
            best = Some((t, n));
        }
    }
    let [a, b, c] = triangle;
    for ((start, end), active) in [(a, b), (b, c), (c, a)].into_iter().zip(active_edges) {
        if !active {
            continue;
        }
        let edge = end - start;
        let length = edge.length();
        let u = edge / length;
        let (v, w) = u.any_orthonormal_pair();
        let center = (start + end) * 0.5;
        let local = |p: Vec3| Vec3::new(v.dot(p), u.dot(p), w.dot(p));
        if let Some((t, n)) = ray_capsule(
            local(origin - center),
            local(direction),
            radius,
            length * 0.5,
        ) {
            let normal = v * n.x + u * n.y + w * n.z;
            if direction.dot(normal) < -1e-12 && best.is_none_or(|(distance, _)| t < distance) {
                best = Some((t, normal));
            }
        }
    }
    best
}
