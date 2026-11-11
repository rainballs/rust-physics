//! Discrete vertex–triangle and edge–edge contacts for deformable surfaces.
use super::{Node, ordered};
use crate::Vec3;
use crate::collision::closest_segment_parameters;
use crate::spatial::{Aabb, Bvh, Proxy};
use std::collections::BTreeSet;

#[derive(Default)]
pub(super) struct SurfaceContacts {
    triangle_bvh: Bvh,
    edge_bvh: Bvh,
    proxies: Vec<Proxy>,
    edges: Vec<[usize; 2]>,
    topology_count: usize,
    edge_pairs: Vec<[usize; 2]>,
}

impl SurfaceContacts {
    pub(super) fn solve(
        &mut self,
        nodes: &mut [Node],
        triangles: &[[usize; 3]],
        adjacent: &BTreeSet<(usize, usize)>,
        thickness: f64,
    ) {
        if self.topology_count != triangles.len() {
            self.edges = triangles
                .iter()
                .flat_map(|&[a, b, c]| [ordered(a, b), ordered(b, c), ordered(c, a)])
                .collect::<BTreeSet<_>>()
                .into_iter()
                .map(|(a, b)| [a, b])
                .collect();
            self.topology_count = triangles.len();
        }
        self.proxies.clear();
        self.proxies
            .extend(triangles.iter().enumerate().map(|(index, face)| {
                let [a, b, c] = face.map(|i| nodes[i].position);
                Proxy {
                    index,
                    bounds: Aabb {
                        min: a.min(b).min(c),
                        max: a.max(b).max(c),
                    }
                    .expanded(thickness),
                }
            }));
        self.triangle_bvh.rebuild(&mut self.proxies);
        for i in 0..nodes.len() {
            let query = Aabb {
                min: nodes[i].position,
                max: nodes[i].position,
            };
            self.triangle_bvh.visit(
                |bounds| bounds.intersects(query),
                |j| {
                    let face = triangles[j];
                    if face
                        .iter()
                        .any(|&v| i == v || adjacent.contains(&ordered(i, v)))
                    {
                        return;
                    }
                    vertex_face(nodes, i, face, thickness);
                },
            );
        }

        self.proxies.clear();
        self.proxies
            .extend(self.edges.iter().enumerate().map(|(index, edge)| {
                let [a, b] = edge.map(|i| nodes[i].position);
                Proxy {
                    index,
                    bounds: Aabb {
                        min: a.min(b),
                        max: a.max(b),
                    }
                    .expanded(thickness * 0.5),
                }
            }));
        self.edge_bvh.rebuild(&mut self.proxies);
        self.edge_pairs.clear();
        self.edge_bvh.pairs(|a, b| self.edge_pairs.push([a, b]));
        for &[a, b] in &self.edge_pairs {
            let [first, second] = [self.edges[a], self.edges[b]];
            if first.into_iter().any(|i| {
                second
                    .into_iter()
                    .any(|j| i == j || adjacent.contains(&ordered(i, j)))
            }) {
                continue;
            }
            edge_edge(nodes, first, second, thickness);
        }
    }
}

fn vertex_face(nodes: &mut [Node], vertex: usize, face: [usize; 3], thickness: f64) {
    let point = nodes[vertex].position;
    let triangle = face.map(|i| nodes[i].position);
    let [a, b, c] = triangle;
    let (closest, weights) = closest_triangle(point, triangle);
    let delta = point - closest;
    let distance = delta.length();
    if distance >= thickness {
        return;
    }
    let normal = delta.try_normalize().unwrap_or_else(|| {
        let face_normal = (b - a).cross(c - a).try_normalize().unwrap_or(Vec3::Y);
        let previous = face
            .into_iter()
            .zip(weights)
            .map(|(i, w)| nodes[i].previous * w)
            .sum::<Vec3>();
        face_normal
            * (nodes[vertex].previous - previous)
                .dot(face_normal)
                .signum()
    });
    let [u, v, w] = weights;
    let [a, b, c] = face;
    project(
        nodes,
        [vertex, a, b, c],
        [1.0, -u, -v, -w],
        normal,
        thickness - distance,
    );
}

fn closest_triangle(point: Vec3, [a, b, c]: [Vec3; 3]) -> (Vec3, [f64; 3]) {
    let u = b - a;
    let v = c - a;
    let normal = u.cross(v);
    let denominator = normal.length_squared();
    if denominator > 1e-24 * u.length_squared() * v.length_squared() {
        let y = (point - a).cross(v).dot(normal) / denominator;
        let z = u.cross(point - a).dot(normal) / denominator;
        if y >= 0.0 && z >= 0.0 && y + z <= 1.0 {
            return (a + u * y + v * z, [1.0 - y - z, y, z]);
        }
    }
    let vertices = [a, b, c];
    [(0, 1), (1, 2), (2, 0)]
        .into_iter()
        .map(|(i, j)| {
            let start = vertices[i];
            let edge = vertices[j] - start;
            let length_squared = edge.length_squared();
            let t = if length_squared > 0.0 {
                ((point - start).dot(edge) / length_squared).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let mut weights = [0.0; 3];
            weights[i] = 1.0 - t;
            weights[j] = t;
            (start + edge * t, weights)
        })
        .min_by(|(a, _), (b, _)| {
            point
                .distance_squared(*a)
                .total_cmp(&point.distance_squared(*b))
        })
        .unwrap()
}

fn edge_edge(nodes: &mut [Node], [a, b]: [usize; 2], [c, d]: [usize; 2], thickness: f64) {
    let [p, q, r, s] = [a, b, c, d].map(|i| nodes[i].position);
    let [u, v] = [q - p, s - r];
    let (first, second) = closest_segment_parameters(p, u, r, v);
    let delta = (r + v * second) - (p + u * first);
    let distance = delta.length();
    if distance >= thickness {
        return;
    }
    let normal = delta.try_normalize().unwrap_or_else(|| {
        let n = u
            .cross(v)
            .try_normalize()
            .or_else(|| u.try_normalize().map(|axis| axis.any_orthonormal_vector()))
            .unwrap_or(Vec3::X);
        let p = nodes[a].previous.lerp(nodes[b].previous, first);
        let q = nodes[c].previous.lerp(nodes[d].previous, second);
        n * (q - p).dot(n).signum()
    });
    project(
        nodes,
        [a, b, c, d],
        [first - 1.0, -first, 1.0 - second, second],
        normal,
        thickness - distance,
    );
}

fn project(nodes: &mut [Node], indices: [usize; 4], weights: [f64; 4], normal: Vec3, depth: f64) {
    let response: f64 = indices
        .iter()
        .zip(weights)
        .map(|(&i, w)| nodes[i].inverse_mass * w * w)
        .sum();
    if response <= 0.0 {
        return;
    }
    let correction = normal * (depth / response);
    for (i, w) in indices.into_iter().zip(weights) {
        nodes[i].position += correction * (nodes[i].inverse_mass * w);
    }
}
