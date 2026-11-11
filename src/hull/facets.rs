//! Coplanar facet reconstruction for clipping, projection, and ray tests.
use crate::terrain::triangle_normal;
use crate::{Vec2, Vec3};

#[derive(Debug, PartialEq)]
pub(super) struct Facet {
    pub normal: Vec3,
    pub offset: f64,
    pub polygon: Box<[usize]>,
}

pub(super) fn build(vertices: &[Vec3], triangles: &[[usize; 3]], scale: f64) -> Vec<Facet> {
    let mut facets: Vec<Facet> = Vec::new();
    let tolerance = scale * 1e-9;
    for &face in triangles {
        let triangle = face.map(|i| vertices[i]);
        let normal = triangle_normal(triangle);
        let offset = normal.dot(triangle[0]);
        if facets.iter().any(|f| {
            f.normal.dot(normal) > 1.0 - 1e-10
                && (f.offset - offset).abs() <= tolerance
                && triangle
                    .iter()
                    .all(|&p| (f.normal.dot(p) - f.offset).abs() <= tolerance)
        }) {
            continue;
        }
        let (u, v) = normal.any_orthonormal_pair();
        let mut points: Vec<_> = vertices
            .iter()
            .enumerate()
            .filter_map(|(index, &p)| {
                ((p.dot(normal) - offset).abs() <= tolerance)
                    .then_some((Vec2::new(p.dot(u), p.dot(v)), index))
            })
            .collect();
        points.sort_unstable_by(|a, b| {
            a.0.x
                .total_cmp(&b.0.x)
                .then(a.0.y.total_cmp(&b.0.y))
                .then(a.1.cmp(&b.1))
        });
        points.dedup_by(|a, b| a.0 == b.0);
        let mut outline = Vec::new();
        extend_chain(points.iter().copied(), &mut outline, 0);
        let lower = outline.len().saturating_sub(1);
        outline.truncate(lower);
        extend_chain(points.iter().rev().copied(), &mut outline, lower);
        outline.pop();
        let mut polygon: Vec<_> = outline.into_iter().map(|(_, i)| i).collect();
        if u.cross(v).dot(normal) < 0.0 {
            polygon.reverse();
        }
        if polygon.len() < 3 {
            polygon = face.to_vec();
        }
        facets.push(Facet {
            normal,
            offset,
            polygon: polygon.into_boxed_slice(),
        });
    }
    facets
}

fn extend_chain(
    points: impl Iterator<Item = (Vec2, usize)>,
    outline: &mut Vec<(Vec2, usize)>,
    start: usize,
) {
    for point in points {
        while outline.len() >= start + 2 {
            let a = outline[outline.len() - 2].0;
            let b = outline[outline.len() - 1].0;
            if (b - a).perp_dot(point.0 - a) > 0.0 {
                break;
            }
            outline.pop();
        }
        outline.push(point);
    }
}
