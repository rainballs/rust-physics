//! Narrow phase for spheres, capsules, oriented boxes, and infinite half-spaces.
use crate::convex::{self, ConvexContact, PrimitiveSupport, TriangleSupport};
use crate::spatial::{Aabb, box_intervals, project};
use crate::terrain::{TriangleMesh, closest_triangle, inside_triangle, triangle_normal};
use crate::{RigidBody, Shape, Vec3};

/// A world-space contact. The normal points in the direction that separates
/// the second body from the first.
#[derive(Clone, Copy, Debug, Default)]
pub struct Contact {
    pub point: Vec3,
    pub normal: Vec3,
    pub depth: f64,
}

/// Emits the current overlap manifold into a caller-owned vector.
pub fn contacts(a: &RigidBody, b: &RigidBody, out: &mut Vec<Contact>) {
    let start = out.len();
    let planar = planar_pair(a, b);
    match (&a.shape, &b.shape) {
        (Shape::Compound(compound), _) => compound_contacts(a, compound, b, out),
        (_, Shape::Compound(compound)) => {
            compound_contacts(b, compound, a, out);
            flip(&mut out[start..]);
        }
        (Shape::Mesh(mesh), Shape::Sphere { radius }) => mesh_sphere(a, mesh, b, *radius, out),
        (Shape::Sphere { radius }, Shape::Mesh(mesh)) => {
            mesh_sphere(b, mesh, a, *radius, out);
            flip(&mut out[start..]);
        }
        (Shape::Mesh(mesh), Shape::Cuboid { half_extents }) => {
            mesh_box(a, mesh, b, *half_extents, out);
        }
        (Shape::Cuboid { half_extents }, Shape::Mesh(mesh)) => {
            mesh_box(b, mesh, a, *half_extents, out);
            flip(&mut out[start..]);
        }
        (
            Shape::Mesh(mesh),
            Shape::Capsule {
                radius,
                half_segment,
            },
        ) => {
            mesh_capsule(a, mesh, b, *radius, *half_segment, out);
        }
        (
            Shape::Capsule {
                radius,
                half_segment,
            },
            Shape::Mesh(mesh),
        ) => {
            mesh_capsule(b, mesh, a, *radius, *half_segment, out);
            flip(&mut out[start..]);
        }
        (Shape::Mesh(mesh), Shape::Cylinder { .. } | Shape::Hull(_)) => {
            mesh_convex(a, mesh, b, out)
        }
        (Shape::Cylinder { .. } | Shape::Hull(_), Shape::Mesh(mesh)) => {
            mesh_convex(b, mesh, a, out);
            flip(&mut out[start..]);
        }
        (Shape::Mesh(_), _) | (_, Shape::Mesh(_)) => {}
        (Shape::Sphere { radius: ar }, Shape::Sphere { radius: br }) => {
            rounded_points(a.position, b.position, *ar, *br, Vec3::X, out);
        }
        (Shape::Plane, Shape::Plane) => {}
        (Shape::Plane, _) => plane_shape(a, b, out),
        (_, Shape::Plane) => {
            plane_shape(b, a, out);
            flip(&mut out[start..]);
        }
        (Shape::Cuboid { .. }, Shape::Sphere { radius }) => {
            shape_sphere(a, b, *radius, out);
        }
        (Shape::Sphere { radius }, Shape::Cuboid { .. }) => {
            shape_sphere(b, a, *radius, out);
            flip(&mut out[start..]);
        }
        (
            Shape::Capsule {
                radius: ar,
                half_segment: ah,
            },
            Shape::Capsule {
                radius: br,
                half_segment: bh,
            },
        ) => {
            let u = a.rotation * Vec3::Y;
            let v = b.rotation * Vec3::Y;
            let (p, q) = closest_segments(
                a.position - u * ah,
                u * (2.0 * ah),
                b.position - v * bh,
                v * (2.0 * bh),
            );
            let normal = u.cross(v).try_normalize().unwrap_or(a.rotation * Vec3::X);
            rounded_points(p, q, *ar, *br, normal, out);
        }
        (Shape::Capsule { .. }, Shape::Sphere { radius }) => capsule_sphere(a, b, *radius, out),
        (Shape::Sphere { radius }, Shape::Capsule { .. }) => {
            capsule_sphere(b, a, *radius, out);
            flip(&mut out[start..]);
        }
        (Shape::Cuboid { half_extents }, Shape::Capsule { .. }) => {
            box_capsule(a, *half_extents, b, out)
        }
        (Shape::Capsule { .. }, Shape::Cuboid { half_extents }) => {
            box_capsule(b, *half_extents, a, out);
            flip(&mut out[start..]);
        }
        (Shape::Hull(_), Shape::Sphere { radius }) => shape_sphere(a, b, *radius, out),
        (Shape::Sphere { radius }, Shape::Hull(_)) => {
            shape_sphere(b, a, *radius, out);
            flip(&mut out[start..]);
        }
        (Shape::Hull(_), _) | (_, Shape::Hull(_)) => convex_pair(a, b, out),
        (Shape::Cylinder { .. }, Shape::Sphere { radius }) => shape_sphere(a, b, *radius, out),
        (Shape::Sphere { radius }, Shape::Cylinder { .. }) => {
            shape_sphere(b, a, *radius, out);
            flip(&mut out[start..]);
        }
        (
            Shape::Cylinder {
                radius: ar,
                half_height: ah,
            },
            Shape::Cylinder {
                radius: br,
                half_height: bh,
            },
        ) => {
            let axis = a.rotation * Vec3::Y;
            if axis.cross(b.rotation * Vec3::Y).length_squared() < 1e-20 {
                parallel_cylinders(a.position, b.position, axis, [*ar, *br], [*ah, *bh], out);
            } else {
                convex_pair(a, b, out);
            }
        }
        (Shape::Cylinder { .. }, _) | (_, Shape::Cylinder { .. }) => convex_pair(a, b, out),
        (Shape::Cuboid { half_extents: ah }, Shape::Cuboid { half_extents: bh }) => {
            box_box(Obb::new(a, *ah), Obb::new(b, *bh), planar, out);
        }
    }
    if planar && out.len() > start {
        for contact in &mut out[start..] {
            contact.point.z = a.position.z;
        }
        reduce_contacts(out, start);
    }
}

fn planar_pair(a: &RigidBody, b: &RigidBody) -> bool {
    [a, b].into_iter().all(|body| {
        body.linear_axes == Vec3::new(1.0, 1.0, 0.0)
            && body.angular_axes == Vec3::Z
            && (body.rotation * Vec3::Z).distance_squared(Vec3::Z) < 1e-20
    }) && a.position.z == b.position.z
}

fn flip(contacts: &mut [Contact]) {
    for contact in contacts {
        contact.normal = -contact.normal;
    }
}

fn plane_shape(plane: &RigidBody, body: &RigidBody, out: &mut Vec<Contact>) {
    let start = out.len();
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
        Shape::Capsule {
            radius,
            half_segment,
        } => {
            let u = body.rotation * Vec3::Y * half_segment;
            add(body.position - u - normal * radius);
            if half_segment > 0.0 {
                add(body.position + u - normal * radius);
            }
        }
        Shape::Cylinder {
            radius,
            half_height,
        } => cylinder_plane_points(body, normal, radius, half_height, add),
        Shape::Hull(ref hull) => {
            for &vertex in hull.surface().vertices() {
                add(body.position + body.rotation * vertex);
            }
            reduce_contacts(out, start);
        }
        Shape::Plane | Shape::Mesh(_) | Shape::Compound(_) => {}
    }
}

fn compound_contacts(
    a: &RigidBody,
    compound: &crate::compound::Compound,
    b: &RigidBody,
    out: &mut Vec<Contact>,
) {
    let start = out.len();
    let bounds = Aabb::of(b);
    for part in compound.parts() {
        let child = a.collider_part(part);
        if bounds.is_none_or(|other| Aabb::of(&child).is_some_and(|local| local.intersects(other)))
        {
            contacts(&child, b, out);
        }
    }
    reduce_contacts(out, start);
}

fn reduce_contacts(out: &mut Vec<Contact>, start: usize) {
    // Contact caching has a fixed eight-point budget. Prefer deeper and
    // distinct witnesses, with stable child order for ties.
    if out.len() - start > 8 {
        out[start..].sort_by(|a, b| b.depth.total_cmp(&a.depth));
    }
    let mut count = start;
    for i in start..out.len() {
        let contact = out[i];
        if !out[start..count].iter().any(|existing| {
            existing.normal.dot(contact.normal) > 1.0 - 1e-8
                && existing.point.distance_squared(contact.point) < 1e-14
        }) {
            out[count] = contact;
            count += 1;
            if count == start + 8 {
                break;
            }
        }
    }
    out.truncate(count);
}

fn mesh_convex(terrain: &RigidBody, mesh: &TriangleMesh, body: &RigidBody, out: &mut Vec<Contact>) {
    let inverse = terrain.rotation.conjugate();
    let mut local = body.clone();
    local.position = inverse * (body.position - terrain.position);
    local.rotation = inverse * body.rotation;
    let bounds = Aabb::of(&local).expect("finite convex bounds");
    let shape = PrimitiveSupport::new(&local).expect("finite convex support");
    let tolerance = (bounds.max - bounds.min).max_element() * 1e-6;
    let start = out.len();
    let mut add = |point: Vec3, normal: Vec3, depth| {
        out.push(Contact {
            point: terrain.position + terrain.rotation * point,
            normal: terrain.rotation * normal,
            depth,
        });
    };
    mesh.visit_triangles(bounds, |triangle| {
        let ConvexContact::Penetrating {
            depth,
            normal,
            point_a,
            point_b,
        } = convex::query(&TriangleSupport(triangle), &shape, tolerance)
        else {
            return;
        };

        // When the separating axis is the triangle face normal, expand the
        // single EPA witness using penetrating feature points on the face.
        let facet = triangle_normal(triangle);
        let mut contributed = false;
        if normal.dot(facet).abs() > 1.0 - 1e-6 {
            let normal = facet * normal.dot(facet).signum();
            let mut sample = |p: Vec3| {
                let signed = (p - triangle[0]).dot(normal);
                if signed <= 0.0 && inside_triangle(p - normal * signed, triangle, facet) {
                    add(p - normal * (signed * 0.5), normal, -signed);
                    contributed = true;
                }
            };
            match &local.shape {
                Shape::Cylinder {
                    radius,
                    half_height,
                } => {
                    cylinder_plane_points(&local, normal, *radius, *half_height, sample);
                }
                Shape::Hull(hull) => {
                    for &vertex in hull.surface().vertices() {
                        sample(local.position + local.rotation * vertex);
                    }
                }
                _ => unreachable!("supported mesh convex geometry"),
            }
        }
        if !contributed {
            add((point_a + point_b) * 0.5, normal, depth);
        }
    });
    reduce_contacts(out, start);
}

fn mesh_sphere(
    terrain: &RigidBody,
    mesh: &TriangleMesh,
    sphere: &RigidBody,
    radius: f64,
    out: &mut Vec<Contact>,
) {
    let center = terrain.rotation.conjugate() * (sphere.position - terrain.position);
    let mut best = None;
    mesh.sphere_contacts(center, radius, |point, normal, depth| {
        if best.is_none_or(|(_, _, deepest)| depth > deepest) {
            best = Some((point, normal, depth));
        }
    });
    if let Some((point, normal, depth)) = best {
        out.push(Contact {
            point: terrain.position + terrain.rotation * (point - normal * depth * 0.5),
            normal: terrain.rotation * normal,
            depth,
        });
    }
}

fn mesh_box(
    terrain: &RigidBody,
    mesh: &TriangleMesh,
    cube: &RigidBody,
    h: Vec3,
    out: &mut Vec<Contact>,
) {
    let inverse = terrain.rotation.conjugate();
    let center = inverse * (cube.position - terrain.position);
    let orientation = inverse * cube.rotation;
    let extent = (orientation * Vec3::X).abs() * h.x
        + (orientation * Vec3::Y).abs() * h.y
        + (orientation * Vec3::Z).abs() * h.z;
    let local = orientation.conjugate();
    let mut patch: Vec<Vec3> = Vec::new();
    let mut deepest = (f64::NEG_INFINITY, Vec3::Y);
    let tolerance = h.max_element() * 1e-7;
    mesh.visit_triangles(
        Aabb {
            min: center - extent,
            max: center + extent,
        },
        |triangle| {
            let triangle = triangle.map(|p| local * (p - center));
            let Some((depth, normal)) = triangle_box_axis(triangle, h) else {
                return;
            };
            if depth > deepest.0 + tolerance {
                if normal.dot(deepest.1) < 1.0 - 1e-8 {
                    patch.clear();
                }
                deepest = (depth, normal);
            }
            if normal.dot(deepest.1) >= 1.0 - 1e-8 && depth >= deepest.0 - tolerance {
                clipped_triangle(triangle, h, &mut patch);
            }
        },
    );
    if patch.is_empty() {
        return;
    }
    let (depth, n) = deepest;
    let (u, v) = n.any_orthonormal_pair();
    let mut chosen = [Vec3::ZERO; 4];
    let mut count = 0;
    for direction in [u + v, u - v, -u + v, -u - v] {
        let point = *patch
            .iter()
            .max_by(|a, b| a.dot(direction).total_cmp(&b.dot(direction)))
            .unwrap();
        if chosen[..count]
            .iter()
            .any(|p| p.distance_squared(point) <= tolerance * tolerance)
        {
            continue;
        }
        chosen[count] = point;
        count += 1;
        out.push(Contact {
            point: cube.position + cube.rotation * (point - n * depth * 0.5),
            normal: cube.rotation * n,
            depth,
        });
    }
}

fn triangle_box_axis(triangle @ [a, b, c]: [Vec3; 3], h: Vec3) -> Option<(f64, Vec3)> {
    let edges = [b - a, c - b, a - c];
    let mut best = (f64::INFINITY, Vec3::Y);
    let triangle_normal = triangle_normal(triangle);
    let axes =
        std::iter::once(triangle_normal)
            .chain(Vec3::AXES)
            .chain(edges.into_iter().flat_map(|edge| {
                let direction = edge.normalize();
                Vec3::AXES.map(|axis| axis.cross(direction))
            }));
    for axis in axes {
        if axis.length_squared() < 1e-16 {
            continue;
        }
        let n = axis.normalize();
        let projection = Vec3::from_array(triangle.map(|p| p.dot(n)));
        let radius = h.dot(n.abs());
        let forward = projection.max_element() + radius;
        let backward = radius - projection.min_element();
        if forward < 0.0 || backward < 0.0 {
            return None;
        }
        let (depth, normal) = if forward <= backward {
            (forward, n)
        } else {
            (backward, -n)
        };
        if depth < best.0 - 1e-10 * h.max_element() {
            best = (depth, normal);
        }
    }
    Some(best)
}

fn clipped_triangle(triangle: [Vec3; 3], h: Vec3, out: &mut Vec<Vec3>) {
    let mut polygon = [Vec3::ZERO; 9];
    let mut scratch = [Vec3::ZERO; 9];
    polygon[..3].copy_from_slice(&triangle);
    let mut count = 3;
    for axis in 0..3 {
        for sign in [-1.0, 1.0] {
            count = clip_polygon(&polygon[..count], &mut scratch, |p| {
                p[axis] * sign - h[axis]
            });
            if count == 0 {
                return;
            }
            std::mem::swap(&mut polygon, &mut scratch);
        }
    }
    out.extend_from_slice(&polygon[..count]);
}

fn mesh_capsule(
    terrain: &RigidBody,
    mesh: &TriangleMesh,
    capsule: &RigidBody,
    radius: f64,
    h: f64,
    out: &mut Vec<Contact>,
) {
    let inverse = terrain.rotation.conjugate();
    let center = inverse * (capsule.position - terrain.position);
    let axis = inverse * (capsule.rotation * Vec3::Y);
    let [a, b] = [center - axis * h, center + axis * h];
    let bounds = Aabb {
        min: a.min(b),
        max: a.max(b),
    }
    .expanded(radius);
    let mut best = None;
    mesh.visit_triangles(bounds, |tri| {
        let (on_segment, on_triangle) = segment_triangle(a, b, tri);
        let delta = on_segment - on_triangle;
        let distance = delta.length();
        let depth = radius - distance;
        if depth < 0.0 || best.is_some_and(|(d, _, _)| d >= depth) {
            return;
        }
        let normal = delta.try_normalize().unwrap_or_else(|| {
            let n = triangle_normal(tri);
            n * (center - tri[0]).dot(n).signum()
        });
        best = Some((depth, normal, on_triangle));
    });
    if let Some((depth, normal, point)) = best {
        out.push(Contact {
            point: terrain.position + terrain.rotation * (point - normal * depth * 0.5),
            normal: terrain.rotation * normal,
            depth,
        });
    }
}

fn segment_triangle(a: Vec3, b: Vec3, triangle @ [x, y, z]: [Vec3; 3]) -> (Vec3, Vec3) {
    let segment = b - a;
    let normal = triangle_normal(triangle);
    let denominator = normal.dot(segment);
    if denominator != 0.0 {
        let t = (x - a).dot(normal) / denominator;
        if (0.0..=1.0).contains(&t) {
            let point = a + segment * t;
            let nearest = closest_triangle(point, triangle);
            if point.distance_squared(nearest) <= 1e-20 * segment.length_squared() {
                return (point, point);
            }
        }
    }
    let edge_pairs = [(x, y), (y, z), (z, x)].map(|(p, q)| closest_segments(a, segment, p, q - p));
    [a, b]
        .map(|p| (p, closest_triangle(p, triangle)))
        .into_iter()
        .chain(edge_pairs)
        .min_by(|(a, b), (c, d)| a.distance_squared(*b).total_cmp(&c.distance_squared(*d)))
        .expect("candidate segment endpoints and triangle edges")
}

fn parallel_cylinders(
    a: Vec3,
    b: Vec3,
    axis: Vec3,
    [ar, br]: [f64; 2],
    [ah, bh]: [f64; 2],
    out: &mut Vec<Contact>,
) {
    let delta = b - a;
    let axial = axis.dot(delta);
    let lateral = delta - axis * axial;
    let lateral_distance = lateral.length();
    let cap_depth = ah + bh - axial.abs();
    let side_depth = ar + br - lateral_distance;
    if cap_depth < 0.0 || side_depth < 0.0 {
        return;
    }
    let (point, normal, depth) = if cap_depth <= side_depth {
        let n = axis * axial.signum();
        let fraction = if lateral_distance > 0.0 {
            let low = (lateral_distance - br).max(-ar);
            let high = (lateral_distance + br).min(ar);
            0.5 * (low + high) / lateral_distance
        } else {
            0.0
        };
        (
            a + lateral * fraction + n * (ah - cap_depth * 0.5),
            n,
            cap_depth,
        )
    } else {
        let n = lateral
            .try_normalize()
            .unwrap_or_else(|| axis.any_orthonormal_vector());
        let low = (axial - bh).max(-ah);
        let high = (axial + bh).min(ah);
        (
            a + axis * ((low + high) * 0.5) + n * (ar - side_depth * 0.5),
            n,
            side_depth,
        )
    };
    out.push(Contact {
        point,
        normal,
        depth,
    });
}

fn convex_pair(a: &RigidBody, b: &RigidBody, out: &mut Vec<Contact>) {
    let [ab, bb] = [a, b].map(|b| Aabb::of(b).expect("finite convex shape"));
    let scale = (ab.max - ab.min).max(bb.max - bb.min).max_element();
    let a = PrimitiveSupport::new(a).expect("finite convex shape");
    let b = PrimitiveSupport::new(b).expect("finite convex shape");
    if let ConvexContact::Penetrating {
        depth,
        normal,
        point_a,
        point_b,
    } = convex::query(&a, &b, scale * 1e-6)
    {
        out.push(Contact {
            point: (point_a + point_b) * 0.5,
            normal,
            depth,
        });
    }
}

fn cylinder_plane_points(
    body: &RigidBody,
    normal: Vec3,
    radius: f64,
    half_height: f64,
    mut add: impl FnMut(Vec3),
) {
    let axis = body.rotation * Vec3::Y;
    let projection = normal.dot(axis);
    let cap = axis * half_height;
    let radial = -(normal - axis * projection);
    if radial.length_squared() < 1e-16 {
        let (x, y) = axis.any_orthonormal_pair();
        let center = body.position - cap * projection.signum();
        for side in [x, -x, y, -y] {
            add(center + side * radius);
        }
    } else {
        let point = body.position + radial.normalize() * radius;
        add(point - cap);
        add(point + cap);
    }
}

fn shape_sphere(cube: &RigidBody, sphere: &RigidBody, radius: f64, out: &mut Vec<Contact>) {
    let (closest, normal, distance) = if planar_pair(cube, sphere)
        && let Shape::Cuboid { half_extents } = cube.shape
    {
        project_planar_box(cube, half_extents, sphere.position)
    } else {
        project(cube, sphere.position)
    };
    let depth = radius - distance;
    if depth >= 0.0 {
        out.push(Contact {
            point: closest - normal * (depth * 0.5),
            normal,
            depth,
        });
    }
}

fn project_planar_box(cube: &RigidBody, h: Vec3, center: Vec3) -> (Vec3, Vec3, f64) {
    let p = cube.rotation.conjugate() * (center - cube.position);
    let mut closest = p.clamp(-h, h);
    closest.z = 0.0;
    let delta = p - closest;
    let (normal, distance) = if let Some(normal) = delta.try_normalize() {
        (normal, delta.length())
    } else {
        let gaps = h - p.abs();
        let axis = usize::from(gaps.y < gaps.x);
        let normal = Vec3::AXES[axis] * p[axis].signum();
        closest[axis] = normal[axis] * h[axis];
        (normal, -gaps[axis])
    };
    (
        cube.position + cube.rotation * closest,
        cube.rotation * normal,
        distance,
    )
}

fn capsule_sphere(capsule: &RigidBody, sphere: &RigidBody, radius: f64, out: &mut Vec<Contact>) {
    let Shape::Capsule {
        radius: capsule_radius,
        half_segment,
    } = capsule.shape
    else {
        unreachable!()
    };
    let u = capsule.rotation * Vec3::Y;
    let offset = (sphere.position - capsule.position)
        .dot(u)
        .clamp(-half_segment, half_segment);
    rounded_points(
        capsule.position + u * offset,
        sphere.position,
        capsule_radius,
        radius,
        capsule.rotation * Vec3::X,
        out,
    );
}

fn rounded_points(a: Vec3, b: Vec3, ar: f64, br: f64, fallback: Vec3, out: &mut Vec<Contact>) {
    let delta = b - a;
    let depth = ar + br - delta.length();
    if depth >= 0.0 {
        let normal = delta.try_normalize().unwrap_or(fallback);
        out.push(Contact {
            point: a + normal * (ar - depth * 0.5),
            normal,
            depth,
        });
    }
}

/// Closest pair on two finite segments, including zero-length segments.
fn closest_segments(a: Vec3, u: Vec3, b: Vec3, v: Vec3) -> (Vec3, Vec3) {
    let w = a - b;
    let uu = u.length_squared();
    let vv = v.length_squared();
    let uv = u.dot(v);
    let uw = u.dot(w);
    let vw = v.dot(w);
    let denominator = u.cross(v).length_squared();
    let s = if denominator > uu * vv * 1e-16 {
        ((uv * vw - vv * uw) / denominator).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let t = if vv > 0.0 {
        ((uv * s + vw) / vv).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let s = if uu > 0.0 {
        ((uv * t - uw) / uu).clamp(0.0, 1.0)
    } else {
        0.0
    };
    (a + s * u, b + t * v)
}

fn box_capsule(cube: &RigidBody, h: Vec3, capsule: &RigidBody, out: &mut Vec<Contact>) {
    let Shape::Capsule {
        radius,
        half_segment,
    } = capsule.shape
    else {
        unreachable!()
    };
    let inverse = cube.rotation.conjugate();
    let center = inverse * (capsule.position - cube.position);
    let axis = inverse * (capsule.rotation * Vec3::Y);
    let origin = center - axis * half_segment;
    let segment = axis * (2.0 * half_segment);
    let (boundaries, count) = box_intervals(origin, segment, h, 1.0);
    let mut best = (f64::INFINITY, center, center);
    for interval in boundaries[..count].windows(2) {
        let [start, end] = [interval[0], interval[1]];
        let sample = origin + segment * ((start + end) * 0.5);
        let fixed = sample.abs().cmpgt(h);
        let feature = sample.clamp(-h, h);
        let delta = Vec3::select(fixed, origin - feature, Vec3::ZERO);
        let motion = Vec3::select(fixed, segment, Vec3::ZERO);
        let t = if motion == Vec3::ZERO {
            (start + end) * 0.5
        } else {
            (-delta.dot(motion) / motion.length_squared()).clamp(start, end)
        };
        let p = origin + segment * t;
        let q = p.clamp(-h, h);
        let d2 = p.distance_squared(q);
        if d2 < best.0 {
            best = (d2, p, q);
        }
    }
    let (d2, p, q) = best;
    if d2 > radius * radius {
        return;
    }
    let (depth, normal, point) = if let Some(n) = (p - q).try_normalize() {
        let depth = radius - d2.sqrt();
        (depth, n, q - n * depth * 0.5)
    } else {
        // Segment intersects the box. The configuration obstacle is a box
        // extruded by the segment, then rounded by the capsule radius. From
        // inside the polytope, the closest supporting face gives the exit.
        let mut best = (f64::INFINITY, Vec3::X);
        for direction in Vec3::AXES
            .into_iter()
            .chain(Vec3::AXES.map(|v| v.cross(axis)))
        {
            let Some(n) = direction.try_normalize() else {
                continue;
            };
            let projection = center.dot(n);
            let depth =
                h.dot(n.abs()) + half_segment * axis.dot(n).abs() + radius - projection.abs();
            if depth < best.0 {
                best = (depth, n * projection.signum());
            }
        }
        let (depth, n) = best;
        let sign = if axis.dot(n).abs() < 1e-10 {
            0.0
        } else {
            axis.dot(n).signum()
        };
        let feature = center - axis * (half_segment * sign) - n * radius;
        (depth, n, feature + n * (depth * 0.5))
    };
    out.push(Contact {
        point: cube.position + cube.rotation * point,
        normal: cube.rotation * normal,
        depth,
    });
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

fn box_box(a: Obb, b: Obb, planar: bool, out: &mut Vec<Contact>) {
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
    for i in 0..if planar { 2 } else { 3 } {
        if !test(a.axes[i], Feature::FaceA(i)) || !test(b.axes[i], Feature::FaceB(i)) {
            return;
        }
    }
    if !planar {
        for i in 0..3 {
            for j in 0..3 {
                if !test(a.axes[i].cross(b.axes[j]), Feature::Edges(i, j)) {
                    return;
                }
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
            let (p, q) = closest_segments(a0, a1 - a0, b0, b1 - b0);
            out.push(Contact {
                point: (p + q) * 0.5,
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
            count = clip_polygon(&polygon[..count], &mut scratch, distance);
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

fn clip_polygon(polygon: &[Vec3], scratch: &mut [Vec3], distance: impl Fn(Vec3) -> f64) -> usize {
    let mut previous = polygon[polygon.len() - 1];
    let mut last = distance(previous);
    let mut written = 0;
    for &point in polygon {
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
    written
}
