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
            Shape::Capsule {
                radius,
                half_segment,
            } => (body.rotation * Vec3::Y).abs() * half_segment + Vec3::splat(radius),
            Shape::Cylinder {
                radius,
                half_height,
            } => {
                let u = body.rotation * Vec3::Y;
                let v = body.rotation * Vec3::X;
                let w = body.rotation * Vec3::Z;
                u.abs() * half_height
                    + Vec3::from_array((v * v + w * w).to_array().map(f64::sqrt)) * radius
            }
            Shape::Hull(ref hull) => return Some(hull.surface().bounds().transformed(body)),
            Shape::Mesh(ref mesh) => return Some(mesh.bounds().transformed(body)),
            Shape::Compound(ref shape) => return Some(shape.bounds().transformed(body)),
            Shape::Plane => return None,
        };
        Some(Self {
            min: body.position - extent,
            max: body.position + extent,
        })
    }

    fn transformed(self, body: &RigidBody) -> Self {
        let center = body.position + body.rotation * self.center();
        let h = (self.max - self.min) * 0.5;
        let extent = (body.rotation * Vec3::X).abs() * h.x
            + (body.rotation * Vec3::Y).abs() * h.y
            + (body.rotation * Vec3::Z).abs() * h.z;
        Self {
            min: center - extent,
            max: center + extent,
        }
    }

    pub fn intersects(self, other: Self) -> bool {
        self.min.cmple(other.max).all() && other.min.cmple(self.max).all()
    }
}

pub(crate) fn ray_body(body: &RigidBody, origin: Vec3, direction: Vec3) -> Option<(f64, Vec3)> {
    let relative = origin - body.position;
    match body.shape {
        Shape::Sphere { radius } => {
            let t = sphere_roots(relative, direction, radius)?
                .into_iter()
                .filter(|&t| t >= 0.0)
                .min_by(f64::total_cmp)?;
            Some((t, (relative + direction * t).normalize_or_zero()))
        }
        Shape::Capsule {
            radius,
            half_segment,
        } => {
            let inverse = body.rotation.conjugate();
            let (distance, normal) = ray_capsule(
                inverse * relative,
                inverse * direction,
                radius,
                half_segment,
            )?;
            Some((distance, body.rotation * normal))
        }
        Shape::Cylinder {
            radius,
            half_height,
        } => {
            let inverse = body.rotation.conjugate();
            let (t, n) =
                ray_cylinder(inverse * relative, inverse * direction, radius, half_height)?;
            Some((t, body.rotation * n))
        }
        Shape::Compound(ref shape) => shape
            .parts()
            .iter()
            .filter_map(|part| ray_body(&body.collider_part(part), origin, direction))
            .min_by(|a, b| a.0.total_cmp(&b.0)),
        Shape::Hull(ref hull) => {
            let inverse = body.rotation.conjugate();
            let (t, n) = hull.ray(inverse * relative, inverse * direction)?;
            Some((t, body.rotation * n))
        }
        Shape::Mesh(ref mesh) => {
            let inverse = body.rotation.conjugate();
            let (t, n) = mesh.ray(inverse * relative, inverse * direction)?;
            Some((t, body.rotation * n))
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
        Shape::Capsule {
            radius,
            half_segment,
        } => {
            let axis = body.rotation * Vec3::Y;
            let segment = axis * offset.dot(axis).clamp(-half_segment, half_segment);
            let delta = offset - segment;
            let normal = delta.try_normalize().unwrap_or(body.rotation * Vec3::X);
            (
                body.position + segment + normal * radius,
                normal,
                delta.length() - radius,
            )
        }
        Shape::Cylinder {
            radius,
            half_height,
        } => {
            let local = body.rotation.conjugate() * offset;
            let radial = Vec3::new(local.x, 0.0, local.z);
            let radial_length = radial.length();
            let u = radial.try_normalize().unwrap_or(Vec3::X);
            let wall_gap = radial_length - radius;
            let cap_gap = local.y.abs() - half_height;
            let up = Vec3::Y * local.y.signum();
            let (normal, distance) = if wall_gap > 0.0 || cap_gap > 0.0 {
                let delta = u * wall_gap.max(0.0) + up * cap_gap.max(0.0);
                (delta.normalize(), delta.length())
            } else if wall_gap >= cap_gap {
                (u, wall_gap)
            } else {
                (up, cap_gap)
            };
            let normal = body.rotation * normal;
            (point - distance * normal, normal, distance)
        }
        Shape::Compound(ref shape) => shape
            .parts()
            .iter()
            .map(|part| project(&body.collider_part(part), point))
            .min_by(|a, b| a.2.total_cmp(&b.2))
            .expect("validated nonempty compound"),
        Shape::Hull(ref hull) => {
            let (p, n, d) = hull.project(body.rotation.conjugate() * offset);
            (body.position + body.rotation * p, body.rotation * n, d)
        }
        Shape::Mesh(ref mesh) => {
            let (p, n, d) = mesh.project(body.rotation.conjugate() * offset);
            (body.position + body.rotation * p, body.rotation * n, d)
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
        Shape::Capsule {
            radius: target_radius,
            half_segment,
        } => {
            let inverse = body.rotation.conjugate();
            let (t, normal) = ray_capsule(
                inverse * offset,
                inverse * direction,
                radius + target_radius,
                half_segment,
            )?;
            let normal = body.rotation * normal;
            if direction.dot(normal) >= 0.0 {
                return None;
            }
            (t, normal)
        }
        Shape::Cylinder { .. } | Shape::Hull(_) => {
            return conservative_sphere_cast(body, origin, direction, radius, max_distance);
        }
        Shape::Compound(ref shape) => {
            return shape
                .parts()
                .iter()
                .filter_map(|part| {
                    sweep_sphere(
                        &body.collider_part(part),
                        origin,
                        direction,
                        radius,
                        max_distance,
                    )
                })
                .min_by(|a, b| a.0.total_cmp(&b.0));
        }
        Shape::Mesh(ref mesh) => {
            let inverse = body.rotation.conjugate();
            let (t, n) =
                mesh.cast_sphere(inverse * offset, inverse * direction, radius, max_distance)?;
            (t, body.rotation * n)
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

fn sphere_roots(offset: Vec3, direction: Vec3, radius: f64) -> Option<[f64; 2]> {
    let a = direction.length_squared();
    let b = offset.dot(direction);
    let c = offset.length_squared() - radius * radius;
    let discriminant = a * radius * radius - offset.cross(direction).length_squared();
    if a <= 0.0 || discriminant < 0.0 {
        return None;
    }
    let q = -b - b.signum() * discriminant.sqrt();
    Some(if q == 0.0 {
        [-b / a; 2]
    } else {
        [q / a, c / q]
    })
}

fn ray_cylinder(o: Vec3, d: Vec3, radius: f64, h: f64) -> Option<(f64, Vec3)> {
    let mut best = None;
    let radial = Vec3::new(o.x, 0.0, o.z);
    let radial_velocity = Vec3::new(d.x, 0.0, d.z);
    if let Some(roots) = sphere_roots(radial, radial_velocity, radius) {
        for t in roots {
            if t >= 0.0
                && (o.y + t * d.y).abs() <= h
                && best.is_none_or(|(distance, _)| t < distance)
            {
                best = Some((t, (radial + t * radial_velocity).normalize_or_zero()));
            }
        }
    }
    if d.y != 0.0 {
        for side in [-1.0, 1.0] {
            let t = (h * side - o.y) / d.y;
            if t >= 0.0
                && (radial + t * radial_velocity).length_squared() <= radius * radius
                && best.is_none_or(|(distance, _)| t < distance)
            {
                best = Some((t, Vec3::Y * side));
            }
        }
    }
    best
}

fn conservative_sphere_cast(
    body: &RigidBody,
    origin: Vec3,
    direction: Vec3,
    radius: f64,
    max_distance: f64,
) -> Option<(f64, Vec3)> {
    let tolerance = radius * 1e-8;
    let mut distance = 0.0;
    for _ in 0..64 {
        let (_, normal, surface_distance) = project(body, origin + direction * distance);
        let gap = surface_distance - radius;
        let approach = -normal.dot(direction);
        if approach <= 0.0 || (distance == 0.0 && gap < -tolerance) {
            return None;
        }
        if gap <= tolerance {
            return Some((distance, normal));
        }
        distance += gap / approach;
        if !distance.is_finite() || distance > max_distance {
            return None;
        }
    }
    None
}

pub(crate) fn ray_capsule(o: Vec3, d: Vec3, radius: f64, h: f64) -> Option<(f64, Vec3)> {
    let mut best: Option<(f64, Vec3)> = None;
    let mut consider = |offset: Vec3, direction: Vec3, valid: &dyn Fn(f64) -> bool| {
        let Some(roots) = sphere_roots(offset, direction, radius) else {
            return;
        };
        for t in roots {
            if t >= 0.0 && valid(t) && best.is_none_or(|(distance, _)| t < distance) {
                best = Some((t, (offset + direction * t).normalize_or_zero()));
            }
        }
    };
    consider(Vec3::new(o.x, 0.0, o.z), Vec3::new(d.x, 0.0, d.z), &|t| {
        (o.y + t * d.y).abs() <= h
    });
    for side in [-1.0, 1.0] {
        consider(o - Vec3::Y * (h * side), d, &|t| {
            (o.y + t * d.y) * side >= h
        });
    }
    best
}

/// Time values at which a line crosses any box coordinate plane.
pub(crate) fn box_intervals(origin: Vec3, direction: Vec3, h: Vec3, end: f64) -> ([f64; 8], usize) {
    let mut boundaries = [0.0; 8];
    boundaries[1] = end;
    let mut count = 2;
    for axis in 0..3 {
        if direction[axis] == 0.0 {
            continue;
        }
        for sign in [-1.0, 1.0] {
            let t = (sign * h[axis] - origin[axis]) / direction[axis];
            if t > 0.0 && t < end {
                boundaries[count] = t;
                count += 1;
            }
        }
    }
    boundaries[..count].sort_unstable_by(f64::total_cmp);
    (boundaries, count)
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
    let (boundaries, count) = box_intervals(origin, direction, h, max_distance);
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

/// Flat median-split bounding volume hierarchy. Leaves refer to dense indices
/// owned by the caller.
#[derive(Debug, Default)]
pub(crate) struct Bvh {
    nodes: Vec<Node>,
}

#[derive(Debug)]
struct Node {
    bounds: Aabb,
    kind: NodeKind,
}

#[derive(Debug)]
enum NodeKind {
    Leaf(usize),
    Branch { right: usize },
}

#[derive(Clone, Copy)]
pub(crate) struct Proxy {
    pub index: usize,
    pub bounds: Aabb,
}

impl Aabb {
    fn center(self) -> Vec3 {
        self.min * 0.5 + self.max * 0.5
    }

    pub(crate) fn union(self, other: Self) -> Self {
        Self {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }

    pub(crate) fn expanded(self, radius: f64) -> Self {
        let extent = Vec3::splat(radius);
        Self {
            min: self.min - extent,
            max: self.max + extent,
        }
    }

    fn distance_squared(self, point: Vec3) -> f64 {
        point.distance_squared(point.clamp(self.min, self.max))
    }

    pub(crate) fn hit_ray(self, origin: Vec3, direction: Vec3, mut far: f64) -> bool {
        let mut near: f64 = 0.0;
        for axis in 0..3 {
            if direction[axis] == 0.0 {
                if origin[axis] < self.min[axis] || origin[axis] > self.max[axis] {
                    return false;
                }
                continue;
            }
            let a = (self.min[axis] - origin[axis]) / direction[axis];
            let b = (self.max[axis] - origin[axis]) / direction[axis];
            near = near.max(a.min(b));
            far = far.min(a.max(b));
            if near > far {
                return false;
            }
        }
        true
    }
}

impl Bvh {
    pub fn rebuild(&mut self, proxies: &mut [Proxy]) {
        self.nodes.clear();
        if !proxies.is_empty() {
            self.build(proxies);
        }
    }

    fn build(&mut self, proxies: &mut [Proxy]) -> usize {
        let (first, rest) = proxies.split_first().expect("nonempty partition");
        let index = self.nodes.len();
        let bounds = rest
            .iter()
            .fold(first.bounds, |acc, proxy| acc.union(proxy.bounds));
        self.nodes.push(Node {
            bounds,
            kind: NodeKind::Leaf(first.index),
        });
        if proxies.len() == 1 {
            return index;
        }
        let mid = proxies.len() / 2;
        let extent = bounds.max - bounds.min;
        let axis = if extent.x >= extent.y && extent.x >= extent.z {
            0
        } else if extent.y >= extent.z {
            1
        } else {
            2
        };
        proxies.select_nth_unstable_by(mid, |a, b| {
            a.bounds.center()[axis]
                .total_cmp(&b.bounds.center()[axis])
                .then(a.index.cmp(&b.index))
        });
        let (left, right) = proxies.split_at_mut(mid);
        self.build(left);
        let right = self.build(right);
        self.nodes[index].kind = NodeKind::Branch { right };
        index
    }

    pub fn visit(
        &self,
        mut intersects: impl FnMut(Aabb) -> bool,
        mut visit_leaf: impl FnMut(usize),
    ) {
        if !self.nodes.is_empty() {
            self.visit_node(0, &mut intersects, &mut visit_leaf);
        }
    }

    pub fn nearest<T>(
        &self,
        point: Vec3,
        mut evaluate: impl FnMut(usize) -> (f64, T),
    ) -> Option<(f64, usize, T)> {
        let mut best = None;
        if !self.nodes.is_empty() {
            self.nearest_node(0, point, &mut evaluate, &mut best);
        }
        best
    }

    fn nearest_node<T>(
        &self,
        i: usize,
        point: Vec3,
        evaluate: &mut impl FnMut(usize) -> (f64, T),
        best: &mut Option<(f64, usize, T)>,
    ) {
        let node = &self.nodes[i];
        if best
            .as_ref()
            .is_some_and(|(d, _, _)| *d < node.bounds.distance_squared(point))
        {
            return;
        }
        match node.kind {
            NodeKind::Leaf(index) => {
                let (distance, value) = evaluate(index);
                if best
                    .as_ref()
                    .is_none_or(|(d, j, _)| distance < *d || (distance == *d && index < *j))
                {
                    *best = Some((distance, index, value));
                }
            }
            NodeKind::Branch { right } => {
                let left = i + 1;
                let dl = self.nodes[left].bounds.distance_squared(point);
                let dr = self.nodes[right].bounds.distance_squared(point);
                let [first, second] = if dl <= dr {
                    [left, right]
                } else {
                    [right, left]
                };
                self.nearest_node(first, point, evaluate, best);
                self.nearest_node(second, point, evaluate, best);
            }
        }
    }

    fn visit_node(
        &self,
        i: usize,
        intersects: &mut impl FnMut(Aabb) -> bool,
        visit_leaf: &mut impl FnMut(usize),
    ) {
        let node = &self.nodes[i];
        if !intersects(node.bounds) {
            return;
        }
        match node.kind {
            NodeKind::Leaf(index) => visit_leaf(index),
            NodeKind::Branch { right } => {
                self.visit_node(i + 1, intersects, visit_leaf);
                self.visit_node(right, intersects, visit_leaf);
            }
        }
    }
}

use crate::{BodyHandle, PhysicsWorld, QueryFilter, Ray, RayHit};

/// A reusable read-only acceleration snapshot for batches of world queries.
/// It borrows the world so geometry stays current for the lifetime of the index.
pub struct SpatialIndex<'a> {
    world: &'a PhysicsWorld,
    bodies: Vec<(BodyHandle, &'a RigidBody)>,
    bvh: Bvh,
    planes: Vec<usize>,
}

impl<'a> SpatialIndex<'a> {
    pub(crate) fn new(world: &'a PhysicsWorld) -> Self {
        let bodies: Vec<_> = world.bodies().collect();
        let mut bvh = Bvh::default();
        let mut proxies = Vec::with_capacity(bodies.len());
        let mut planes = Vec::new();
        for (index, &(_, body)) in bodies.iter().enumerate() {
            if let Some(bounds) = Aabb::of(body) {
                proxies.push(Proxy { index, bounds });
            } else {
                planes.push(index);
            }
        }
        bvh.rebuild(&mut proxies);
        Self {
            world,
            bodies,
            bvh,
            planes,
        }
    }

    pub fn world(&self) -> &'a PhysicsWorld {
        self.world
    }

    pub fn raycast(&self, ray: Ray, filter: QueryFilter) -> Option<RayHit> {
        let direction = ray.direction.try_normalize()?;
        if !ray.origin.is_finite() || ray.max_distance.is_nan() || ray.max_distance < 0.0 {
            return None;
        }
        let mut best: Option<RayHit> = None;
        let mut test = |index: usize| {
            let (body, b) = self.bodies[index];
            if !filter.accepts(body, b) {
                return;
            }
            if let Some((distance, normal)) = ray_body(b, ray.origin, direction)
                && distance.is_finite()
                && distance <= ray.max_distance
                && best.is_none_or(|hit| {
                    distance < hit.distance || (distance == hit.distance && body < hit.body)
                })
            {
                best = Some(RayHit {
                    body,
                    position: ray.origin + direction * distance,
                    normal,
                    distance,
                });
            }
        };
        self.bvh.visit(
            |aabb| aabb.hit_ray(ray.origin, direction, ray.max_distance),
            &mut test,
        );
        for &plane in &self.planes {
            test(plane);
        }
        best
    }

    /// Writes intersecting finite collider AABBs in deterministic tree order.
    pub fn query_aabb(&self, bounds: Aabb, filter: QueryFilter, mut visit: impl FnMut(BodyHandle)) {
        self.bvh.visit(
            |aabb| aabb.intersects(bounds),
            |index| {
                let (id, body) = self.bodies[index];
                if filter.accepts(id, body) {
                    visit(id);
                }
            },
        );
    }

    pub fn raycast_batch<'b>(
        &'b self,
        rays: &'b [Ray],
        filter: QueryFilter,
    ) -> impl ExactSizeIterator<Item = Option<RayHit>> + 'b {
        rays.iter().map(move |&ray| self.raycast(ray, filter))
    }
}
