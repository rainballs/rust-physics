//! Deterministic rigid-body stepping, contacts, collision events, and spatial queries.
use std::collections::BTreeMap;
use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::collision::{Contact, contacts};
use crate::joints::{self, Joint, JointHandle, JointRow};
use crate::spatial::{Aabb, project, ray_body, sweep_sphere};
use crate::{BodyKind, RigidBody, Shape, Vec3};

static NEXT_WORLD: AtomicU64 = AtomicU64::new(1);

/// Stable identifier scoped to the world that issued it. Removed IDs are never reused.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BodyHandle {
    world: u64,
    serial: u64,
}

#[derive(Clone, Debug)]
pub struct Settings {
    pub gravity: Vec3,
    pub fixed_dt: f64,
    pub substeps: usize,
    pub iterations: usize,
    pub contact_slop: f64,
    pub position_correction: f64,
    pub bounce_threshold: f64,
    pub sleep_speed: f64,
    pub sleep_time: f64,
    pub ccd: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            gravity: Vec3::new(0.0, -9.81, 0.0),
            fixed_dt: 1.0 / 60.0,
            substeps: 2,
            iterations: 12,
            contact_slop: 0.001,
            position_correction: 0.65,
            bounce_threshold: 0.5,
            sleep_speed: 0.035,
            sleep_time: 0.6,
            ccd: true,
        }
    }
}

impl Settings {
    fn valid(&self) -> bool {
        self.gravity.is_finite()
            && self.fixed_dt.is_finite()
            && self.fixed_dt > 0.0
            && self.substeps > 0
            && self.iterations > 0
            && (0.0..=1.0).contains(&self.position_correction)
            && [
                self.contact_slop,
                self.bounce_threshold,
                self.sleep_speed,
                self.sleep_time,
            ]
            .iter()
            .all(|&v| v.is_finite() && v >= 0.0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StepError {
    InvalidTimestep,
    InvalidSettings,
}

impl std::fmt::Display for StepError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::InvalidTimestep => "timestep must be finite and positive after subdivision",
            Self::InvalidSettings => "invalid world settings",
        })
    }
}
impl std::error::Error for StepError {}

#[derive(Clone, Copy, Debug, Default)]
pub struct StepStats {
    /// Candidate pairs examined by the narrow phase across all substeps.
    pub candidate_pairs: usize,
    pub contact_points: usize,
    pub sleeping_bodies: usize,
    pub ccd_impacts: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CollisionEvent {
    pub a: BodyHandle,
    pub b: BodyHandle,
    pub sensor: bool,
    pub started: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct RayHit {
    pub body: BodyHandle,
    pub position: Vec3,
    pub normal: Vec3,
    pub distance: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct PointHit {
    pub body: BodyHandle,
    pub position: Vec3,
    pub normal: Vec3,
    /// Signed surface distance, negative for points inside the collider.
    pub distance: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
    pub max_distance: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct QueryFilter {
    pub layers: u32,
    pub include_sensors: bool,
    pub exclude: Option<BodyHandle>,
}

impl Default for QueryFilter {
    fn default() -> Self {
        Self {
            layers: u32::MAX,
            include_sensors: true,
            exclude: None,
        }
    }
}

impl QueryFilter {
    pub(crate) fn accepts(self, handle: BodyHandle, b: &RigidBody) -> bool {
        b.layer & self.layers != 0
            && (self.include_sensors || !b.sensor)
            && self.exclude != Some(handle)
    }
}

type Pair = (BodyHandle, BodyHandle);

#[derive(Clone, Copy)]
struct Cached {
    local: [Vec3; 2],
    normal: Vec3,
    normal_impulse: f64,
    friction: Vec3,
}

struct Row {
    contact: Contact,
    offsets: [Vec3; 2],
    normal_mass: f64,
    target_speed: f64,
    cache: Cached,
}

struct Manifold {
    pair: Pair,
    indices: [usize; 2],
    points: Range<usize>,
    rows: Range<usize>,
}

#[derive(Default)]
struct SolverScratch {
    manifolds: Vec<Manifold>,
    rows: Vec<Row>,
    joint_rows: Vec<JointRow>,
    joint_batches: Vec<([usize; 2], Range<usize>)>,
    next_cache: Vec<(Pair, Range<usize>)>,
    next_impulses: Vec<Cached>,
    active: Vec<(Pair, bool)>,
    islands: Islands,
}

#[derive(Default)]
struct Islands {
    nodes: Vec<IslandNode>,
}

#[derive(Default)]
struct IslandNode {
    parent: usize,
    rank: u8,
    grounded: bool,
    awake: bool,
    busy: bool,
    min_quiet_time: f64,
}

impl Islands {
    fn reset(&mut self, count: usize) {
        self.nodes.clear();
        self.nodes.extend((0..count).map(|parent| IslandNode {
            parent,
            min_quiet_time: f64::INFINITY,
            ..IslandNode::default()
        }));
    }

    fn root(&mut self, mut i: usize) -> usize {
        while self.nodes[i].parent != i {
            self.nodes[i].parent = self.nodes[self.nodes[i].parent].parent;
            i = self.nodes[i].parent;
        }
        i
    }

    fn join(&mut self, a: usize, b: usize) {
        let mut a = self.root(a);
        let mut b = self.root(b);
        if a == b {
            return;
        }
        if self.nodes[a].rank < self.nodes[b].rank {
            std::mem::swap(&mut a, &mut b);
        }
        self.nodes[b].parent = a;
        if self.nodes[a].rank == self.nodes[b].rank {
            self.nodes[a].rank += 1;
        }
    }
}

#[derive(Clone, Copy)]
struct Proxy {
    index: usize,
    bounds: Aabb,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InvalidJoint;

impl std::fmt::Display for InvalidJoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("invalid joint parameters or body handles")
    }
}
impl std::error::Error for InvalidJoint {}

pub struct PhysicsWorld {
    id: u64,
    next: u64,
    settings: Settings,
    bodies: Vec<(BodyHandle, RigidBody)>,
    indices: BTreeMap<BodyHandle, usize>,
    cache: Vec<(Pair, Range<usize>)>,
    impulses: Vec<Cached>,
    scratch: SolverScratch,
    previous_h: Option<f64>,
    touching: Vec<(Pair, bool)>,
    events: Vec<CollisionEvent>,
    debug_contacts: Vec<(BodyHandle, BodyHandle, Contact)>,
    proxies: Vec<Proxy>,
    planes: Vec<usize>,
    candidates: Vec<[usize; 2]>,
    ccd_targets: Vec<usize>,
    joints: BTreeMap<JointHandle, Joint>,
}

impl Default for PhysicsWorld {
    fn default() -> Self {
        Self::new()
    }
}

impl PhysicsWorld {
    pub fn new() -> Self {
        Self {
            id: NEXT_WORLD
                .try_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
                .expect("world identifiers exhausted"),
            next: 0,
            settings: Settings::default(),
            bodies: Vec::new(),
            indices: BTreeMap::new(),
            cache: Vec::new(),
            impulses: Vec::new(),
            scratch: SolverScratch::default(),
            previous_h: None,
            touching: Vec::new(),
            events: Vec::new(),
            debug_contacts: Vec::new(),
            proxies: Vec::new(),
            planes: Vec::new(),
            candidates: Vec::new(),
            ccd_targets: Vec::new(),
            joints: BTreeMap::new(),
        }
    }

    pub fn settings(&self) -> &Settings {
        &self.settings
    }

    pub fn set_settings(&mut self, settings: Settings) -> Result<(), StepError> {
        if !settings.valid() {
            return Err(StepError::InvalidSettings);
        }
        self.settings = settings;
        self.wake_all();
        Ok(())
    }

    pub fn set_gravity(&mut self, gravity: Vec3) -> Result<(), StepError> {
        self.set_settings(Settings {
            gravity,
            ..self.settings.clone()
        })
    }

    fn wake_all(&mut self) {
        for (_, body) in &mut self.bodies {
            body.wake();
        }
    }

    fn allocate_handle(&mut self) -> BodyHandle {
        let handle = BodyHandle {
            world: self.id,
            serial: self.next,
        };
        self.next = self
            .next
            .checked_add(1)
            .expect("world identifiers exhausted");
        handle
    }

    pub fn add_joint(&mut self, joint: Joint) -> Result<JointHandle, InvalidJoint> {
        if !joint.valid() || joint.bodies.iter().any(|id| !self.indices.contains_key(id)) {
            return Err(InvalidJoint);
        }
        let handle = JointHandle(self.allocate_handle());
        self.joints.insert(handle, joint);
        for body in joint.bodies {
            self.body_mut(body).expect("validated joint").wake();
        }
        Ok(handle)
    }

    pub fn joint(&self, handle: JointHandle) -> Option<&Joint> {
        self.joints.get(&handle)
    }

    pub fn remove_joint(&mut self, handle: JointHandle) -> Option<Joint> {
        let joint = self.joints.remove(&handle)?;
        for body in joint.bodies {
            self.body_mut(body).expect("live joint").wake();
        }
        Some(joint)
    }

    pub fn add_body(&mut self, mut body: RigidBody) -> BodyHandle {
        let handle = self.allocate_handle();
        self.indices.insert(handle, self.bodies.len());
        body.wake();
        self.bodies.push((handle, body));
        handle
    }

    pub fn remove_body(&mut self, handle: BodyHandle) -> Option<RigidBody> {
        let index = self.indices.remove(&handle)?;
        let (_, removed) = self.bodies.swap_remove(index);
        if let Some((moved, _)) = self.bodies.get(index) {
            self.indices.insert(*moved, index);
        }
        self.invalidate_cache(handle);
        self.joints
            .retain(|_, joint| !joint.bodies.contains(&handle));
        self.debug_contacts
            .retain(|&(a, b, _)| a != handle && b != handle);
        self.wake_all();
        Some(removed)
    }

    pub fn body(&self, handle: BodyHandle) -> Option<&RigidBody> {
        Some(&self.bodies[*self.indices.get(&handle)?].1)
    }

    pub fn body_mut(&mut self, handle: BodyHandle) -> Option<&mut RigidBody> {
        let index = *self.indices.get(&handle)?;
        self.invalidate_cache(handle);
        self.bodies[index].1.wake();
        Some(&mut self.bodies[index].1)
    }

    fn invalidate_cache(&mut self, handle: BodyHandle) {
        self.cache
            .retain(|((a, b), _)| *a != handle && *b != handle);
    }

    pub fn bodies(&self) -> impl ExactSizeIterator<Item = (BodyHandle, &RigidBody)> {
        self.bodies.iter().map(|(id, body)| (*id, body))
    }

    /// Transitions observed during the last successful step, in substep order.
    pub fn events(&self) -> &[CollisionEvent] {
        &self.events
    }

    /// World-space contacts sampled before the final substep's penetration correction.
    pub fn debug_contacts(&self) -> &[(BodyHandle, BodyHandle, Contact)] {
        &self.debug_contacts
    }

    pub fn raycast(
        &self,
        origin: Vec3,
        direction: Vec3,
        max_distance: f64,
        filter: QueryFilter,
    ) -> Option<RayHit> {
        let direction = direction.try_normalize()?;
        if !origin.is_finite() || max_distance.is_nan() || max_distance < 0.0 {
            return None;
        }
        self.bodies()
            .filter(|&(id, b)| filter.accepts(id, b))
            .filter_map(|(body, b)| {
                let (distance, normal) = ray_body(b, origin, direction)?;
                (distance.is_finite() && distance <= max_distance).then_some(RayHit {
                    body,
                    normal,
                    distance,
                    position: origin + direction * distance,
                })
            })
            .min_by(|a, b| a.distance.total_cmp(&b.distance).then(a.body.cmp(&b.body)))
    }

    pub fn raycast_batch<'a>(
        &'a self,
        rays: &'a [Ray],
        filter: QueryFilter,
    ) -> impl ExactSizeIterator<Item = Option<RayHit>> + 'a {
        rays.iter()
            .map(move |ray| self.raycast(ray.origin, ray.direction, ray.max_distance, filter))
    }

    /// Finds the nearest collider surface, including the surface of a collider
    /// containing the point. Selection uses absolute surface distance.
    pub fn nearest_surface(&self, point: Vec3, filter: QueryFilter) -> Option<PointHit> {
        if !point.is_finite() {
            return None;
        }
        self.bodies()
            .filter(|&(id, b)| filter.accepts(id, b))
            .map(|(body, b)| {
                let (position, normal, distance) = project(b, point);
                PointHit {
                    body,
                    position,
                    normal,
                    distance,
                }
            })
            .min_by(|a, b| {
                a.distance
                    .abs()
                    .total_cmp(&b.distance.abs())
                    .then(a.body.cmp(&b.body))
            })
    }

    /// Exact sphere overlap against the supported collider shapes.
    pub fn overlap_sphere(
        &self,
        center: Vec3,
        radius: f64,
        filter: QueryFilter,
    ) -> impl Iterator<Item = BodyHandle> + '_ {
        self.bodies().filter_map(move |(id, b)| {
            (radius.is_finite()
                && radius >= 0.0
                && center.is_finite()
                && filter.accepts(id, b)
                && project(b, center).2 <= radius)
                .then_some(id)
        })
    }

    /// Casts a sphere and reports the first target surface hit. Starting
    /// overlaps are returned at zero distance.
    pub fn cast_sphere(
        &self,
        origin: Vec3,
        radius: f64,
        direction: Vec3,
        max_distance: f64,
        filter: QueryFilter,
    ) -> Option<RayHit> {
        let direction = direction.try_normalize()?;
        if !origin.is_finite()
            || !radius.is_finite()
            || radius < 0.0
            || max_distance.is_nan()
            || max_distance < 0.0
        {
            return None;
        }
        if radius == 0.0 {
            return self.raycast(origin, direction, max_distance, filter);
        }
        self.bodies()
            .filter(|&(id, b)| filter.accepts(id, b))
            .filter_map(|(body, b)| {
                let (position, normal, gap) = project(b, origin);
                if gap <= radius {
                    return Some(RayHit {
                        body,
                        position,
                        normal,
                        distance: 0.0,
                    });
                }
                let (distance, normal) = sweep_sphere(b, origin, direction, radius, max_distance)?;
                Some(RayHit {
                    body,
                    position: origin + direction * distance - normal * radius,
                    normal,
                    distance,
                })
            })
            .min_by(|a, b| a.distance.total_cmp(&b.distance).then(a.body.cmp(&b.body)))
    }

    /// Finite colliders whose bounding boxes intersect the given region.
    pub fn query_aabb(
        &self,
        bounds: Aabb,
        filter: QueryFilter,
    ) -> impl Iterator<Item = BodyHandle> + '_ {
        self.bodies().filter_map(move |(id, b)| {
            (filter.accepts(id, b) && Aabb::of(b).is_some_and(|a| a.intersects(bounds)))
                .then_some(id)
        })
    }

    pub fn step_fixed(&mut self) -> Result<StepStats, StepError> {
        self.step(self.settings.fixed_dt)
    }

    pub fn step(&mut self, dt: f64) -> Result<StepStats, StepError> {
        let h = dt / self.settings.substeps as f64;
        if !h.is_finite() || h <= 0.0 {
            return Err(StepError::InvalidTimestep);
        }
        let mut stats = StepStats::default();
        self.events.clear();
        for _ in 0..self.settings.substeps {
            self.substep(h, &mut stats);
        }
        for (_, body) in &mut self.bodies {
            body.force = Vec3::ZERO;
            body.torque = Vec3::ZERO;
            stats.sleeping_bodies += usize::from(body.asleep);
        }
        Ok(stats)
    }

    fn substep(&mut self, h: f64, stats: &mut StepStats) {
        for (_, body) in &mut self.bodies {
            if body.kind == BodyKind::Dynamic && !body.asleep {
                body.velocity += (self.settings.gravity * body.gravity_scale
                    + body.force * body.inverse_mass)
                    * h;
                body.angular_momentum += body.torque * h;
                body.velocity *= (-body.damping[0] * h).exp();
                body.angular_momentum *= (-body.damping[1] * h).exp();
            }
        }
        self.prepare_joints(h);
        for _ in 0..self.settings.iterations {
            self.solve_joints();
        }
        self.advance_poses(h, stats);
        self.collect_candidates();
        stats.candidate_pairs += self.candidates.len();

        self.build_manifolds(stats);
        if self.settings.sleep_time > 0.0 {
            self.prepare_islands();
        }
        self.prepare_solver(h);
        self.solve_rows();
        self.finish_contacts();
        if self.settings.sleep_time > 0.0 {
            self.update_sleep(h);
        }
        self.publish_events();
        self.previous_h = Some(h);
    }

    fn build_manifolds(&mut self, stats: &mut StepStats) {
        let scratch = &mut self.scratch;
        scratch.active.clear();
        scratch.manifolds.clear();
        self.debug_contacts.clear();
        let mut geometry = Vec::with_capacity(8);
        for &indices in &self.candidates {
            let [ai, bi] = indices;
            let (ah, a) = &self.bodies[ai];
            let (bh, b) = &self.bodies[bi];
            geometry.clear();
            contacts(a, b, &mut geometry);
            if geometry.is_empty() {
                continue;
            }
            let pair = (*ah, *bh);
            let sensor = a.sensor || b.sensor;
            scratch.active.push((pair, sensor));
            stats.contact_points += geometry.len();
            let start = self.debug_contacts.len();
            self.debug_contacts
                .extend(geometry.iter().map(|&c| (pair.0, pair.1, c)));
            if !sensor {
                scratch.manifolds.push(Manifold {
                    pair,
                    indices,
                    points: start..self.debug_contacts.len(),
                    rows: 0..0,
                });
            }
        }
    }

    fn prepare_islands(&mut self) {
        let islands = &mut self.scratch.islands;
        islands.reset(self.bodies.len());
        let kind = |i: usize| self.bodies[i].1.kind;
        for m in &self.scratch.manifolds {
            let [a, b] = m.indices;
            if kind(a) == BodyKind::Dynamic && kind(b) == BodyKind::Dynamic {
                islands.join(a, b);
            }
        }
        for joint in self.joints.values() {
            let [a, b] = joint.bodies.map(|id| self.indices[&id]);
            if kind(a) == BodyKind::Dynamic && kind(b) == BodyKind::Dynamic {
                islands.join(a, b);
            }
        }
        for m in &self.scratch.manifolds {
            let [a, b] = m.indices;
            for (dynamic, support) in [(a, b), (b, a)] {
                if kind(dynamic) == BodyKind::Dynamic && kind(support) != BodyKind::Dynamic {
                    let root = islands.root(dynamic);
                    islands.nodes[root].grounded = true;
                    islands.nodes[root].busy |= should_wake(&self.bodies[support].1);
                }
            }
        }
        for joint in self.joints.values() {
            for id in joint.bodies {
                let index = self.indices[&id];
                let root = islands.root(index);
                // Motors and spring forces can do work even from zero velocity.
                islands.nodes[root].busy = true;
            }
        }
        for (i, (_, body)) in self.bodies.iter().enumerate() {
            if body.kind == BodyKind::Dynamic && !body.asleep {
                let root = islands.root(i);
                islands.nodes[root].awake = true;
            }
        }
        for (i, (_, body)) in self.bodies.iter_mut().enumerate() {
            if body.asleep {
                let root = islands.root(i);
                let island = &islands.nodes[root];
                if island.awake || island.busy || !island.grounded {
                    body.wake();
                }
            }
        }
    }

    fn prepare_joints(&mut self, h: f64) {
        let scratch = &mut self.scratch;
        scratch.joint_rows.clear();
        scratch.joint_batches.clear();
        for joint in self.joints.values() {
            let [ai, bi] = joint.bodies.map(|id| self.indices[&id]);
            let start = scratch.joint_rows.len();
            joints::prepare(
                *joint,
                &self.bodies[ai].1,
                &self.bodies[bi].1,
                h,
                &mut scratch.joint_rows,
            );
            scratch
                .joint_batches
                .push(([ai, bi], start..scratch.joint_rows.len()));
        }
    }

    fn solve_joints(&mut self) {
        for (indices, rows) in &self.scratch.joint_batches {
            let [(_, a), (_, b)] = self
                .bodies
                .get_disjoint_mut(*indices)
                .expect("distinct joint bodies");
            for row in &mut self.scratch.joint_rows[rows.clone()] {
                row.solve(a, b);
            }
        }
    }

    fn prepare_solver(&mut self, h: f64) {
        let scratch = &mut self.scratch;
        let ratio = self.previous_h.map_or(0.0, |previous| h / previous);
        scratch.rows.clear();
        let mut old = self.cache.iter().peekable();
        for m in &mut scratch.manifolds {
            let [ai, bi] = m.indices;
            let (a, b) = (&self.bodies[ai].1, &self.bodies[bi].1);
            if effective_mass(a) + effective_mass(b) == 0.0 {
                continue;
            }
            while old.peek().is_some_and(|(pair, _)| *pair < m.pair) {
                old.next();
            }
            let cached = match old.peek() {
                Some((pair, range)) if *pair == m.pair => &self.impulses[range.clone()],
                _ => &[],
            };
            let mut used = [false; 8];
            let start = scratch.rows.len();
            for &(_, _, contact) in &self.debug_contacts[m.points.clone()] {
                scratch.rows.push(make_row(
                    a,
                    b,
                    contact,
                    cached,
                    &mut used,
                    ratio,
                    self.settings.bounce_threshold,
                ));
            }
            m.rows = start..scratch.rows.len();
        }
    }

    fn solve_rows(&mut self) {
        let scratch = &mut self.scratch;
        for m in &scratch.manifolds {
            if m.rows.is_empty() {
                continue;
            }
            let [(_, a), (_, b)] = self
                .bodies
                .get_disjoint_mut(m.indices)
                .expect("distinct pairs");
            for row in &scratch.rows[m.rows.clone()] {
                let impulse = row.contact.normal * row.cache.normal_impulse + row.cache.friction;
                apply(a, b, row.offsets, impulse);
            }
        }
        for _ in 0..self.settings.iterations {
            self.solve_joints();
            let scratch = &mut self.scratch;
            for m in &scratch.manifolds {
                if m.rows.is_empty() {
                    continue;
                }
                let [(_, a), (_, b)] = self
                    .bodies
                    .get_disjoint_mut(m.indices)
                    .expect("distinct pairs");
                for row in &mut scratch.rows[m.rows.clone()] {
                    solve(a, b, row);
                }
            }
        }
    }

    fn finish_contacts(&mut self) {
        let scratch = &mut self.scratch;
        scratch.next_cache.clear();
        scratch.next_impulses.clear();
        for m in &scratch.manifolds {
            if m.rows.is_empty() {
                continue;
            }
            let [(_, a), (_, b)] = self
                .bodies
                .get_disjoint_mut(m.indices)
                .expect("distinct pairs");
            let rows = &scratch.rows[m.rows.clone()];
            let deepest = rows
                .iter()
                .max_by(|a, b| a.contact.depth.total_cmp(&b.contact.depth))
                .expect("nonempty manifold")
                .contact;
            let inv = [effective_mass(a), effective_mass(b)];
            let correction = deepest.normal
                * ((deepest.depth - self.settings.contact_slop).max(0.0)
                    * self.settings.position_correction
                    / (inv[0] + inv[1]));
            a.position -= correction * inv[0];
            b.position += correction * inv[1];
            let start = scratch.next_impulses.len();
            scratch
                .next_impulses
                .extend(rows.iter().map(|row| row.cache));
            scratch
                .next_cache
                .push((m.pair, start..scratch.next_impulses.len()));
        }
        std::mem::swap(&mut self.cache, &mut scratch.next_cache);
        std::mem::swap(&mut self.impulses, &mut scratch.next_impulses);
    }

    fn update_sleep(&mut self, h: f64) {
        let islands = &mut self.scratch.islands;
        let threshold = self.settings.sleep_speed * self.settings.sleep_speed;
        for (i, (_, b)) in self.bodies.iter_mut().enumerate() {
            if b.kind != BodyKind::Dynamic || b.asleep {
                continue;
            }
            let root = islands.root(i);
            let island = &mut islands.nodes[root];
            let slow = b.velocity.length_squared() < threshold
                && b.angular_velocity().length_squared() < threshold;
            if slow
                && island.grounded
                && !island.busy
                && b.force == Vec3::ZERO
                && b.torque == Vec3::ZERO
            {
                b.sleep_time += h;
            } else {
                b.sleep_time = 0.0;
            }
            island.min_quiet_time = island.min_quiet_time.min(b.sleep_time);
        }
        for (i, (_, b)) in self.bodies.iter_mut().enumerate() {
            if b.kind != BodyKind::Dynamic || b.asleep {
                continue;
            }
            let root = islands.root(i);
            if islands.nodes[root].min_quiet_time >= self.settings.sleep_time {
                b.asleep = true;
                b.velocity = Vec3::ZERO;
                b.angular_momentum = Vec3::ZERO;
            }
        }
    }

    fn publish_events(&mut self) {
        let mut previous = self.touching.iter().peekable();
        for &(pair, sensor) in &self.scratch.active {
            while let Some(&&((a, b), sensor)) = previous.peek() {
                if (a, b) >= pair {
                    break;
                }
                previous.next();
                self.events.push(CollisionEvent {
                    a,
                    b,
                    sensor,
                    started: false,
                });
            }
            if previous
                .peek()
                .is_some_and(|(old, sensor_was)| *old == pair && *sensor_was == sensor)
            {
                previous.next();
            } else {
                if let Some(&&((a, b), sensor_was)) = previous.peek()
                    && (a, b) == pair
                {
                    previous.next();
                    self.events.push(CollisionEvent {
                        a,
                        b,
                        sensor: sensor_was,
                        started: false,
                    });
                }
                let (a, b) = pair;
                self.events.push(CollisionEvent {
                    a,
                    b,
                    sensor,
                    started: true,
                });
            }
        }
        self.events
            .extend(previous.map(|&((a, b), sensor)| CollisionEvent {
                a,
                b,
                sensor,
                started: false,
            }));
        std::mem::swap(&mut self.touching, &mut self.scratch.active);
    }

    fn collect_candidates(&mut self) {
        self.proxies.clear();
        self.planes.clear();
        self.candidates.clear();
        let mut center_min = Vec3::splat(f64::INFINITY);
        let mut center_max = Vec3::splat(f64::NEG_INFINITY);
        for (index, (_, body)) in self.bodies.iter().enumerate() {
            if let Some(bounds) = Aabb::of(body) {
                self.proxies.push(Proxy { index, bounds });
                center_min = center_min.min(body.position);
                center_max = center_max.max(body.position);
            } else {
                self.planes.push(index);
            }
        }
        let extent = center_max - center_min;
        let axis = if extent.x >= extent.y && extent.x >= extent.z {
            0
        } else if extent.y >= extent.z {
            1
        } else {
            2
        };
        self.proxies.sort_unstable_by(|a, b| {
            a.bounds.min[axis]
                .total_cmp(&b.bounds.min[axis])
                .then(a.index.cmp(&b.index))
        });
        for (i, a) in self.proxies.iter().enumerate() {
            for b in &self.proxies[i + 1..] {
                if b.bounds.min[axis] > a.bounds.max[axis] {
                    break;
                }
                if a.bounds.intersects(b.bounds) {
                    push_pair(&self.bodies, &mut self.candidates, a.index, b.index);
                }
            }
            for &plane in &self.planes {
                let p = &self.bodies[plane].1;
                let normal = p.rotation * Vec3::Y;
                let support = normal.cmpge(Vec3::ZERO);
                let low = Vec3::select(support, a.bounds.min, a.bounds.max);
                if (low - p.position).dot(normal) <= 0.0 {
                    push_pair(&self.bodies, &mut self.candidates, a.index, plane);
                }
            }
        }
        self.candidates
            .sort_unstable_by_key(|&[a, b]| (self.bodies[a].0, self.bodies[b].0));
    }

    fn advance_poses(&mut self, h: f64, stats: &mut StepStats) {
        // Targets are static, so advancing the moving bodies in dense storage
        // order cannot change any cast.
        self.ccd_targets.clear();
        if self.settings.ccd {
            self.ccd_targets.extend(
                self.bodies.iter().enumerate().filter_map(|(i, (_, b))| {
                    (b.kind == BodyKind::Static && !b.sensor).then_some(i)
                }),
            );
        }
        for i in 0..self.bodies.len() {
            let mut remaining = h;
            for _ in 0..4 {
                let Some((distance, radius, speed, normal, j)) = self.first_impact(i, remaining)
                else {
                    self.bodies[i].1.integrate_pose(remaining);
                    break;
                };
                let [(_, body), (_, target)] = self
                    .bodies
                    .get_disjoint_mut([i, j])
                    .expect("distinct CCD bodies");
                let time = distance / speed;
                body.integrate_pose(time);
                let contact = Contact {
                    point: body.position - normal * radius,
                    normal,
                    depth: 0.0,
                };
                let mut row = make_row(
                    target,
                    body,
                    contact,
                    &[],
                    &mut [false; 8],
                    0.0,
                    self.settings.bounce_threshold,
                );
                solve(target, body, &mut row);
                // Separation keeps the next cast from immediately re-hitting
                // the same surface because of roundoff.
                body.position += normal * (radius * 1e-9);
                stats.ccd_impacts += 1;
                remaining -= time;
                if remaining <= 0.0 {
                    break;
                }
            }
        }
    }

    fn first_impact(&self, index: usize, dt: f64) -> Option<(f64, f64, f64, Vec3, usize)> {
        let body = &self.bodies[index].1;
        if self.ccd_targets.is_empty()
            || !body.ccd
            || body.sensor
            || body.asleep
            || body.kind != BodyKind::Dynamic
        {
            return None;
        }
        let Shape::Sphere { radius } = body.shape else {
            return None;
        };
        let direction = body.velocity.try_normalize()?;
        let speed = body.velocity.length();
        let mut hit = None;
        for &j in &self.ccd_targets {
            let (handle, target) = &self.bodies[j];
            if !compatible(body, target) {
                continue;
            }
            if let Some((distance, normal)) =
                sweep_sphere(target, body.position, direction, radius, speed * dt)
                && hit.is_none_or(|(best, id, _, _)| {
                    distance < best || (distance == best && *handle < id)
                })
            {
                hit = Some((distance, *handle, normal, j));
            }
        }
        hit.map(|(distance, _, normal, j)| (distance, radius, speed, normal, j))
    }
}

fn compatible(a: &RigidBody, b: &RigidBody) -> bool {
    a.mask & b.layer != 0 && b.mask & a.layer != 0
}

fn push_pair(bodies: &[(BodyHandle, RigidBody)], pairs: &mut Vec<[usize; 2]>, i: usize, j: usize) {
    let (ah, a) = &bodies[i];
    let (bh, b) = &bodies[j];
    if (a.sensor || b.sensor || a.kind == BodyKind::Dynamic || b.kind == BodyKind::Dynamic)
        && compatible(a, b)
    {
        pairs.push(if ah < bh { [i, j] } else { [j, i] });
    }
}

pub(crate) fn effective_mass(b: &RigidBody) -> f64 {
    if b.asleep { 0.0 } else { b.inverse_mass }
}

fn should_wake(b: &RigidBody) -> bool {
    !b.asleep
        && (b.velocity.length_squared() > 1e-4 || b.angular_velocity().length_squared() > 1e-4)
}

pub(crate) fn inverse_effective_mass(
    a: &RigidBody,
    b: &RigidBody,
    offsets: [Vec3; 2],
    direction: Vec3,
) -> f64 {
    [(a, offsets[0]), (b, offsets[1])]
        .into_iter()
        .map(|(body, r)| {
            if effective_mass(body) == 0.0 {
                return 0.0;
            }
            let axis = r.cross(direction);
            body.inverse_mass + axis.dot(body.apply_inverse_inertia(axis))
        })
        .sum()
}

fn apply(a: &mut RigidBody, b: &mut RigidBody, r: [Vec3; 2], impulse: Vec3) {
    if impulse != Vec3::ZERO {
        a.solver_impulse(-impulse, r[0]);
        b.solver_impulse(impulse, r[1]);
    }
}

pub(crate) fn relative_velocity(a: &RigidBody, b: &RigidBody, r: [Vec3; 2]) -> Vec3 {
    b.point_velocity(r[1]) - a.point_velocity(r[0])
}

fn make_row(
    a: &RigidBody,
    b: &RigidBody,
    contact: Contact,
    old: &[Cached],
    used: &mut [bool; 8],
    ratio: f64,
    bounce_threshold: f64,
) -> Row {
    let offsets = [contact.point - a.position, contact.point - b.position];
    let local = [
        a.rotation.conjugate() * offsets[0],
        b.rotation.conjugate() * offsets[1],
    ];
    let n = contact.normal;
    let approach = relative_velocity(a, b, offsets).dot(n);
    let bounce = a.material.restitution.max(b.material.restitution);
    let target_speed = if approach < -bounce_threshold {
        -approach * bounce
    } else {
        0.0
    };
    let mut cache = Cached {
        local,
        normal: n,
        normal_impulse: 0.0,
        friction: Vec3::ZERO,
    };
    if (0.5..=2.0).contains(&ratio) {
        let threshold = 0.05;
        let closest = old
            .iter()
            .enumerate()
            .filter_map(|(i, c)| {
                let error =
                    local[0].distance_squared(c.local[0]) + local[1].distance_squared(c.local[1]);
                (!used[i] && c.normal.dot(n) > 0.98 && error < threshold * threshold)
                    .then_some((i, c, error))
            })
            .min_by(|a, b| a.2.total_cmp(&b.2));
        if let Some((i, c, _)) = closest {
            used[i] = true;
            cache.normal_impulse = c.normal_impulse * ratio;
            cache.friction = (c.friction - n * c.friction.dot(n)) * ratio;
        }
    }
    Row {
        contact,
        offsets,
        normal_mass: inverse_effective_mass(a, b, offsets, n).recip(),
        target_speed,
        cache,
    }
}

fn solve(a: &mut RigidBody, b: &mut RigidBody, row: &mut Row) {
    let n = row.contact.normal;
    let r = row.offsets;
    let speed = relative_velocity(a, b, r).dot(n);
    let old = row.cache.normal_impulse;
    let next = (old + (row.target_speed - speed) * row.normal_mass).max(0.0);
    row.cache.normal_impulse = next;
    apply(a, b, r, n * (next - old));

    let velocity = relative_velocity(a, b, r);
    let slip = velocity - n * velocity.dot(n);
    let old = row.cache.friction;
    let mut candidate = old;
    if let Some(direction) = slip.try_normalize() {
        candidate -= slip / inverse_effective_mass(a, b, r, direction);
    }
    let static_limit = a.material.static_friction.sqrt() * b.material.static_friction.sqrt() * next;
    if candidate.length_squared() > static_limit * static_limit {
        let dynamic_limit =
            a.material.dynamic_friction.sqrt() * b.material.dynamic_friction.sqrt() * next;
        candidate = candidate.clamp_length_max(dynamic_limit);
    }
    row.cache.friction = candidate;
    apply(a, b, r, candidate - old);
}
