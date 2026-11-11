//! Deterministic rigid-body stepping, contacts, collision events, and spatial queries.
use std::collections::BTreeMap;
use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::collision::{Contact, contacts};
use crate::dynamic_tree::DynamicTree;
use crate::grid::ProxyGrid;
use crate::joints::{self, Joint, JointHandle, JointImpulse, JointRow};
use crate::spatial::{
    Aabb, Bvh, Proxy, SpatialIndex, cast_convex_body, project, ray_body, sphere_hit, sweep_sphere,
};
use crate::{BodyKind, RigidBody, RigidBodyBuilder, Shape, Vec3};

static NEXT_WORLD: AtomicU64 = AtomicU64::new(1);

/// Stable identifier scoped to the world that issued it. Removed IDs are never reused.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BodyHandle {
    world: u64,
    serial: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BroadPhase {
    #[default]
    SweepAndPrune,
    Bvh,
    UniformGrid,
    DynamicTree,
}

#[derive(Clone, Debug)]
pub struct Settings {
    pub gravity: Vec3,
    pub broad_phase: BroadPhase,
    pub grid_cell_size: f64,
    /// Uses Rayon for independent narrow-phase contact generation when available.
    pub parallel_narrow_phase: bool,
    pub fixed_dt: f64,
    pub substeps: usize,
    pub iterations: usize,
    /// Translation-only stabilization passes for coupled contact manifolds.
    pub position_iterations: usize,
    pub contact_slop: f64,
    pub position_correction: f64,
    pub bounce_threshold: f64,
    pub sleep_speed: f64,
    pub sleep_time: f64,
    pub ccd: bool,
    /// Enables synchronized swept collisions involving translating spheres.
    pub ccd_moving_spheres: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            gravity: Vec3::new(0.0, -9.81, 0.0),
            broad_phase: BroadPhase::SweepAndPrune,
            grid_cell_size: 1.0,
            parallel_narrow_phase: false,
            fixed_dt: 1.0 / 60.0,
            substeps: 2,
            iterations: 12,
            position_iterations: 3,
            contact_slop: 0.001,
            position_correction: 0.65,
            bounce_threshold: 0.5,
            sleep_speed: 0.035,
            sleep_time: 0.6,
            ccd: true,
            ccd_moving_spheres: false,
        }
    }
}

impl Settings {
    fn valid(&self) -> bool {
        (!self.parallel_narrow_phase || cfg!(feature = "rayon"))
            && self.gravity.is_finite()
            && self.grid_cell_size.is_finite()
            && self.grid_cell_size > 0.0
            && self.fixed_dt.is_finite()
            && self.fixed_dt > 0.0
            && self.substeps > 0
            && self.iterations > 0
            && self.position_iterations > 0
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
    /// Swept passes that consumed their event budget and finished discretely.
    pub ccd_budget_fallbacks: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CollisionEvent {
    pub a: BodyHandle,
    pub b: BodyHandle,
    pub sensor: bool,
    pub started: bool,
}

/// A swept impact observed during the last public step.
#[derive(Clone, Copy, Debug)]
pub struct CcdImpact {
    pub moving: BodyHandle,
    pub target: BodyHandle,
    pub point: Vec3,
    /// World normal from the target toward the moving sphere.
    pub normal: Vec3,
    /// Seconds since the start of the public step.
    pub time: f64,
}

struct ImpactCandidate {
    time: f64,
    moving: usize,
    target: usize,
    radius: f64,
    normal: Vec3,
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

#[cfg(feature = "rayon")]
#[derive(Clone, Default)]
struct GeometryPatch {
    contacts: [Contact; 8],
    len: usize,
}

struct JointBatch {
    handle: JointHandle,
    indices: [usize; 2],
    rows: Range<usize>,
}

#[derive(Default)]
struct SolverScratch {
    manifolds: Vec<Manifold>,
    rows: Vec<Row>,
    joint_rows: Vec<JointRow>,
    joint_batches: Vec<JointBatch>,
    next_cache: Vec<(Pair, Range<usize>)>,
    next_impulses: Vec<Cached>,
    active: Vec<(Pair, bool)>,
    islands: Islands,
    solver_degree: Vec<u8>,
    coupled_manifolds: Vec<usize>,
    #[cfg(feature = "rayon")]
    parallel_geometry: Vec<GeometryPatch>,
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
    bvh: Bvh,
    proxy_grid: ProxyGrid,
    dynamic_tree: DynamicTree,
    planes: Vec<usize>,
    candidates: Vec<[usize; 2]>,
    ccd_targets: Vec<usize>,
    impacts: Vec<CcdImpact>,
    joints: BTreeMap<JointHandle, Joint>,
    joint_cache: BTreeMap<JointHandle, Vec<JointImpulse>>,
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
            bvh: Bvh::default(),
            proxy_grid: ProxyGrid::default(),
            dynamic_tree: DynamicTree::default(),
            planes: Vec::new(),
            candidates: Vec::new(),
            ccd_targets: Vec::new(),
            impacts: Vec::new(),
            joints: BTreeMap::new(),
            joint_cache: BTreeMap::new(),
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

    /// Moves the coordinate origin by an offset without changing relative
    /// body poses, velocities, constraints, or local contact anchors.
    pub fn rebase_origin(&mut self, offset: Vec3) -> bool {
        if !offset.is_finite()
            || self
                .bodies
                .iter()
                .any(|(_, body)| !(body.position - offset).is_finite())
            || self
                .debug_contacts
                .iter()
                .any(|(_, _, c)| !(c.point - offset).is_finite())
            || self
                .impacts
                .iter()
                .any(|hit| !(hit.point - offset).is_finite())
        {
            return false;
        }
        for (_, body) in &mut self.bodies {
            body.position -= offset;
        }
        for (_, _, contact) in &mut self.debug_contacts {
            contact.point -= offset;
        }
        for impact in &mut self.impacts {
            impact.point -= offset;
        }
        true
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

    pub fn joints(&self) -> impl ExactSizeIterator<Item = (JointHandle, &Joint)> {
        self.joints.iter().map(|(&handle, joint)| (handle, joint))
    }

    pub fn remove_joint(&mut self, handle: JointHandle) -> Option<Joint> {
        let joint = self.joints.remove(&handle)?;
        self.joint_cache.remove(&handle);
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

    /// Replaces an unrestricted dynamic compound with its component bodies.
    /// Child velocities inherit the parent velocity field, preserving total
    /// linear and angular momentum. Linked joints are removed with the parent.
    /// On invalid input, the world is unchanged.
    pub fn fracture_compound(&mut self, handle: BodyHandle) -> Option<Vec<BodyHandle>> {
        let parent = self.body(handle)?;
        let Shape::Compound(compound) = parent.shape() else {
            return None;
        };
        if parent.kind != BodyKind::Dynamic
            || parent.linear_axes != Vec3::ONE
            || parent.angular_axes != Vec3::ONE
            || parent.custom_inertia
        {
            return None;
        }
        let total: f64 = compound.parts().iter().map(|part| part.weight).sum();
        let spin = parent.angular_velocity();
        let fragments: Vec<_> = compound
            .parts()
            .iter()
            .map(|part| {
                let offset = parent.rotation * part.position;
                RigidBodyBuilder::dynamic()
                    .shape(part.shape.clone())
                    .position(parent.position + offset)
                    .rotation(parent.rotation * part.rotation)
                    .velocity(parent.velocity + spin.cross(offset))
                    .angular_velocity(spin)
                    .mass((part.weight / total) / parent.inverse_mass)
                    .material(parent.material)
                    .gravity_scale(parent.gravity_scale)
                    .damping(parent.damping[0], parent.damping[1])
                    .filter(parent.layer, parent.mask)
                    .sensor(parent.sensor)
                    .ccd(parent.ccd)
                    .try_build()
                    .ok()
            })
            .collect::<Option<_>>()?;
        self.remove_body(handle)?;
        Some(
            fragments
                .into_iter()
                .map(|body| self.add_body(body))
                .collect(),
        )
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

    /// Accumulates a finite force at a world-space point on a live dynamic body.
    /// This path preserves cached contacts because the collider pose stays current.
    pub fn add_force_at_point(&mut self, handle: BodyHandle, force: Vec3, point: Vec3) -> bool {
        let Some(&index) = self.indices.get(&handle) else {
            return false;
        };
        self.bodies[index].1.add_force_at_point(force, point)
    }

    /// Applies an instantaneous impulse at a world-space point on a live dynamic body.
    pub fn apply_impulse(&mut self, handle: BodyHandle, impulse: Vec3, point: Vec3) -> bool {
        let Some(&index) = self.indices.get(&handle) else {
            return false;
        };
        self.bodies[index].1.apply_impulse(impulse, point)
    }

    fn invalidate_cache(&mut self, handle: BodyHandle) {
        self.cache
            .retain(|((a, b), _)| *a != handle && *b != handle);
        self.joint_cache
            .retain(|id, _| !self.joints[id].bodies.contains(&handle));
    }

    pub fn bodies(&self) -> impl ExactSizeIterator<Item = (BodyHandle, &RigidBody)> {
        self.bodies.iter().map(|(id, body)| (*id, body))
    }

    /// Transitions observed during the last successful step, in substep order.
    pub fn events(&self) -> &[CollisionEvent] {
        &self.events
    }

    /// Swept impacts sorted by time within the last successful step.
    pub fn ccd_impacts(&self) -> &[CcdImpact] {
        &self.impacts
    }

    /// World-space contacts sampled before the final substep's penetration correction.
    pub fn debug_contacts(&self) -> &[(BodyHandle, BodyHandle, Contact)] {
        &self.debug_contacts
    }

    pub fn spatial_index(&self) -> SpatialIndex<'_> {
        SpatialIndex::new(self)
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
            .filter_map(|(body, b)| sphere_hit(body, b, origin, direction, radius, max_distance))
            .min_by(|a, b| a.distance.total_cmp(&b.distance).then(a.body.cmp(&b.body)))
    }

    /// Translates a finite convex collider from its current pose against
    /// other convex colliders, compounds, planes, and static meshes. Moving
    /// spheres use the specialized sphere-cast paths.
    pub fn cast_shape(
        &self,
        moving: &RigidBody,
        direction: Vec3,
        max_distance: f64,
        filter: QueryFilter,
    ) -> Option<RayHit> {
        let direction = direction.try_normalize()?;
        if max_distance.is_nan() || max_distance < 0.0 {
            return None;
        }
        if let Shape::Sphere { radius } = *moving.shape() {
            return self.cast_sphere(moving.position(), radius, direction, max_distance, filter);
        }
        self.bodies()
            .filter(|&(id, b)| filter.accepts(id, b))
            .filter_map(|(body, target)| {
                let (distance, normal, position) =
                    cast_convex_body(moving, target, direction, max_distance)?;
                Some(RayHit {
                    body,
                    position,
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
        self.impacts.clear();
        for substep in 0..self.settings.substeps {
            self.substep(h, h * substep as f64, &mut stats);
        }
        self.impacts.sort_unstable_by(|a, b| {
            a.time
                .total_cmp(&b.time)
                .then(a.moving.cmp(&b.moving))
                .then(a.target.cmp(&b.target))
        });
        stats.ccd_impacts = self.impacts.len();
        for (_, body) in &mut self.bodies {
            body.force = Vec3::ZERO;
            body.torque = Vec3::ZERO;
            stats.sleeping_bodies += usize::from(body.asleep);
        }
        Ok(stats)
    }

    fn substep(&mut self, h: f64, start: f64, stats: &mut StepStats) {
        for (_, body) in &mut self.bodies {
            if body.kind == BodyKind::Dynamic && !body.asleep {
                body.velocity += (self.settings.gravity * body.gravity_scale * body.linear_axes
                    + body.apply_inverse_mass(body.force))
                    * h;
                body.angular_momentum += body.torque * body.angular_axes * h;
                body.velocity *= (-body.damping[0] * h).exp();
                body.angular_momentum *= (-body.damping[1] * h).exp();
            }
        }
        self.prepare_joints(h);
        self.warm_start_joints(h);
        for _ in 0..self.settings.iterations {
            self.solve_joints();
        }
        stats.ccd_budget_fallbacks += self.advance_poses(h, start);
        self.collect_candidates();
        stats.candidate_pairs += self.candidates.len();

        self.build_manifolds(stats);
        if self.settings.sleep_time > 0.0 {
            self.prepare_islands();
        }
        self.prepare_solver(h);
        self.solve_rows();
        self.cache_joint_impulses();
        self.finish_contacts();
        if self.settings.sleep_time > 0.0 {
            self.update_sleep(h);
        }
        self.publish_events();
        self.previous_h = Some(h);
    }

    fn build_manifolds(&mut self, stats: &mut StepStats) {
        self.scratch.active.clear();
        self.scratch.manifolds.clear();
        self.debug_contacts.clear();

        #[cfg(feature = "rayon")]
        if self.settings.parallel_narrow_phase && self.candidates.len() >= 128 {
            self.build_parallel_manifolds(stats);
            return;
        }

        let mut geometry = Vec::with_capacity(8);
        for i in 0..self.candidates.len() {
            let indices @ [ai, bi] = self.candidates[i];
            geometry.clear();
            contacts(&self.bodies[ai].1, &self.bodies[bi].1, &mut geometry);
            self.record_manifold(indices, &geometry, stats);
        }
    }

    #[cfg(feature = "rayon")]
    fn build_parallel_manifolds(&mut self, stats: &mut StepStats) {
        use rayon::prelude::*;
        let mut patches = std::mem::take(&mut self.scratch.parallel_geometry);
        patches.resize(self.candidates.len(), GeometryPatch::default());
        patches.par_iter_mut().zip(&self.candidates).for_each_init(
            || Vec::with_capacity(8),
            |scratch, (patch, &[ai, bi])| {
                scratch.clear();
                contacts(&self.bodies[ai].1, &self.bodies[bi].1, scratch);
                patch.len = scratch.len();
                patch.contacts[..patch.len].copy_from_slice(scratch);
            },
        );
        // Preserve the same candidate and manifold order as the serial path.
        for (i, patch) in patches.iter().enumerate() {
            self.record_manifold(self.candidates[i], &patch.contacts[..patch.len], stats);
        }
        self.scratch.parallel_geometry = patches;
    }

    fn record_manifold(
        &mut self,
        indices: [usize; 2],
        geometry: &[Contact],
        stats: &mut StepStats,
    ) {
        if geometry.is_empty() {
            return;
        }
        let [ai, bi] = indices;
        let (ah, a) = &self.bodies[ai];
        let (bh, b) = &self.bodies[bi];
        let pair = (*ah, *bh);
        let sensor = a.sensor || b.sensor;
        self.scratch.active.push((pair, sensor));
        stats.contact_points += geometry.len();
        let start = self.debug_contacts.len();
        self.debug_contacts
            .extend(geometry.iter().map(|&c| (pair.0, pair.1, c)));
        if !sensor {
            self.scratch.manifolds.push(Manifold {
                pair,
                indices,
                points: start..self.debug_contacts.len(),
                rows: 0..0,
            });
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
        for (&handle, joint) in &self.joints {
            let [ai, bi] = joint.bodies.map(|id| self.indices[&id]);
            let start = scratch.joint_rows.len();
            joints::prepare(
                *joint,
                &self.bodies[ai].1,
                &self.bodies[bi].1,
                h,
                &mut scratch.joint_rows,
            );
            scratch.joint_batches.push(JointBatch {
                handle,
                indices: [ai, bi],
                rows: start..scratch.joint_rows.len(),
            });
        }
    }

    fn warm_start_joints(&mut self, h: f64) {
        let ratio = self.previous_h.map_or(0.0, |previous| h / previous);
        if !(0.5..=2.0).contains(&ratio) {
            return;
        }
        for batch in &self.scratch.joint_batches {
            let Some(cache) = self.joint_cache.get(&batch.handle) else {
                continue;
            };
            if cache.len() != batch.rows.len() {
                continue;
            }
            let [(_, a), (_, b)] = self
                .bodies
                .get_disjoint_mut(batch.indices)
                .expect("distinct joint bodies");
            for (row, &previous) in self.scratch.joint_rows[batch.rows.clone()]
                .iter_mut()
                .zip(cache)
            {
                row.warm_start(previous, ratio, a, b);
            }
        }
    }

    fn solve_joints(&mut self) {
        for batch in &self.scratch.joint_batches {
            let [(_, a), (_, b)] = self
                .bodies
                .get_disjoint_mut(batch.indices)
                .expect("distinct joint bodies");
            for row in &mut self.scratch.joint_rows[batch.rows.clone()] {
                row.solve(a, b);
            }
        }
    }

    fn cache_joint_impulses(&mut self) {
        for batch in &self.scratch.joint_batches {
            let cache = self.joint_cache.entry(batch.handle).or_default();
            cache.clear();
            cache.extend(
                self.scratch.joint_rows[batch.rows.clone()]
                    .iter()
                    .map(JointRow::cache),
            );
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
                scratch.rows.extend(make_row(
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
        scratch.solver_degree.clear();
        scratch.solver_degree.resize(self.bodies.len(), 0);
        scratch.coupled_manifolds.clear();

        // Independent manifolds can run all iterations while their body
        // state and rows are still in cache. Shared static and kinematic
        // endpoints carry no solver state and do not connect islands.
        for m in &scratch.manifolds {
            if m.rows.is_empty() {
                continue;
            }
            for i in m.indices {
                scratch.solver_degree[i] = scratch.solver_degree[i].saturating_add(1);
            }
        }
        for batch in &scratch.joint_batches {
            for i in batch.indices {
                scratch.solver_degree[i] = scratch.solver_degree[i].saturating_add(1);
            }
        }
        for (index, m) in scratch.manifolds.iter().enumerate() {
            if m.rows.is_empty() {
                continue;
            }
            let [(_, a), (_, b)] = self
                .bodies
                .get_disjoint_mut(m.indices)
                .expect("distinct pairs");
            let rows = &mut scratch.rows[m.rows.clone()];
            for row in &*rows {
                let impulse = row.contact.normal * row.cache.normal_impulse + row.cache.friction;
                apply(a, b, row.offsets, impulse);
            }
            let coupled = m
                .indices
                .into_iter()
                .zip([&*a, &*b])
                .any(|(i, body)| effective_mass(body) > 0.0 && scratch.solver_degree[i] > 1);
            if coupled {
                scratch.coupled_manifolds.push(index);
            } else {
                for _ in 0..self.settings.iterations {
                    let mut changed = false;
                    for row in &mut *rows {
                        changed |= solve(a, b, row);
                    }
                    if !changed {
                        break;
                    }
                }
            }
        }
        for _ in 0..self.settings.iterations {
            self.solve_joints();
            let scratch = &mut self.scratch;
            for &index in &scratch.coupled_manifolds {
                let m = &scratch.manifolds[index];
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
            correct_pair(a, b, rows, &self.settings);
            let start = scratch.next_impulses.len();
            scratch
                .next_impulses
                .extend(rows.iter().map(|row| row.cache));
            scratch
                .next_cache
                .push((m.pair, start..scratch.next_impulses.len()));
        }
        for _ in 1..self.settings.position_iterations {
            for &index in &scratch.coupled_manifolds {
                let manifold = &scratch.manifolds[index];
                let [(_, a), (_, b)] = self
                    .bodies
                    .get_disjoint_mut(manifold.indices)
                    .expect("distinct pairs");
                correct_pair(a, b, &scratch.rows[manifold.rows.clone()], &self.settings);
            }
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
        match self.settings.broad_phase {
            BroadPhase::SweepAndPrune => {
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
                }
            }
            BroadPhase::UniformGrid => {
                self.proxy_grid
                    .pairs(self.settings.grid_cell_size, &self.proxies, |a, b| {
                        push_pair(&self.bodies, &mut self.candidates, a, b);
                    })
            }
            BroadPhase::DynamicTree => {
                self.dynamic_tree
                    .pairs(self.bodies.len(), &self.proxies, |a, b| {
                        push_pair(&self.bodies, &mut self.candidates, a, b);
                    });
            }
            BroadPhase::Bvh => {
                self.bvh.rebuild(&mut self.proxies);
                self.bvh
                    .pairs(|a, b| push_pair(&self.bodies, &mut self.candidates, a, b));
            }
        }
        for a in &self.proxies {
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

    fn advance_poses(&mut self, h: f64, start: f64) -> usize {
        self.ccd_targets.clear();
        let sweep = self.settings.ccd && self.bodies.iter().any(|(_, b)| sweeps(b));
        if !sweep {
            for (_, body) in &mut self.bodies {
                body.integrate_pose(h);
            }
            return 0;
        }
        let synchronized = self.settings.ccd_moving_spheres;
        self.ccd_targets
            .extend(self.bodies.iter().enumerate().filter_map(|(i, (_, b))| {
                (!b.sensor
                    && (b.kind == BodyKind::Static
                        || (synchronized && matches!(b.shape, Shape::Sphere { .. }))))
                .then_some(i)
            }));
        if synchronized {
            return self.advance_synchronized(h, start);
        }

        // Static targets make each dynamic sphere's cast independent.
        let mut fallbacks = 0;
        for i in 0..self.bodies.len() {
            let mut remaining = h;
            for _ in 0..4 {
                let Some(hit) = self.first_impact(i, remaining, false) else {
                    self.bodies[i].1.integrate_pose(remaining);
                    remaining = 0.0;
                    break;
                };
                self.bodies[i].1.integrate_pose(hit.time);
                remaining = (remaining - hit.time).max(0.0);
                self.resolve_impact(hit, start + (h - remaining));
                if remaining == 0.0 {
                    break;
                }
            }
            if remaining > 0.0 {
                self.bodies[i].1.integrate_pose(remaining);
                fallbacks += 1;
            }
        }
        fallbacks
    }

    fn advance_synchronized(&mut self, h: f64, start: f64) -> usize {
        let mut remaining = h;
        for _ in 0..32 {
            let next = (0..self.bodies.len())
                .filter_map(|i| self.first_impact(i, remaining, true))
                .min_by(|a, b| {
                    a.time
                        .total_cmp(&b.time)
                        .then(self.bodies[a.moving].0.cmp(&self.bodies[b.moving].0))
                        .then(self.bodies[a.target].0.cmp(&self.bodies[b.target].0))
                });
            let dt = next.as_ref().map_or(remaining, |hit| hit.time);
            for (_, body) in &mut self.bodies {
                body.integrate_pose(dt);
            }
            remaining = (remaining - dt).max(0.0);
            if let Some(hit) = next {
                self.resolve_impact(hit, start + (h - remaining));
            }
            if remaining == 0.0 {
                return 0;
            }
        }
        // Closely spaced impacts can consume the entire event budget near
        // the same time. Integrate the remainder before the discrete solve
        // so force integration and elapsed simulation time stay consistent.
        for (_, body) in &mut self.bodies {
            body.integrate_pose(remaining);
        }
        1
    }

    fn first_impact(&self, index: usize, dt: f64, synchronized: bool) -> Option<ImpactCandidate> {
        let (handle, body) = &self.bodies[index];
        if !sweeps(body) {
            return None;
        }
        let Shape::Sphere { radius } = body.shape else {
            unreachable!("validated moving sphere");
        };
        let mut best: Option<ImpactCandidate> = None;
        for &j in &self.ccd_targets {
            let (other_handle, target) = &self.bodies[j];
            if index == j
                || !compatible(body, target)
                || (synchronized && sweeps(target) && handle > other_handle)
            {
                continue;
            }
            let relative = body.velocity - target.velocity;
            let Some(direction) = relative.try_normalize() else {
                continue;
            };
            let speed = relative.length();
            let travel = speed * dt;
            if travel <= radius * 0.1 {
                continue;
            }
            if let Some((distance, normal)) =
                sweep_sphere(target, body.position, direction, radius, travel)
            {
                if normal.dot(body.apply_inverse_mass(normal) + target.apply_inverse_mass(normal))
                    <= 0.0
                {
                    continue;
                }
                let time = distance / speed;
                if best.as_ref().is_none_or(|previous| {
                    time < previous.time
                        || (time == previous.time && *other_handle < self.bodies[previous.target].0)
                }) {
                    best = Some(ImpactCandidate {
                        time,
                        moving: index,
                        target: j,
                        radius,
                        normal,
                    });
                }
            }
        }
        best
    }

    fn resolve_impact(&mut self, hit: ImpactCandidate, time: f64) {
        let [(moving, a), (target, b)] = self
            .bodies
            .get_disjoint_mut([hit.moving, hit.target])
            .expect("distinct CCD bodies");
        if b.asleep {
            b.wake();
        }
        let normal = hit.normal;
        let contact = Contact {
            point: a.position - normal * hit.radius,
            normal,
            depth: 0.0,
        };
        if let Some(mut row) = make_row(
            b,
            a,
            contact,
            &[],
            &mut [false; 8],
            0.0,
            self.settings.bounce_threshold,
        ) {
            solve(b, a, &mut row);
        }
        // Separate along the response direction to avoid repeated zero-time
        // hits without violating either body's locked axes.
        separate(b, a, normal, hit.radius * 1e-9);
        self.impacts.push(CcdImpact {
            moving: *moving,
            target: *target,
            point: contact.point,
            normal,
            time,
        });
    }
}

fn sweeps(body: &RigidBody) -> bool {
    body.ccd
        && !body.sensor
        && !body.asleep
        && body.kind == BodyKind::Dynamic
        && matches!(body.shape, Shape::Sphere { .. })
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
            direction.dot(body.apply_inverse_mass(direction))
                + axis.dot(body.apply_inverse_inertia(axis))
        })
        .sum()
}

fn correct_pair(a: &mut RigidBody, b: &mut RigidBody, rows: &[Row], settings: &Settings) {
    let deepest = rows
        .iter()
        .max_by(|a, b| a.contact.depth.total_cmp(&b.contact.depth))
        .expect("nonempty manifold");
    let normal = deepest.contact.normal;
    let initial = deepest.offsets[0] - deepest.offsets[1];
    let correction = ((b.position - a.position) - initial).dot(normal);
    let error = deepest.contact.depth - settings.contact_slop - correction;
    separate(a, b, normal, error.max(0.0) * settings.position_correction);
}

fn separate(a: &mut RigidBody, b: &mut RigidBody, normal: Vec3, amount: f64) {
    if amount == 0.0 {
        return;
    }
    let ra = if a.asleep {
        Vec3::ZERO
    } else {
        a.apply_inverse_mass(normal)
    };
    let rb = if b.asleep {
        Vec3::ZERO
    } else {
        b.apply_inverse_mass(normal)
    };
    let total = normal.dot(ra + rb);
    if total > 0.0 {
        let impulse = amount / total;
        a.position -= ra * impulse;
        b.position += rb * impulse;
    }
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
) -> Option<Row> {
    let offsets = [contact.point - a.position, contact.point - b.position];
    let local = [
        a.rotation.conjugate() * offsets[0],
        b.rotation.conjugate() * offsets[1],
    ];
    let n = contact.normal;
    let response = inverse_effective_mass(a, b, offsets, n);
    if response <= 0.0 {
        return None;
    }
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
    Some(Row {
        contact,
        offsets,
        normal_mass: response.recip(),
        target_speed,
        cache,
    })
}

fn solve(a: &mut RigidBody, b: &mut RigidBody, row: &mut Row) -> bool {
    let n = row.contact.normal;
    let r = row.offsets;
    let mut velocity = relative_velocity(a, b, r);
    let speed = velocity.dot(n);
    let old = row.cache.normal_impulse;
    let next = (old + (row.target_speed - speed) * row.normal_mass).max(0.0);
    row.cache.normal_impulse = next;
    let normal_changed = old != next;
    if normal_changed {
        apply(a, b, r, n * (next - old));
        velocity = relative_velocity(a, b, r);
    }

    let slip = velocity - n * velocity.dot(n);
    let old = row.cache.friction;
    if old == Vec3::ZERO && slip == Vec3::ZERO {
        return normal_changed;
    }
    let mut candidate = old;
    if let Some(direction) = slip.try_normalize() {
        let response = inverse_effective_mass(a, b, r, direction);
        if response > 0.0 {
            candidate -= slip / response;
        }
    }
    let static_limit = a.material.static_friction.sqrt() * b.material.static_friction.sqrt() * next;
    if candidate.length_squared() > static_limit * static_limit {
        let dynamic_limit =
            a.material.dynamic_friction.sqrt() * b.material.dynamic_friction.sqrt() * next;
        candidate = candidate.clamp_length_max(dynamic_limit);
    }
    row.cache.friction = candidate;
    apply(a, b, r, candidate - old);
    normal_changed || candidate != old
}
