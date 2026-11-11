//! Deterministic rigid-body stepping, contacts, collision events, and spatial queries.
use std::collections::BTreeMap;
use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use crate::collision::{Contact, contacts};
use crate::dynamic_tree::DynamicTree;
use crate::grid::ProxyGrid;
use crate::joints::{self, Joint, JointHandle, JointImpulse, JointRow};
use crate::spatial::{
    Aabb, Bvh, Proxy, SpatialIndex, cast_convex_body, project, ray_body, sphere_hit, sweep_sphere,
};
use crate::{BodyKind, RigidBody, RigidBodyBuilder, Shape, Vec3};

#[cfg(feature = "rayon")]
mod parallel;

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
    /// Solves sufficiently large disjoint contact islands with Rayon.
    pub parallel_islands: bool,
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
    /// Enables experimental translating and rotating convex CCD against static targets.
    pub ccd_convex: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            gravity: Vec3::new(0.0, -9.81, 0.0),
            broad_phase: BroadPhase::SweepAndPrune,
            grid_cell_size: 1.0,
            parallel_narrow_phase: false,
            parallel_islands: false,
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
            ccd_convex: false,
        }
    }
}

impl Settings {
    fn valid(&self) -> bool {
        (!(self.parallel_narrow_phase || self.parallel_islands) || cfg!(feature = "rayon"))
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
    NumericalFailure(NumericalStage),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NumericalStage {
    Forces,
    Poses,
    Contacts,
    Solve,
}

impl std::fmt::Display for StepError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::InvalidTimestep => "timestep must be finite and positive after subdivision",
            Self::InvalidSettings => "invalid world settings",
            Self::NumericalFailure(_) => "physics step produced invalid numerical state",
        })
    }
}
impl std::error::Error for StepError {}

#[derive(Clone, Copy, Debug, Default)]
pub struct StepStats {
    /// Candidate pairs examined by the narrow phase across all substeps.
    pub candidate_pairs: usize,
    pub contact_points: usize,
    pub reused_sleeping_manifolds: usize,
    pub sleeping_bodies: usize,
    pub solved_parallel_islands: usize,
    pub broken_joints: usize,
    pub ccd_impacts: usize,
    /// Swept passes that consumed their event budget and finished discretely.
    pub ccd_budget_fallbacks: usize,
}

/// Wall-clock timings aggregated across the substeps of one public step.
#[derive(Clone, Copy, Debug, Default)]
pub struct StepTimings {
    pub forces_and_joints: Duration,
    pub pose_and_ccd: Duration,
    pub broad_phase: Duration,
    pub narrow_phase: Duration,
    pub solver: Duration,
    pub sleep_and_events: Duration,
    pub total: Duration,
}

#[derive(Clone, Copy, Debug)]
pub struct ProfiledStep {
    pub stats: StepStats,
    pub timings: StepTimings,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CollisionEvent {
    pub a: BodyHandle,
    pub b: BodyHandle,
    pub sensor: bool,
    pub started: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct JointBreakEvent {
    pub joint: JointHandle,
    pub bodies: [BodyHandle; 2],
    pub force: f64,
    pub torque: f64,
}

/// A swept impact observed during the last public step.
#[derive(Clone, Copy, Debug)]
pub struct CcdImpact {
    pub moving: BodyHandle,
    pub target: BodyHandle,
    pub point: Vec3,
    /// World normal from the target toward the moving body.
    pub normal: Vec3,
    /// Seconds since the start of the public step.
    pub time: f64,
}

struct ImpactCandidate {
    time: f64,
    moving: usize,
    target: usize,
    contact: ImpactGeometry,
}

enum ImpactGeometry {
    Sphere {
        radius: f64,
        normal: Vec3,
    },
    Convex {
        point: Vec3,
        normal: Vec3,
        separation: f64,
    },
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
        b.enabled
            && b.layer & self.layers != 0
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

#[cfg_attr(feature = "rayon", derive(Clone))]
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
    geometry: Vec<Contact>,
    previous_geometry: Vec<(BodyHandle, BodyHandle, Contact)>,
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

/// In-memory checkpoint of a world's simulation and warm-start state.
/// Geometry assets are shared through their immutable reference counts.
#[derive(Clone)]
pub struct WorldSnapshot {
    id: u64,
    next: u64,
    settings: Settings,
    bodies: Vec<(BodyHandle, RigidBody)>,
    cache: Vec<(Pair, Range<usize>)>,
    impulses: Vec<Cached>,
    previous_h: Option<f64>,
    touching: Vec<(Pair, bool)>,
    events: Vec<CollisionEvent>,
    debug_contacts: Vec<(BodyHandle, BodyHandle, Contact)>,
    impacts: Vec<CcdImpact>,
    joints: BTreeMap<JointHandle, Joint>,
    joint_cache: BTreeMap<JointHandle, Vec<JointImpulse>>,
    joint_breaks: Vec<JointBreakEvent>,
}

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
    joint_breaks: Vec<JointBreakEvent>,
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
            joint_breaks: Vec::new(),
        }
    }

    /// Captures bodies, queued forces, joints, events, and solver caches.
    pub fn snapshot(&self) -> WorldSnapshot {
        WorldSnapshot {
            id: self.id,
            next: self.next,
            settings: self.settings.clone(),
            bodies: self.bodies.clone(),
            cache: self.cache.clone(),
            impulses: self.impulses.clone(),
            previous_h: self.previous_h,
            touching: self.touching.clone(),
            events: self.events.clone(),
            debug_contacts: self.debug_contacts.clone(),
            impacts: self.impacts.clone(),
            joints: self.joints.clone(),
            joint_cache: self.joint_cache.clone(),
            joint_breaks: self.joint_breaks.clone(),
        }
    }

    /// Restores a checkpoint created by this world. Identifiers allocated
    /// after the snapshot remain retired, so speculative handles cannot alias.
    pub fn restore(&mut self, snapshot: &WorldSnapshot) -> bool {
        if snapshot.id != self.id {
            return false;
        }
        self.next = self.next.max(snapshot.next);
        self.settings.clone_from(&snapshot.settings);
        self.bodies.clone_from(&snapshot.bodies);
        self.indices.clear();
        self.indices.extend(
            self.bodies
                .iter()
                .enumerate()
                .map(|(index, (id, _))| (*id, index)),
        );
        self.cache.clone_from(&snapshot.cache);
        self.impulses.clone_from(&snapshot.impulses);
        self.previous_h = snapshot.previous_h;
        self.touching.clone_from(&snapshot.touching);
        self.events.clone_from(&snapshot.events);
        self.debug_contacts.clone_from(&snapshot.debug_contacts);
        self.impacts.clone_from(&snapshot.impacts);
        self.joints.clone_from(&snapshot.joints);
        self.joint_cache.clone_from(&snapshot.joint_cache);
        self.joint_breaks.clone_from(&snapshot.joint_breaks);
        self.dynamic_tree = DynamicTree::default();
        true
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

    /// Updates an existing joint while preserving its handle.
    pub fn update_joint(&mut self, handle: JointHandle, joint: Joint) -> Result<(), InvalidJoint> {
        let previous = *self.joints.get(&handle).ok_or(InvalidJoint)?;
        if !joint.valid() || joint.bodies.iter().any(|id| !self.indices.contains_key(id)) {
            return Err(InvalidJoint);
        }
        self.joints.insert(handle, joint);
        self.joint_cache.remove(&handle);
        for body in previous.bodies.into_iter().chain(joint.bodies) {
            self.bodies[self.indices[&body]].1.wake();
        }
        Ok(())
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
        body.geometry_dirty = true;
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
            || !parent.enabled
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
        self.wake_all();
        Some(removed)
    }

    /// Enables or deactivates a body while preserving its pose and velocity.
    /// Deactivated bodies are skipped by stepping, joints, contacts, and queries.
    pub fn set_enabled(&mut self, handle: BodyHandle, enabled: bool) -> bool {
        let Some(&index) = self.indices.get(&handle) else {
            return false;
        };
        if self.bodies[index].1.enabled == enabled {
            return true;
        }
        self.invalidate_cache(handle);
        let body = &mut self.bodies[index].1;
        body.enabled = enabled;
        body.geometry_dirty = true;
        body.force = Vec3::ZERO;
        body.torque = Vec3::ZERO;
        body.wake();
        if !enabled {
            self.wake_all();
        }
        true
    }

    pub fn body(&self, handle: BodyHandle) -> Option<&RigidBody> {
        Some(&self.bodies[*self.indices.get(&handle)?].1)
    }

    pub fn body_mut(&mut self, handle: BodyHandle) -> Option<&mut RigidBody> {
        let index = *self.indices.get(&handle)?;
        self.invalidate_cache(handle);
        self.bodies[index].1.wake();
        self.bodies[index].1.geometry_dirty = true;
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
        self.debug_contacts
            .retain(|&(a, b, _)| a != handle && b != handle);
        for &((a, b), sensor) in &self.touching {
            if sensor {
                continue;
            }
            let other = if a == handle {
                b
            } else if b == handle {
                a
            } else {
                continue;
            };
            if let Some(&index) = self.indices.get(&other) {
                self.bodies[index].1.wake();
            }
        }
        self.joint_cache
            .retain(|id, _| !self.joints[id].bodies.contains(&handle));
        for joint in self
            .joints
            .values()
            .filter(|joint| joint.bodies.contains(&handle))
        {
            for &endpoint in &joint.bodies {
                if let Some(&index) = self.indices.get(&endpoint) {
                    self.bodies[index].1.wake();
                }
            }
        }
    }

    pub fn bodies(&self) -> impl ExactSizeIterator<Item = (BodyHandle, &RigidBody)> {
        self.bodies.iter().map(|(id, body)| (*id, body))
    }

    /// Transitions observed during the last successful step, in substep order.
    pub fn events(&self) -> &[CollisionEvent] {
        &self.events
    }

    /// Joint failures observed during the last successful step.
    pub fn joint_break_events(&self) -> &[JointBreakEvent] {
        &self.joint_breaks
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
        Ok(self.step_inner::<false, false>(dt)?.stats)
    }

    /// Advances exactly as [Self::step] and records phase timings.
    pub fn step_profiled(&mut self, dt: f64) -> Result<ProfiledStep, StepError> {
        self.step_inner::<true, false>(dt)
    }

    pub fn step_fixed_profiled(&mut self) -> Result<ProfiledStep, StepError> {
        self.step_profiled(self.settings.fixed_dt)
    }

    /// Advances with numerical checks and rolls back on a detected failure.
    /// Capturing a checkpoint and validating each stage adds overhead.
    pub fn step_checked(&mut self, dt: f64) -> Result<StepStats, StepError> {
        let snapshot = self.snapshot();
        match self.step_inner::<false, true>(dt) {
            Ok(step) => Ok(step.stats),
            Err(error) => {
                if matches!(error, StepError::NumericalFailure(_)) {
                    self.restore(&snapshot);
                }
                Err(error)
            }
        }
    }

    fn step_inner<const PROFILE: bool, const CHECKED: bool>(
        &mut self,
        dt: f64,
    ) -> Result<ProfiledStep, StepError> {
        let h = dt / self.settings.substeps as f64;
        if !h.is_finite() || h <= 0.0 {
            return Err(StepError::InvalidTimestep);
        }
        let start = PROFILE.then(Instant::now);
        let mut stats = StepStats::default();
        let mut timings = StepTimings::default();
        self.events.clear();
        self.impacts.clear();
        self.joint_breaks.clear();
        for substep in 0..self.settings.substeps {
            self.substep::<PROFILE, CHECKED>(h, h * substep as f64, &mut stats, &mut timings)?;
        }
        self.impacts.sort_unstable_by(|a, b| {
            a.time
                .total_cmp(&b.time)
                .then(a.moving.cmp(&b.moving))
                .then(a.target.cmp(&b.target))
        });
        stats.ccd_impacts = self.impacts.len();
        stats.broken_joints = self.joint_breaks.len();
        for (_, body) in &mut self.bodies {
            body.force = Vec3::ZERO;
            body.torque = Vec3::ZERO;
            stats.sleeping_bodies += usize::from(body.asleep);
        }
        if let Some(start) = start {
            timings.total = start.elapsed();
        }
        Ok(ProfiledStep { stats, timings })
    }

    fn substep<const PROFILE: bool, const CHECKED: bool>(
        &mut self,
        h: f64,
        start: f64,
        stats: &mut StepStats,
        timings: &mut StepTimings,
    ) -> Result<(), StepError> {
        let mut mark = PROFILE.then(Instant::now);
        self.wake_joint_network();
        for (_, body) in &mut self.bodies {
            if body.kind == BodyKind::Dynamic && !body.asleep && body.enabled {
                body.velocity += (self.settings.gravity * body.gravity_scale * body.linear_axes
                    + body.apply_inverse_mass(body.force))
                    * h;
                body.angular_momentum += body.torque * body.angular_axes * h;
                body.velocity *= (-body.damping[0] * h).exp();
                body.angular_momentum *= (-body.damping[1] * h).exp();
            }
        }
        if CHECKED {
            self.check_state(NumericalStage::Forces)?;
        }
        self.prepare_joints(h);
        self.warm_start_joints(h);
        for _ in 0..self.settings.iterations {
            self.solve_joints();
        }
        if CHECKED {
            self.check_state(NumericalStage::Forces)?;
        }
        record_phase::<PROFILE>(&mut mark, &mut timings.forces_and_joints);
        stats.ccd_budget_fallbacks += self.advance_poses(h, start);
        record_phase::<PROFILE>(&mut mark, &mut timings.pose_and_ccd);
        if CHECKED {
            self.check_state(NumericalStage::Poses)?;
            if self.bodies.iter().any(|(_, b)| {
                b.enabled
                    && Aabb::of(b)
                        .is_some_and(|aabb| !aabb.min.is_finite() || !aabb.max.is_finite())
            }) || self
                .impacts
                .iter()
                .any(|hit| !hit.point.is_finite() || !hit.normal.is_normalized())
            {
                return Err(StepError::NumericalFailure(NumericalStage::Poses));
            }
        }
        self.collect_candidates();
        stats.candidate_pairs += self.candidates.len();
        record_phase::<PROFILE>(&mut mark, &mut timings.broad_phase);

        self.build_manifolds(stats);
        if CHECKED
            && self.debug_contacts.iter().any(|(_, _, contact)| {
                !contact.point.is_finite()
                    || !contact.normal.is_normalized()
                    || !contact.depth.is_finite()
                    || contact.depth < 0.0
            })
        {
            return Err(StepError::NumericalFailure(NumericalStage::Contacts));
        }
        record_phase::<PROFILE>(&mut mark, &mut timings.narrow_phase);
        if self.settings.sleep_time > 0.0 {
            self.prepare_islands();
        }
        for (_, body) in &mut self.bodies {
            body.geometry_dirty = false;
        }
        self.prepare_solver(h);
        stats.solved_parallel_islands += self.solve_rows();
        self.cache_joint_impulses();
        self.finish_contacts();
        self.break_joints(h);
        if CHECKED {
            self.check_state(NumericalStage::Solve)?;
        }
        record_phase::<PROFILE>(&mut mark, &mut timings.solver);
        if self.settings.sleep_time > 0.0 {
            self.update_sleep(h);
        }
        self.publish_events();
        self.previous_h = Some(h);
        record_phase::<PROFILE>(&mut mark, &mut timings.sleep_and_events);
        Ok(())
    }

    fn check_state(&self, stage: NumericalStage) -> Result<(), StepError> {
        if self.bodies.iter().any(|(_, body)| {
            !body.position.is_finite()
                || !body.rotation.is_normalized()
                || !body.velocity.is_finite()
                || !body.angular_momentum.is_finite()
                || !body.angular_velocity().is_finite()
        }) {
            Err(StepError::NumericalFailure(stage))
        } else {
            Ok(())
        }
    }

    fn build_manifolds(&mut self, stats: &mut StepStats) {
        self.scratch.active.clear();
        self.scratch.manifolds.clear();
        #[cfg(feature = "rayon")]
        if self.settings.parallel_narrow_phase && self.candidates.len() >= 128 {
            self.debug_contacts.clear();
            self.build_parallel_manifolds(stats);
            return;
        }
        let reuse = self.bodies.iter().any(|(_, b)| b.asleep) && !self.debug_contacts.is_empty();
        let mut previous = std::mem::take(&mut self.scratch.previous_geometry);
        if reuse {
            std::mem::swap(&mut previous, &mut self.debug_contacts);
        }
        self.debug_contacts.clear();

        let mut history = previous
            .chunk_by(|a, b| (a.0, a.1) == (b.0, b.1))
            .peekable();
        let mut geometry = std::mem::take(&mut self.scratch.geometry);
        for i in 0..self.candidates.len() {
            let indices @ [ai, bi] = self.candidates[i];
            let (ah, a) = &self.bodies[ai];
            let (bh, b) = &self.bodies[bi];
            geometry.clear();
            if reuse
                && (a.asleep || b.asleep)
                && !a.geometry_dirty
                && !b.geometry_dirty
                && effective_mass(a) + effective_mass(b) == 0.0
            {
                let pair = (*ah, *bh);
                while history
                    .peek()
                    .is_some_and(|old| (old[0].0, old[0].1) < pair)
                {
                    history.next();
                }
                if let Some(old) = history.peek()
                    && (old[0].0, old[0].1) == pair
                {
                    geometry.extend(old.iter().map(|&(_, _, point)| point));
                    stats.reused_sleeping_manifolds += 1;
                }
            }
            if geometry.is_empty() {
                contacts(a, b, &mut geometry);
            }
            self.record_manifold(indices, &geometry, stats);
        }
        self.scratch.geometry = geometry;
        self.scratch.previous_geometry = previous;
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

    fn wake_joint_network(&mut self) {
        if self.joints.is_empty() || !self.bodies.iter().any(|(_, b)| b.asleep) {
            return;
        }
        let islands = &mut self.scratch.islands;
        islands.reset(self.bodies.len());
        for joint in self.joints.values() {
            let Some([a, b]) = joint_indices(joint, &self.indices, &self.bodies) else {
                continue;
            };
            if self.bodies[a].1.kind == BodyKind::Dynamic
                && self.bodies[b].1.kind == BodyKind::Dynamic
            {
                islands.join(a, b);
            }
        }
        for joint in self.joints.values() {
            let Some([a, b]) = joint_indices(joint, &self.indices, &self.bodies) else {
                continue;
            };
            for (dynamic, other) in [(a, b), (b, a)] {
                let body = &self.bodies[dynamic].1;
                let other = &self.bodies[other].1;
                if body.kind != BodyKind::Dynamic {
                    continue;
                }
                let root = islands.root(dynamic);
                islands.nodes[root].awake |=
                    !body.asleep || should_wake(other) || joint.requires_awake();
            }
        }
        for (i, (_, body)) in self.bodies.iter_mut().enumerate() {
            if body.asleep {
                let root = islands.root(i);
                if islands.nodes[root].awake {
                    body.wake();
                }
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
            let Some([a, b]) = joint_indices(joint, &self.indices, &self.bodies) else {
                continue;
            };
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
                    let support = &self.bodies[support].1;
                    islands.nodes[root].busy |= should_wake(support) || support.geometry_dirty;
                }
            }
        }
        for joint in self.joints.values() {
            let Some([a, b]) = joint_indices(joint, &self.indices, &self.bodies) else {
                continue;
            };
            for (dynamic, other) in [(a, b), (b, a)] {
                if kind(dynamic) != BodyKind::Dynamic {
                    continue;
                }
                let root = islands.root(dynamic);
                islands.nodes[root].grounded |= kind(other) != BodyKind::Dynamic;
                islands.nodes[root].busy |= joint.requires_awake()
                    || (kind(other) == BodyKind::Kinematic && should_wake(&self.bodies[other].1));
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
            let Some([ai, bi]) = joint_indices(joint, &self.indices, &self.bodies) else {
                continue;
            };
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

    fn break_joints(&mut self, h: f64) {
        for batch in &self.scratch.joint_batches {
            let joint = self.joints[&batch.handle];
            let Some(limit) = joint.break_limit else {
                continue;
            };
            let (linear, angular) = self.scratch.joint_rows[batch.rows.clone()]
                .iter()
                .map(JointRow::reaction)
                .fold((Vec3::ZERO, Vec3::ZERO), |(a, b), (c, d)| (a + c, b + d));
            let force = linear.length() / h;
            let torque = angular.length() / h;
            if limit.force.is_some_and(|max| force > max)
                || limit.torque.is_some_and(|max| torque > max)
            {
                self.joint_breaks.push(JointBreakEvent {
                    joint: batch.handle,
                    bodies: joint.bodies,
                    force,
                    torque,
                });
                self.joints.remove(&batch.handle);
                self.joint_cache.remove(&batch.handle);
                for i in batch.indices {
                    self.bodies[i].1.wake();
                }
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

    fn solve_rows(&mut self) -> usize {
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
            let coupled = m
                .indices
                .into_iter()
                .zip([&*a, &*b])
                .any(|(i, body)| effective_mass(body) > 0.0 && scratch.solver_degree[i] > 1);
            if coupled {
                warm_contact_rows(a, b, rows);
                scratch.coupled_manifolds.push(index);
            } else {
                solve_independent(a, b, rows, self.settings.iterations);
            }
        }
        #[cfg(feature = "rayon")]
        if self.settings.parallel_islands {
            let islands = self.solve_parallel_islands();
            if islands > 0 {
                return islands;
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
        0
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
            if !body.enabled {
                continue;
            }
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
        let sweep = self.settings.ccd
            && self
                .bodies
                .iter()
                .any(|(_, b)| sweeps(b, self.settings.ccd_convex));
        if !sweep {
            for (_, body) in &mut self.bodies {
                body.integrate_pose(h);
            }
            return 0;
        }
        let synchronized = self.settings.ccd_moving_spheres;
        self.ccd_targets
            .extend(self.bodies.iter().enumerate().filter_map(|(i, (_, b))| {
                (b.enabled
                    && !b.sensor
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
                    advance_ccd_pose(&mut self.bodies[i].1, remaining, self.settings.ccd_convex);
                    remaining = 0.0;
                    break;
                };
                advance_ccd_pose(&mut self.bodies[i].1, hit.time, self.settings.ccd_convex);
                remaining = (remaining - hit.time).max(0.0);
                self.resolve_impact(hit, start + (h - remaining));
                if remaining == 0.0 {
                    break;
                }
            }
            if remaining > 0.0 {
                advance_ccd_pose(&mut self.bodies[i].1, remaining, self.settings.ccd_convex);
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
                advance_ccd_pose(body, dt, self.settings.ccd_convex);
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
            advance_ccd_pose(body, remaining, self.settings.ccd_convex);
        }
        1
    }

    fn first_impact(&self, index: usize, dt: f64, synchronized: bool) -> Option<ImpactCandidate> {
        let (handle, body) = &self.bodies[index];
        if !sweeps(body, self.settings.ccd_convex) {
            return None;
        }
        let mut best: Option<ImpactCandidate> = None;
        for &j in &self.ccd_targets {
            let (other_handle, target) = &self.bodies[j];
            if index == j || !compatible(body, target) {
                continue;
            }
            let max_time = best.as_ref().map_or(dt, |hit| hit.time);
            let (time, contact) = if let Shape::Sphere { radius } = body.shape {
                if synchronized && sweeps(target, false) && handle > other_handle {
                    continue;
                }
                let relative = body.velocity - target.velocity;
                let Some(direction) = relative.try_normalize() else {
                    continue;
                };
                let speed = relative.length();
                if speed * dt <= radius * 0.1 {
                    continue;
                }
                let Some((distance, normal)) =
                    sweep_sphere(target, body.position, direction, radius, speed * max_time)
                else {
                    continue;
                };
                if normal.dot(body.apply_inverse_mass(normal) + target.apply_inverse_mass(normal))
                    <= 0.0
                {
                    continue;
                }
                (distance / speed, ImpactGeometry::Sphere { radius, normal })
            } else {
                if target.kind != BodyKind::Static {
                    continue;
                }
                let Some(hit) = crate::motion::cast(body, target, max_time) else {
                    continue;
                };
                (
                    hit.time,
                    ImpactGeometry::Convex {
                        point: hit.point,
                        normal: hit.normal,
                        separation: hit.separation,
                    },
                )
            };
            if best.as_ref().is_none_or(|previous| {
                time < previous.time
                    || (time == previous.time && *other_handle < self.bodies[previous.target].0)
            }) {
                best = Some(ImpactCandidate {
                    time,
                    moving: index,
                    target: j,
                    contact,
                });
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
        let (point, normal, separation) = match hit.contact {
            ImpactGeometry::Sphere { radius, normal } => {
                (a.position - normal * radius, normal, radius * 1e-9)
            }
            ImpactGeometry::Convex {
                point,
                normal,
                separation,
            } => (point, normal, separation),
        };
        let contact = Contact {
            point,
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
        separate(b, a, normal, separation);
        self.impacts.push(CcdImpact {
            moving: *moving,
            target: *target,
            point: contact.point,
            normal,
            time,
        });
    }
}

fn record_phase<const PROFILE: bool>(mark: &mut Option<Instant>, duration: &mut Duration) {
    if PROFILE {
        let now = Instant::now();
        if let Some(previous) = mark.replace(now) {
            *duration += now - previous;
        }
    }
}

fn advance_ccd_pose(body: &mut RigidBody, dt: f64, convex: bool) {
    if convex && !matches!(body.shape, Shape::Sphere { .. }) && sweeps(body, true) {
        body.integrate_swept_pose(dt);
    } else {
        body.integrate_pose(dt);
    }
}

fn sweeps(body: &RigidBody, convex: bool) -> bool {
    body.ccd
        && body.enabled
        && !body.sensor
        && !body.asleep
        && body.kind == BodyKind::Dynamic
        && (matches!(body.shape, Shape::Sphere { .. })
            || (convex && crate::convex::PrimitiveSupport::new(body).is_some()))
}

fn joint_indices(
    joint: &Joint,
    indices: &BTreeMap<BodyHandle, usize>,
    bodies: &[(BodyHandle, RigidBody)],
) -> Option<[usize; 2]> {
    let pair = joint.bodies.map(|id| indices[&id]);
    pair.iter().all(|&i| bodies[i].1.enabled).then_some(pair)
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
    if b.asleep || !b.enabled {
        0.0
    } else {
        b.inverse_mass
    }
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
            let linear = direction.dot(body.apply_inverse_mass(direction));
            if axis == Vec3::ZERO {
                linear
            } else {
                linear + axis.dot(body.apply_inverse_inertia(axis))
            }
        })
        .sum()
}

fn warm_contact_rows(a: &mut RigidBody, b: &mut RigidBody, rows: &[Row]) {
    for row in rows {
        let impulse = row.contact.normal * row.cache.normal_impulse + row.cache.friction;
        apply(a, b, row.offsets, impulse);
    }
}

fn solve_independent(a: &mut RigidBody, b: &mut RigidBody, rows: &mut [Row], iterations: usize) {
    warm_contact_rows(a, b, rows);
    for _ in 0..iterations {
        let mut changed = false;
        for row in &mut *rows {
            changed |= solve(a, b, row);
        }
        if !changed {
            break;
        }
    }
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
        a.geometry_dirty |= ra != Vec3::ZERO;
        b.geometry_dirty |= rb != Vec3::ZERO;
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
    let local = [(a, offsets[0]), (b, offsets[1])].map(|(body, offset)| {
        // A sphere's contact anchor depends on its center and normal, not
        // on the orientation of its otherwise identical surface.
        if matches!(body.shape, Shape::Sphere { .. }) || body.rotation == crate::Quat::IDENTITY {
            offset
        } else {
            body.rotation.conjugate() * offset
        }
    });
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
