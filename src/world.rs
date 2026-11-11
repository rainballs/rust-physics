//! Deterministic rigid-body stepping, contacts, collision events, and spatial queries.
use std::collections::BTreeMap;
use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use crate::collision::Contact;
use crate::dynamic_tree::DynamicTree;
use crate::grid::ProxyGrid;
use crate::joints::{Joint, JointHandle, JointImpulse, JointRow};
use crate::spatial::{Bvh, Proxy};
use crate::{BodyKind, RigidBody, RigidBodyBuilder, Shape, Vec3};

mod broad_phase;
mod ccd;
mod manifolds;
mod queries;
mod solver;
mod step;

pub(crate) use solver::{effective_mass, inverse_effective_mass, relative_velocity};

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
}
