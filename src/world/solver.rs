//! Contact and joint preparation, impulse iterations, and island sleep state.
use super::{
    BodyHandle, BodyKind, Cached, Contact, Joint, JointBatch, JointBreakEvent, JointRow,
    PhysicsWorld, RigidBody, Row, Settings, Shape, Vec3,
};
use crate::joints;
use std::collections::BTreeMap;

impl PhysicsWorld {
    pub(super) fn wake_joint_network(&mut self) {
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

    pub(super) fn prepare_islands(&mut self) {
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

    pub(super) fn prepare_joints(&mut self, h: f64) {
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

    pub(super) fn warm_start_joints(&mut self, h: f64) {
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

    pub(super) fn solve_joints(&mut self) {
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

    pub(super) fn break_joints(&mut self, h: f64) {
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

    pub(super) fn cache_joint_impulses(&mut self) {
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

    pub(super) fn prepare_solver(&mut self, h: f64) {
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

    pub(super) fn solve_rows(&mut self) -> usize {
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

    pub(super) fn finish_contacts(&mut self) {
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

    pub(super) fn update_sleep(&mut self, h: f64) {
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
}

fn joint_indices(
    joint: &Joint,
    indices: &BTreeMap<BodyHandle, usize>,
    bodies: &[(BodyHandle, RigidBody)],
) -> Option<[usize; 2]> {
    let pair = joint.bodies.map(|id| indices[&id]);
    pair.iter().all(|&i| bodies[i].1.enabled).then_some(pair)
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

pub(super) fn separate(a: &mut RigidBody, b: &mut RigidBody, normal: Vec3, amount: f64) {
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

pub(super) fn make_row(
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

pub(super) fn solve(a: &mut RigidBody, b: &mut RigidBody, row: &mut Row) -> bool {
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
