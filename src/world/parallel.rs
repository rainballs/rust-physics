//! Parallel solving for disjoint coupled islands. Sequential ordering is
//! preserved within each island and shared fixed endpoints are read-only.
use super::solver::solve;
use super::{JointRow, PhysicsWorld, RigidBody, Row, effective_mass};
use crate::Vec3;
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::ops::Range;

#[derive(Default)]
struct Work {
    joints: Vec<usize>,
    contacts: Vec<usize>,
}

struct Rows {
    source: Range<usize>,
    local: Range<usize>,
    bodies: [usize; 2],
}

struct Result {
    momenta: Vec<(usize, Vec3, Vec3)>,
    joints: Vec<Rows>,
    joint_rows: Vec<JointRow>,
    contacts: Vec<Rows>,
    contact_rows: Vec<Row>,
}

impl PhysicsWorld {
    pub(super) fn solve_parallel_islands(&mut self) -> usize {
        let scratch = &mut self.scratch;
        if scratch.coupled_manifolds.len() + scratch.joint_batches.len() < 256 {
            return 0;
        }
        if self.settings.sleep_time == 0.0 {
            scratch.islands.reset(self.bodies.len());
            let pairs = scratch
                .coupled_manifolds
                .iter()
                .map(|&i| scratch.manifolds[i].indices)
                .chain(scratch.joint_batches.iter().map(|b| b.indices));
            for [a, b] in pairs {
                if effective_mass(&self.bodies[a].1) > 0.0
                    && effective_mass(&self.bodies[b].1) > 0.0
                {
                    scratch.islands.join(a, b);
                }
            }
        }
        let mut islands: BTreeMap<usize, Work> = BTreeMap::new();
        for (index, batch) in scratch.joint_batches.iter().enumerate() {
            if batch.rows.is_empty() {
                continue;
            }
            if let Some(body) = batch
                .indices
                .into_iter()
                .find(|&i| effective_mass(&self.bodies[i].1) > 0.0)
            {
                islands
                    .entry(scratch.islands.root(body))
                    .or_default()
                    .joints
                    .push(index);
            }
        }
        for &index in &scratch.coupled_manifolds {
            let manifold = &scratch.manifolds[index];
            let body = manifold
                .indices
                .into_iter()
                .find(|&i| effective_mass(&self.bodies[i].1) > 0.0)
                .expect("a solved manifold has a dynamic endpoint");
            islands
                .entry(scratch.islands.root(body))
                .or_default()
                .contacts
                .push(index);
        }
        if islands.len() < 2 {
            return 0;
        }
        let jobs: Vec<_> = islands.into_values().collect();
        let count = jobs.len();
        let results: Vec<_> = jobs
            .into_par_iter()
            .map(|work| solve_work(self, work))
            .collect();
        for output in results {
            for (index, velocity, momentum) in output.momenta {
                let body = &mut self.bodies[index].1;
                body.velocity = velocity;
                body.angular_momentum = momentum;
            }
            for batch in output.joints {
                self.scratch.joint_rows[batch.source]
                    .clone_from_slice(&output.joint_rows[batch.local]);
            }
            for batch in output.contacts {
                self.scratch.rows[batch.source].clone_from_slice(&output.contact_rows[batch.local]);
            }
        }
        count
    }
}

fn solve_work(world: &PhysicsWorld, work: Work) -> Result {
    let scratch = &world.scratch;
    let joint_batches = work.joints.iter().map(|&i| {
        let batch = &scratch.joint_batches[i];
        (batch.indices, batch.rows.clone())
    });
    let contact_batches = work.contacts.iter().map(|&i| {
        let manifold = &scratch.manifolds[i];
        (manifold.indices, manifold.rows.clone())
    });
    let mut ids: Vec<_> = joint_batches
        .clone()
        .chain(contact_batches.clone())
        .flat_map(|(ids, _)| ids)
        .collect();
    ids.sort_unstable();
    ids.dedup();
    let mut bodies: Vec<RigidBody> = ids.iter().map(|&i| world.bodies[i].1.clone()).collect();
    let mut joint_rows = Vec::new();
    let mut contact_rows = Vec::new();
    let joints = copy_rows(joint_batches, &scratch.joint_rows, &ids, &mut joint_rows);
    let contacts = copy_rows(contact_batches, &scratch.rows, &ids, &mut contact_rows);

    for _ in 0..world.settings.iterations {
        for batch in &joints {
            let [a, b] = bodies
                .get_disjoint_mut(batch.bodies)
                .expect("distinct endpoints");
            for row in &mut joint_rows[batch.local.clone()] {
                row.solve(a, b);
            }
        }
        for batch in &contacts {
            let [a, b] = bodies
                .get_disjoint_mut(batch.bodies)
                .expect("distinct endpoints");
            for row in &mut contact_rows[batch.local.clone()] {
                solve(a, b, row);
            }
        }
    }
    let momenta = ids
        .into_iter()
        .zip(bodies)
        .filter_map(|(index, body)| {
            (effective_mass(&body) > 0.0).then_some((index, body.velocity, body.angular_momentum))
        })
        .collect();
    Result {
        momenta,
        joints,
        joint_rows,
        contacts,
        contact_rows,
    }
}

fn copy_rows<T: Clone>(
    batches: impl Iterator<Item = ([usize; 2], Range<usize>)>,
    source: &[T],
    ids: &[usize],
    out: &mut Vec<T>,
) -> Vec<Rows> {
    batches
        .map(|(indices, range)| {
            let bodies = indices.map(|id| ids.binary_search(&id).expect("island endpoint"));
            let start = out.len();
            out.extend_from_slice(&source[range.clone()]);
            Rows {
                source: range,
                local: start..out.len(),
                bodies,
            }
        })
        .collect()
}
