//! Narrow-phase contact collection and overlap transitions.
#[cfg(feature = "rayon")]
use super::GeometryPatch;
use super::{CollisionEvent, Contact, Manifold, PhysicsWorld, StepStats, effective_mass};
use crate::collision::contacts;

impl PhysicsWorld {
    pub(super) fn build_manifolds(&mut self, stats: &mut StepStats) {
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

    pub(super) fn publish_events(&mut self) {
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
}
