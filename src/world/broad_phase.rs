//! Broad-phase candidate enumeration and deterministic pair ordering.
use super::{BodyHandle, BodyKind, BroadPhase, PhysicsWorld, Proxy, RigidBody, Vec3};
use crate::spatial::Aabb;

impl PhysicsWorld {
    pub(super) fn collect_candidates(&mut self) {
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
}

pub(super) fn compatible(a: &RigidBody, b: &RigidBody) -> bool {
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
