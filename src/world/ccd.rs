//! Static-target and synchronized continuous collision advancement.
use super::broad_phase::compatible;
use super::solver::{make_row, separate, solve};
use super::{BodyKind, CcdImpact, Contact, PhysicsWorld, RigidBody, Shape, Vec3};
use crate::spatial::sweep_sphere;

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

impl PhysicsWorld {
    pub(super) fn advance_poses(&mut self, h: f64, start: f64) -> usize {
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
