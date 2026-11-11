//! Structure-of-arrays particles with emitters and independent motion updates.
use crate::grid::SpatialHash;
use crate::spatial::project;
use crate::{BodyHandle, PhysicsWorld, QueryFilter, RigidBody, Vec3};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Particle {
    pub position: Vec3,
    pub velocity: Vec3,
    pub remaining: f64,
}

#[derive(Clone, Copy, Debug)]
pub enum Field {
    /// Plummer-softened attraction. Negative strength repels.
    Attractor {
        center: Vec3,
        strength: f64,
        softening: f64,
    },
    /// Tangential acceleration proportional to distance from a unit axis.
    Vortex {
        center: Vec3,
        axis: Vec3,
        strength: f64,
    },
    /// Bounded periodic acceleration for effects.
    Waves { amplitude: f64, frequency: f64 },
}

impl Field {
    fn valid(self) -> bool {
        match self {
            Self::Attractor {
                center,
                strength,
                softening,
            } => {
                center.is_finite()
                    && strength.is_finite()
                    && softening.is_finite()
                    && softening > 0.0
            }
            Self::Vortex {
                center,
                axis,
                strength,
            } => center.is_finite() && axis.is_normalized() && strength.is_finite(),
            Self::Waves {
                amplitude,
                frequency,
            } => amplitude.is_finite() && frequency.is_finite(),
        }
    }

    fn acceleration(self, p: Vec3) -> Vec3 {
        match self {
            Self::Attractor {
                center,
                strength,
                softening,
            } => {
                let offset = center - p;
                offset * (strength / (offset.length_squared() + softening * softening).powf(1.5))
            }
            Self::Vortex {
                center,
                axis,
                strength,
            } => axis.cross(p - center) * strength,
            Self::Waves {
                amplitude,
                frequency,
            } => {
                Vec3::new(
                    (p.y * frequency).sin(),
                    (p.z * frequency).sin(),
                    (p.x * frequency).sin(),
                ) * amplitude
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ParticleSettings {
    pub gravity: Vec3,
    pub drag: f64,
    pub radius: f64,
    pub restitution: f64,
}

impl Default for ParticleSettings {
    fn default() -> Self {
        Self {
            gravity: Vec3::new(0.0, -9.81, 0.0),
            drag: 0.0,
            radius: 0.02,
            restitution: 0.4,
        }
    }
}

impl ParticleSettings {
    fn valid(self) -> bool {
        self.gravity.is_finite()
            && self.drag.is_finite()
            && self.drag >= 0.0
            && self.radius.is_finite()
            && self.radius >= 0.0
            && (0.0..=1.0).contains(&self.restitution)
    }
}

#[derive(Default)]
pub struct Particles {
    positions: Vec<Vec3>,
    velocities: Vec<Vec3>,
    lifetimes: Vec<f64>,
    pair_grid: Option<(f64, SpatialHash)>,
    pair_positions: Vec<Vec3>,
}

impl Particles {
    pub fn len(&self) -> usize {
        self.positions.len()
    }
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }
    pub fn positions(&self) -> &[Vec3] {
        &self.positions
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = Particle> + '_ {
        self.positions
            .iter()
            .zip(&self.velocities)
            .zip(&self.lifetimes)
            .map(|((&position, &velocity), &remaining)| Particle {
                position,
                velocity,
                remaining,
            })
    }

    pub fn push(&mut self, particle: Particle) -> bool {
        if !particle.position.is_finite()
            || !particle.velocity.is_finite()
            || !particle.remaining.is_finite()
            || particle.remaining <= 0.0
        {
            return false;
        }
        self.positions.push(particle.position);
        self.velocities.push(particle.velocity);
        self.lifetimes.push(particle.remaining);
        true
    }

    /// Advances particle motion and removes expired particles in insertion order.
    /// Returns false for invalid inputs without advancing the pool.
    pub fn step(&mut self, dt: f64, settings: ParticleSettings, fields: &[Field]) -> bool {
        if !valid_step(dt, settings, fields) {
            return false;
        }
        let damping = (-settings.drag * dt).exp();
        for ((p, v), life) in self
            .positions
            .iter_mut()
            .zip(&mut self.velocities)
            .zip(&mut self.lifetimes)
        {
            integrate(p, v, life, dt, settings.gravity, damping, fields);
        }
        self.retain_alive();
        true
    }

    /// Parallel independent update. Cleanup retains the same ordering as [Self::step].
    #[cfg(feature = "rayon")]
    pub fn step_parallel(&mut self, dt: f64, settings: ParticleSettings, fields: &[Field]) -> bool {
        use rayon::prelude::*;
        if !valid_step(dt, settings, fields) {
            return false;
        }
        let damping = (-settings.drag * dt).exp();
        self.positions
            .par_iter_mut()
            .zip(&mut self.velocities)
            .zip(&mut self.lifetimes)
            .for_each(|((p, v), life)| {
                integrate(p, v, life, dt, settings.gravity, damping, fields)
            });
        self.retain_alive();
        true
    }

    fn retain_alive(&mut self) {
        let mut target = 0;
        for i in 0..self.len() {
            if self.lifetimes[i] > 0.0 {
                if i != target {
                    self.positions[target] = self.positions[i];
                    self.velocities[target] = self.velocities[i];
                    self.lifetimes[target] = self.lifetimes[i];
                }
                target += 1;
            }
        }
        self.positions.truncate(target);
        self.velocities.truncate(target);
        self.lifetimes.truncate(target);
    }

    /// Resolves one pass of equal-mass sphere pairs using a spatial hash.
    /// Runs after motion and expiration. Additional passes can resolve dense
    /// overlaps introduced by the first pass.
    pub fn collide_pairs(&mut self, radius: f64, restitution: f64) -> bool {
        let diameter = 2.0 * radius;
        if !diameter.is_finite() || diameter <= 0.0 || !(0.0..=1.0).contains(&restitution) {
            return false;
        }
        if self
            .pair_grid
            .as_ref()
            .is_none_or(|(width, _)| *width != diameter)
        {
            self.pair_grid = Some((
                diameter,
                SpatialHash::new(diameter).expect("validated diameter"),
            ));
        }
        let (_, grid) = self.pair_grid.as_mut().expect("initialized pair grid");
        self.pair_positions.clear();
        self.pair_positions.extend_from_slice(&self.positions);
        grid.rebuild(&self.pair_positions);
        for (i, &position) in self.pair_positions.iter().enumerate() {
            for j in grid.neighbors(position).filter(|&j| j > i) {
                let [a, b] = self
                    .positions
                    .get_disjoint_mut([i, j])
                    .expect("distinct particles");
                let delta = *b - *a;
                let distance = delta.length();
                if distance >= diameter {
                    continue;
                }
                let normal = delta.try_normalize().unwrap_or(Vec3::AXES[(i + j) % 3]);
                let correction = normal * ((diameter - distance) * 0.5);
                *a -= correction;
                *b += correction;
                let [va, vb] = self
                    .velocities
                    .get_disjoint_mut([i, j])
                    .expect("distinct particles");
                let speed = (*vb - *va).dot(normal);
                if speed < 0.0 {
                    let impulse = normal * (-(1.0 + restitution) * speed * 0.5);
                    *va -= impulse;
                    *vb += impulse;
                }
            }
        }
        true
    }

    /// Exchanges normal impulses with dynamic rigid boundaries. Particles
    /// share the supplied mass, and the rigid world advances separately.
    pub fn collide_coupled(
        &mut self,
        world: &mut PhysicsWorld,
        filter: QueryFilter,
        settings: ParticleSettings,
        mass: f64,
    ) -> bool {
        let inverse_mass = mass.recip();
        if !settings.valid() || !inverse_mass.is_finite() || inverse_mass <= 0.0 {
            return false;
        }
        let targets = boundary_targets(world, filter);
        for (position, velocity) in self.positions.iter_mut().zip(&mut self.velocities) {
            coupled_boundary(
                position,
                velocity,
                settings.radius,
                settings.restitution,
                inverse_mass,
                world,
                &targets,
            );
        }
        true
    }

    /// Resolves discrete particle contacts against the rigid world. Collider
    /// surface velocity contributes to bounce. Coupling is one-way.
    pub fn collide(
        &mut self,
        world: &PhysicsWorld,
        filter: QueryFilter,
        settings: ParticleSettings,
    ) {
        if !settings.valid() {
            return;
        }
        for (position, velocity) in self.positions.iter_mut().zip(&mut self.velocities) {
            collide_point(
                position,
                velocity,
                settings.radius,
                settings.restitution,
                world,
                filter,
            );
        }
    }
}

fn valid_step(dt: f64, settings: ParticleSettings, fields: &[Field]) -> bool {
    dt.is_finite() && dt > 0.0 && settings.valid() && fields.iter().all(|field| field.valid())
}

fn integrate(
    p: &mut Vec3,
    v: &mut Vec3,
    life: &mut f64,
    dt: f64,
    gravity: Vec3,
    damping: f64,
    fields: &[Field],
) {
    *life -= dt;
    if *life <= 0.0 {
        return;
    }
    let acceleration = fields
        .iter()
        .fold(gravity, |acc, &field| acc + field.acceleration(*p));
    *v = (*v + acceleration * dt) * damping;
    *p += *v * dt;
}

pub(crate) fn collide_point(
    position: &mut Vec3,
    velocity: &mut Vec3,
    radius: f64,
    restitution: f64,
    world: &PhysicsWorld,
    filter: QueryFilter,
) {
    project_boundaries(position, radius, world, filter, |body, surface, normal| {
        let surface_speed = body.point_velocity(surface - body.position());
        let speed = (*velocity - surface_speed).dot(normal);
        if speed < 0.0 {
            *velocity -= (1.0 + restitution) * speed * normal;
        }
    });
}

pub(crate) fn project_boundaries(
    position: &mut Vec3,
    radius: f64,
    world: &PhysicsWorld,
    filter: QueryFilter,
    mut on_contact: impl FnMut(&RigidBody, Vec3, Vec3),
) {
    for (id, body) in world.bodies() {
        if body.is_sensor() || !filter.accepts(id, body) {
            continue;
        }
        let (surface, normal, distance) = project(body, *position);
        if distance < radius {
            *position = surface + radius * normal;
            on_contact(body, surface, normal);
        }
    }
}

pub(crate) fn boundary_targets(world: &PhysicsWorld, filter: QueryFilter) -> Vec<BodyHandle> {
    world
        .bodies()
        .filter_map(|(id, body)| (!body.is_sensor() && filter.accepts(id, body)).then_some(id))
        .collect()
}

pub(crate) fn coupled_boundary(
    position: &mut Vec3,
    velocity: &mut Vec3,
    radius: f64,
    restitution: f64,
    inverse_mass: f64,
    world: &mut PhysicsWorld,
    targets: &[BodyHandle],
) {
    for &handle in targets {
        let body = world.body(handle).expect("collected boundary");
        let (surface, normal, distance) = project(body, *position);
        if distance >= radius {
            continue;
        }
        *position = surface + radius * normal;
        let offset = surface - body.position();
        let speed = (*velocity - body.point_velocity(offset)).dot(normal);
        if speed >= 0.0 {
            continue;
        }
        let cross = offset.cross(normal);
        let response = normal.dot(body.apply_inverse_mass(normal))
            + cross.dot(body.apply_inverse_inertia(cross));
        let impulse = normal * (-(1.0 + restitution) * speed / (inverse_mass + response));
        *velocity += impulse * inverse_mass;
        world.apply_impulse(handle, -impulse, surface);
    }
}

/// Reproducible burst emitter. Each velocity component varies independently.
pub struct Emitter {
    state: u64,
}

impl Emitter {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn sample(&mut self) -> f64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut n = self.state;
        n = (n ^ (n >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        n = (n ^ (n >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        ((n ^ (n >> 31)) >> 11) as f64 * (2.0 / (1_u64 << 53) as f64) - 1.0
    }

    pub fn burst(
        &mut self,
        pool: &mut Particles,
        count: usize,
        template: Particle,
        spread: Vec3,
    ) -> usize {
        if !spread.is_finite() || spread.min_element() < 0.0 {
            return 0;
        }
        (0..count)
            .filter(|_| {
                let jitter = Vec3::new(self.sample(), self.sample(), self.sample()) * spread;
                pool.push(Particle {
                    velocity: template.velocity + jitter,
                    ..template
                })
            })
            .count()
    }
}
