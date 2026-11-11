//! Experimental weakly compressible smoothed particle hydrodynamics (SPH).
use crate::grid::SpatialHash;
use crate::particles::{boundary_targets, collide_point, coupled_boundary};
use crate::{PhysicsWorld, QueryFilter, Vec3};
use std::f64::consts::PI;
mod pbf;
pub use pbf::{PbfFluid, PbfSettings};

#[derive(Clone, Copy, Debug)]
pub struct FluidSettings {
    pub smoothing_radius: f64,
    pub particle_mass: f64,
    pub rest_density: f64,
    /// Slope of the linear pressure equation of state (speed of sound squared).
    pub pressure_stiffness: f64,
    pub viscosity: f64,
    /// Color-field surface tension coefficient. Zero disables the term.
    pub surface_tension: f64,
    pub gravity: Vec3,
    pub collision_radius: f64,
    pub restitution: f64,
}

impl Default for FluidSettings {
    fn default() -> Self {
        Self {
            smoothing_radius: 0.32,
            particle_mass: 4.096,
            rest_density: 1_000.0,
            pressure_stiffness: 50.0,
            viscosity: 2.0,
            surface_tension: 0.0,
            gravity: Vec3::new(0.0, -9.81, 0.0),
            collision_radius: 0.07,
            restitution: 0.05,
        }
    }
}

pub struct Fluid {
    settings: FluidSettings,
    positions: Vec<Vec3>,
    velocities: Vec<Vec3>,
    densities: Vec<f64>,
    accelerations: Vec<Vec3>,
    neighbors: SpatialHash,
    pairs: Vec<(usize, usize)>,
    pressure_terms: Vec<f64>,
    color_gradient: Vec<Vec3>,
    color_laplacian: Vec<f64>,
    kernels: Kernels,
}

struct Kernels {
    h2: f64,
    poly6: f64,
    spiky: f64,
    viscosity: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct FluidProgress {
    pub substeps: usize,
    /// Time left after the substep budget or a representability limit.
    pub remaining: f64,
}

impl Fluid {
    pub fn new(settings: FluidSettings) -> Option<Self> {
        let h = settings.smoothing_radius;
        let kernels = Kernels {
            h2: h * h,
            poly6: 315.0 / (64.0 * PI * h.powi(9)),
            spiky: 45.0 / (PI * h.powi(6)),
            viscosity: 45.0 / (PI * h.powi(6)),
        };
        if ![
            settings.particle_mass,
            settings.rest_density,
            kernels.h2,
            kernels.poly6,
            kernels.spiky,
        ]
        .iter()
        .all(|&v| v.is_finite() && v > 0.0)
            || ![
                settings.pressure_stiffness,
                settings.viscosity,
                settings.surface_tension,
                settings.collision_radius,
            ]
            .iter()
            .all(|&v| v.is_finite() && v >= 0.0)
            || !settings.gravity.is_finite()
            || !(0.0..=1.0).contains(&settings.restitution)
        {
            return None;
        }
        Some(Self {
            settings,
            positions: Vec::new(),
            velocities: Vec::new(),
            densities: Vec::new(),
            accelerations: Vec::new(),
            neighbors: SpatialHash::new(h)?,
            pairs: Vec::new(),
            pressure_terms: Vec::new(),
            color_gradient: Vec::new(),
            color_laplacian: Vec::new(),
            kernels,
        })
    }

    pub fn positions(&self) -> &[Vec3] {
        &self.positions
    }
    pub fn velocities(&self) -> &[Vec3] {
        &self.velocities
    }

    /// Densities sampled at the start of the most recent step. Newly inserted
    /// particles carry the configured rest density until their first step.
    pub fn densities(&self) -> &[f64] {
        &self.densities
    }

    pub fn push(&mut self, position: Vec3, velocity: Vec3) -> bool {
        if !position.is_finite() || !velocity.is_finite() {
            return false;
        }
        self.positions.push(position);
        self.velocities.push(velocity);
        self.densities.push(self.settings.rest_density);
        true
    }

    /// An acoustic, viscous, and capillary step estimate. Strong acceleration
    /// or boundary impacts can require smaller steps.
    pub fn suggested_dt(&self) -> f64 {
        let s = self.settings;
        let speed = self
            .velocities
            .iter()
            .map(|v| v.length())
            .fold(0.0, f64::max);
        let acoustic = 0.2 * s.smoothing_radius / (speed + s.pressure_stiffness.sqrt()).max(1e-8);
        let viscous = 0.125 * s.rest_density * self.kernels.h2 / s.viscosity;
        let capillary =
            0.2 * (s.rest_density * s.smoothing_radius.powi(3) / s.surface_tension).sqrt();
        acoustic.min(viscous).min(capillary)
    }

    /// Advances up to the requested duration using the estimated step bound.
    /// Returns unprocessed time if the caller's substep budget is exhausted.
    pub fn advance(
        &mut self,
        duration: f64,
        budget: usize,
        boundary: Option<(&PhysicsWorld, QueryFilter)>,
    ) -> Option<FluidProgress> {
        if !duration.is_finite() || duration < 0.0 || budget == 0 {
            return None;
        }
        let mut progress = FluidProgress {
            substeps: 0,
            remaining: duration,
        };
        for _ in 0..budget {
            let dt = progress.remaining.min(self.suggested_dt());
            let next = progress.remaining - dt;
            if dt <= 0.0 || next == progress.remaining || !self.step(dt, boundary) {
                break;
            }
            progress.remaining = next;
            progress.substeps += 1;
        }
        Some(progress)
    }

    /// Advances SPH and exchanges normal impulses with dynamic boundaries.
    /// The caller advances the rigid world separately. Static and kinematic
    /// boundaries retain their prescribed motion.
    pub fn step_coupled(&mut self, dt: f64, world: &mut PhysicsWorld, filter: QueryFilter) -> bool {
        if !self.step(dt, None) {
            return false;
        }
        let targets = boundary_targets(world, filter);
        let s = self.settings;
        for (position, velocity) in self.positions.iter_mut().zip(&mut self.velocities) {
            coupled_boundary(
                position,
                velocity,
                s.collision_radius,
                s.restitution,
                s.particle_mass.recip(),
                world,
                &targets,
            );
        }
        true
    }

    pub fn step(&mut self, dt: f64, boundary: Option<(&PhysicsWorld, QueryFilter)>) -> bool {
        if !dt.is_finite() || dt <= 0.0 {
            return false;
        }
        let s = self.settings;
        let k = &self.kernels;
        self.neighbors.rebuild(&self.positions);
        self.pairs.clear();
        self.pairs.extend(self.neighbors.pairs(&self.positions));
        let density_scale = k.poly6 * s.particle_mass;
        self.densities.fill(density_scale * k.h2.powi(3));
        for &(i, j) in &self.pairs {
            let gap = k.h2 - self.positions[i].distance_squared(self.positions[j]);
            let density = density_scale * gap.powi(3);
            self.densities[i] += density;
            self.densities[j] += density;
        }
        self.pressure_terms.clear();
        self.pressure_terms.extend(
            self.densities
                .iter()
                .map(|&rho| s.pressure_stiffness * (rho - s.rest_density).max(0.0) / rho.powi(2)),
        );
        self.accelerations.clear();
        self.accelerations.resize(self.positions.len(), s.gravity);
        let visc_scale = s.viscosity * s.particle_mass * k.viscosity;
        for &(i, j) in &self.pairs {
            let delta = self.positions[i] - self.positions[j];
            let distance = delta.length();
            let gap = s.smoothing_radius - distance;
            let repulsion = if distance > 0.0 {
                (delta / distance)
                    * (s.particle_mass
                        * (self.pressure_terms[i] + self.pressure_terms[j])
                        * k.spiky
                        * gap
                        * gap)
            } else {
                Vec3::ZERO
            };
            let viscosity = (self.velocities[j] - self.velocities[i])
                * (visc_scale * gap / (self.densities[i] * self.densities[j]));
            let pair = repulsion + viscosity;
            self.accelerations[i] += pair;
            self.accelerations[j] -= pair;
        }
        if s.surface_tension > 0.0 {
            self.apply_surface_tension();
        }
        for ((p, v), a) in self
            .positions
            .iter_mut()
            .zip(&mut self.velocities)
            .zip(&self.accelerations)
        {
            *v += *a * dt;
            *p += *v * dt;
            if let Some((world, filter)) = boundary {
                collide_point(p, v, s.collision_radius, s.restitution, world, filter);
            }
        }
        true
    }
    fn apply_surface_tension(&mut self) {
        let s = self.settings;
        let k = &self.kernels;
        self.color_gradient.clear();
        self.color_gradient.resize(self.positions.len(), Vec3::ZERO);
        self.color_laplacian.clear();
        self.color_laplacian.extend(
            self.densities
                .iter()
                .map(|&rho| -18.0 * k.poly6 * k.h2 * k.h2 * s.particle_mass / rho),
        );
        for &(i, j) in &self.pairs {
            let delta = self.positions[i] - self.positions[j];
            let r2 = delta.length_squared();
            let gap = k.h2 - r2;
            let gradient = delta * (-6.0 * k.poly6 * gap * gap * s.particle_mass);
            let laplacian = -6.0 * k.poly6 * gap * (3.0 * k.h2 - 7.0 * r2) * s.particle_mass;
            self.color_gradient[i] += gradient / self.densities[j];
            self.color_gradient[j] -= gradient / self.densities[i];
            self.color_laplacian[i] += laplacian / self.densities[j];
            self.color_laplacian[j] += laplacian / self.densities[i];
        }
        let threshold = 0.1 / s.smoothing_radius;
        for (i, a) in self.accelerations.iter_mut().enumerate() {
            let gradient = self.color_gradient[i];
            let length = gradient.length();
            if length > threshold {
                *a -= gradient
                    * (s.surface_tension * self.color_laplacian[i] / (length * self.densities[i]));
            }
        }
    }
}
