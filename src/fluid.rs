//! Experimental weakly compressible smoothed particle hydrodynamics (SPH).
use crate::grid::SpatialHash;
use crate::particles::collide_point;
use crate::{PhysicsWorld, QueryFilter, Vec3};
use std::f64::consts::PI;

#[derive(Clone, Copy, Debug)]
pub struct FluidSettings {
    pub smoothing_radius: f64,
    pub particle_mass: f64,
    pub rest_density: f64,
    /// Slope of the linear pressure equation of state (speed of sound squared).
    pub pressure_stiffness: f64,
    pub viscosity: f64,
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
    kernels: Kernels,
}

struct Kernels {
    h2: f64,
    poly6: f64,
    spiky: f64,
    viscosity: f64,
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

    /// A conservative acoustic CFL estimate. Strong acceleration, viscosity,
    /// or boundary impacts can require smaller steps.
    pub fn suggested_dt(&self) -> f64 {
        let max_speed = self
            .velocities
            .iter()
            .map(|v| v.length())
            .fold(0.0, f64::max);
        0.2 * self.settings.smoothing_radius
            / (max_speed + self.settings.pressure_stiffness.sqrt()).max(1e-8)
    }

    pub fn step(&mut self, dt: f64, boundary: Option<(&PhysicsWorld, QueryFilter)>) -> bool {
        if !dt.is_finite() || dt <= 0.0 {
            return false;
        }
        let s = self.settings;
        let k = &self.kernels;
        self.neighbors.rebuild(&self.positions);
        for (i, &position) in self.positions.iter().enumerate() {
            self.densities[i] = self
                .neighbors
                .neighbors(position)
                .map(|j| {
                    let distance_squared = position.distance_squared(self.positions[j]);
                    (k.h2 - distance_squared).max(0.0).powi(3)
                })
                .sum::<f64>()
                * k.poly6
                * s.particle_mass;
        }
        self.accelerations.clear();
        self.accelerations.resize(self.positions.len(), s.gravity);
        let pressure =
            |rho: f64| s.pressure_stiffness * (rho - s.rest_density).max(0.0) / rho.powi(2);
        for (i, &position) in self.positions.iter().enumerate() {
            let density_i = self.densities[i];
            let pressure_i = pressure(density_i);
            for j in self.neighbors.neighbors(position).filter(|&j| j > i) {
                let delta = position - self.positions[j];
                let r2 = delta.length_squared();
                if r2 >= k.h2 {
                    continue;
                }
                let r = r2.sqrt();
                let gap = s.smoothing_radius - r;
                let density_j = self.densities[j];
                let repulsion = if r > 0.0 {
                    (delta / r)
                        * (s.particle_mass
                            * (pressure_i + pressure(density_j))
                            * k.spiky
                            * gap
                            * gap)
                } else {
                    Vec3::ZERO
                };
                let viscosity = (self.velocities[j] - self.velocities[i])
                    * (s.viscosity * s.particle_mass * k.viscosity * gap / (density_i * density_j));
                let pair = repulsion + viscosity;
                self.accelerations[i] += pair;
                self.accelerations[j] -= pair;
            }
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
}
