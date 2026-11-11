//! Experimental position-based density projection.
use crate::grid::SpatialHash;
use crate::particles::project_boundaries;
use crate::{PhysicsWorld, QueryFilter, Vec3};
use std::f64::consts::PI;

#[derive(Clone, Copy, Debug)]
pub struct PbfSettings {
    pub smoothing_radius: f64,
    pub particle_mass: f64,
    pub rest_density: f64,
    pub gravity: Vec3,
    pub collision_radius: f64,
    pub iterations: usize,
    /// Positive regularization in the density multiplier denominator.
    pub relaxation: f64,
    /// Dimensionless coefficient for symmetric post-projection velocity smoothing.
    pub viscosity: f64,
}

impl Default for PbfSettings {
    fn default() -> Self {
        Self {
            smoothing_radius: 0.32,
            particle_mass: 4.096,
            rest_density: 1_000.0,
            gravity: Vec3::new(0.0, -9.81, 0.0),
            collision_radius: 0.07,
            iterations: 5,
            relaxation: 0.1,
            viscosity: 0.01,
        }
    }
}

pub struct PbfFluid {
    settings: PbfSettings,
    positions: Vec<Vec3>,
    previous: Vec<Vec3>,
    velocities: Vec<Vec3>,
    deltas: Vec<Vec3>,
    densities: Vec<f64>,
    lambdas: Vec<f64>,
    hash: SpatialHash,
    pairs: Vec<(usize, usize)>,
    gradients: Vec<Vec3>,
    squared_gradients: Vec<f64>,
    radius_squared: f64,
    poly6: f64,
}

impl PbfFluid {
    pub fn new(settings: PbfSettings) -> Option<Self> {
        let h = settings.smoothing_radius;
        let radius_squared = h * h;
        let poly6 = 315.0 / (64.0 * PI * h.powi(9));
        if ![
            radius_squared,
            poly6,
            settings.particle_mass,
            settings.rest_density,
            settings.relaxation,
        ]
        .iter()
        .all(|&v| v.is_finite() && v > 0.0)
            || !settings.gravity.is_finite()
            || settings.iterations == 0
            || !settings.collision_radius.is_finite()
            || settings.collision_radius < 0.0
            || !(0.0..=1.0).contains(&settings.viscosity)
        {
            return None;
        }
        Some(Self {
            settings,
            positions: Vec::new(),
            previous: Vec::new(),
            velocities: Vec::new(),
            deltas: Vec::new(),
            densities: Vec::new(),
            lambdas: Vec::new(),
            hash: SpatialHash::new(h)?,
            pairs: Vec::new(),
            gradients: Vec::new(),
            squared_gradients: Vec::new(),
            radius_squared,
            poly6,
        })
    }

    pub fn positions(&self) -> &[Vec3] {
        &self.positions
    }
    pub fn velocities(&self) -> &[Vec3] {
        &self.velocities
    }
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
        self.lambdas.push(0.0);
        self.deltas.push(Vec3::ZERO);
        self.gradients.push(Vec3::ZERO);
        self.squared_gradients.push(0.0);
        true
    }

    /// Predicts motion, projects compression and optional rigid boundaries,
    /// then reconstructs velocity. Rigid boundary coupling is one-way.
    pub fn step(&mut self, dt: f64, boundary: Option<(&PhysicsWorld, QueryFilter)>) -> bool {
        if !dt.is_finite() || dt <= 0.0 {
            return false;
        }
        self.previous.clear();
        self.previous.extend_from_slice(&self.positions);
        for (p, v) in self.positions.iter_mut().zip(&mut self.velocities) {
            *v += self.settings.gravity * dt;
            *p += *v * dt;
        }
        self.enforce_boundary(boundary);
        for _ in 0..self.settings.iterations {
            self.rebuild_pairs();
            self.compute_multipliers();
            self.compute_position_corrections();
            for (p, change) in self.positions.iter_mut().zip(&self.deltas) {
                *p += *change;
            }
            self.enforce_boundary(boundary);
        }
        for ((v, p), old) in self
            .velocities
            .iter_mut()
            .zip(&self.positions)
            .zip(&self.previous)
        {
            *v = (*p - *old) / dt;
        }
        self.rebuild_pairs();
        self.compute_multipliers();
        self.smooth_velocities();
        true
    }

    fn enforce_boundary(&mut self, boundary: Option<(&PhysicsWorld, QueryFilter)>) {
        if let Some((world, filter)) = boundary {
            for p in &mut self.positions {
                project_boundaries(
                    p,
                    self.settings.collision_radius,
                    world,
                    filter,
                    |_, _, _| {},
                );
            }
        }
    }

    fn rebuild_pairs(&mut self) {
        self.hash.rebuild(&self.positions);
        self.pairs.clear();
        self.pairs.extend(self.hash.pairs(&self.positions));
    }

    fn compute_multipliers(&mut self) {
        let s = self.settings;
        let density_scale = self.poly6 * s.particle_mass;
        let gradient_scale = -6.0 * density_scale / s.rest_density;
        self.densities
            .fill(density_scale * self.radius_squared.powi(3));
        self.gradients.fill(Vec3::ZERO);
        self.squared_gradients.fill(0.0);
        for &(i, j) in &self.pairs {
            let offset = self.positions[i] - self.positions[j];
            let gap = self.radius_squared - offset.length_squared();
            let density = density_scale * gap.powi(3);
            let gradient = offset * (gradient_scale * gap * gap);
            let squared = gradient.length_squared();
            self.densities[i] += density;
            self.densities[j] += density;
            self.gradients[i] += gradient;
            self.gradients[j] -= gradient;
            self.squared_gradients[i] += squared;
            self.squared_gradients[j] += squared;
        }
        for (i, lambda) in self.lambdas.iter_mut().enumerate() {
            let compression = (self.densities[i] / s.rest_density - 1.0).max(0.0);
            *lambda = -compression
                / (self.squared_gradients[i] + self.gradients[i].length_squared() + s.relaxation);
        }
    }

    fn compute_position_corrections(&mut self) {
        self.deltas.fill(Vec3::ZERO);
        let gradient_scale =
            -6.0 * self.poly6 * self.settings.particle_mass / self.settings.rest_density;
        for &(i, j) in &self.pairs {
            let offset = self.positions[i] - self.positions[j];
            let gap = self.radius_squared - offset.length_squared();
            let change =
                offset * ((self.lambdas[i] + self.lambdas[j]) * gradient_scale * gap * gap);
            self.deltas[i] += change;
            self.deltas[j] -= change;
        }
    }

    fn smooth_velocities(&mut self) {
        let coefficient = self.settings.viscosity * self.poly6 * self.settings.particle_mass;
        if coefficient == 0.0 {
            return;
        }
        self.deltas.fill(Vec3::ZERO);
        for &(i, j) in &self.pairs {
            let gap = self.radius_squared - self.positions[i].distance_squared(self.positions[j]);
            let weight = coefficient * gap.powi(3) / self.densities[i].max(self.densities[j]);
            let change = (self.velocities[j] - self.velocities[i]) * weight;
            self.deltas[i] += change;
            self.deltas[j] -= change;
        }
        for (v, change) in self.velocities.iter_mut().zip(&self.deltas) {
            *v += *change;
        }
    }
}
