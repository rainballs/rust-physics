//! Fixed and variable time stepping, profiling, and numerical validation.
use super::{
    BodyKind, NumericalStage, PhysicsWorld, ProfiledStep, StepError, StepStats, StepTimings, Vec3,
};
use crate::spatial::Aabb;
use std::time::{Duration, Instant};

impl PhysicsWorld {
    pub fn step_fixed(&mut self) -> Result<StepStats, StepError> {
        self.step(self.settings.fixed_dt)
    }

    pub fn step(&mut self, dt: f64) -> Result<StepStats, StepError> {
        Ok(self.step_inner::<false, false>(dt)?.stats)
    }

    /// Advances exactly as [Self::step] and records phase timings.
    pub fn step_profiled(&mut self, dt: f64) -> Result<ProfiledStep, StepError> {
        self.step_inner::<true, false>(dt)
    }

    pub fn step_fixed_profiled(&mut self) -> Result<ProfiledStep, StepError> {
        self.step_profiled(self.settings.fixed_dt)
    }

    /// Advances with numerical checks and rolls back on a detected failure.
    /// Capturing a checkpoint and validating each stage adds overhead.
    pub fn step_checked(&mut self, dt: f64) -> Result<StepStats, StepError> {
        let snapshot = self.snapshot();
        match self.step_inner::<false, true>(dt) {
            Ok(step) => Ok(step.stats),
            Err(error) => {
                if matches!(error, StepError::NumericalFailure(_)) {
                    self.restore(&snapshot);
                }
                Err(error)
            }
        }
    }

    fn step_inner<const PROFILE: bool, const CHECKED: bool>(
        &mut self,
        dt: f64,
    ) -> Result<ProfiledStep, StepError> {
        let h = dt / self.settings.substeps as f64;
        if !h.is_finite() || h <= 0.0 {
            return Err(StepError::InvalidTimestep);
        }
        let start = PROFILE.then(Instant::now);
        let mut stats = StepStats::default();
        let mut timings = StepTimings::default();
        self.events.clear();
        self.impacts.clear();
        self.joint_breaks.clear();
        for substep in 0..self.settings.substeps {
            self.substep::<PROFILE, CHECKED>(h, h * substep as f64, &mut stats, &mut timings)?;
        }
        self.impacts.sort_unstable_by(|a, b| {
            a.time
                .total_cmp(&b.time)
                .then(a.moving.cmp(&b.moving))
                .then(a.target.cmp(&b.target))
        });
        stats.ccd_impacts = self.impacts.len();
        stats.broken_joints = self.joint_breaks.len();
        for (_, body) in &mut self.bodies {
            body.force = Vec3::ZERO;
            body.torque = Vec3::ZERO;
            stats.sleeping_bodies += usize::from(body.asleep);
        }
        if let Some(start) = start {
            timings.total = start.elapsed();
        }
        Ok(ProfiledStep { stats, timings })
    }

    fn substep<const PROFILE: bool, const CHECKED: bool>(
        &mut self,
        h: f64,
        start: f64,
        stats: &mut StepStats,
        timings: &mut StepTimings,
    ) -> Result<(), StepError> {
        let mut mark = PROFILE.then(Instant::now);
        self.wake_joint_network();
        for (_, body) in &mut self.bodies {
            if body.kind == BodyKind::Dynamic && !body.asleep && body.enabled {
                body.velocity += (self.settings.gravity * body.gravity_scale * body.linear_axes
                    + body.apply_inverse_mass(body.force))
                    * h;
                body.angular_momentum += body.torque * body.angular_axes * h;
                body.velocity *= (-body.damping[0] * h).exp();
                body.angular_momentum *= (-body.damping[1] * h).exp();
            }
        }
        if CHECKED {
            self.check_state(NumericalStage::Forces)?;
        }
        self.prepare_joints(h);
        self.warm_start_joints(h);
        for _ in 0..self.settings.iterations {
            self.solve_joints();
        }
        if CHECKED {
            self.check_state(NumericalStage::Forces)?;
        }
        record_phase::<PROFILE>(&mut mark, &mut timings.forces_and_joints);
        stats.ccd_budget_fallbacks += self.advance_poses(h, start);
        record_phase::<PROFILE>(&mut mark, &mut timings.pose_and_ccd);
        if CHECKED {
            self.check_state(NumericalStage::Poses)?;
            if self.bodies.iter().any(|(_, b)| {
                b.enabled
                    && Aabb::of(b)
                        .is_some_and(|aabb| !aabb.min.is_finite() || !aabb.max.is_finite())
            }) || self
                .impacts
                .iter()
                .any(|hit| !hit.point.is_finite() || !hit.normal.is_normalized())
            {
                return Err(StepError::NumericalFailure(NumericalStage::Poses));
            }
        }
        self.collect_candidates();
        stats.candidate_pairs += self.candidates.len();
        record_phase::<PROFILE>(&mut mark, &mut timings.broad_phase);

        self.build_manifolds(stats);
        if CHECKED
            && self.debug_contacts.iter().any(|(_, _, contact)| {
                !contact.point.is_finite()
                    || !contact.normal.is_normalized()
                    || !contact.depth.is_finite()
                    || contact.depth < 0.0
            })
        {
            return Err(StepError::NumericalFailure(NumericalStage::Contacts));
        }
        record_phase::<PROFILE>(&mut mark, &mut timings.narrow_phase);
        if self.settings.sleep_time > 0.0 {
            self.prepare_islands();
        }
        for (_, body) in &mut self.bodies {
            body.geometry_dirty = false;
        }
        self.prepare_solver(h);
        stats.solved_parallel_islands += self.solve_rows();
        self.cache_joint_impulses();
        self.finish_contacts();
        self.break_joints(h);
        if CHECKED {
            self.check_state(NumericalStage::Solve)?;
        }
        record_phase::<PROFILE>(&mut mark, &mut timings.solver);
        if self.settings.sleep_time > 0.0 {
            self.update_sleep(h);
        }
        self.publish_events();
        self.previous_h = Some(h);
        record_phase::<PROFILE>(&mut mark, &mut timings.sleep_and_events);
        Ok(())
    }

    fn check_state(&self, stage: NumericalStage) -> Result<(), StepError> {
        if self.bodies.iter().any(|(_, body)| {
            !body.position.is_finite()
                || !body.rotation.is_normalized()
                || !body.velocity.is_finite()
                || !body.angular_momentum.is_finite()
                || !body.angular_velocity().is_finite()
        }) {
            Err(StepError::NumericalFailure(stage))
        } else {
            Ok(())
        }
    }
}

fn record_phase<const PROFILE: bool>(mark: &mut Option<Instant>, duration: &mut Duration) {
    if PROFILE {
        let now = Instant::now();
        if let Some(previous) = mark.replace(now) {
            *duration += now - previous;
        }
    }
}
