//! Translational integration routines for systems with position-dependent acceleration.
use crate::Vec3;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct State {
    pub position: Vec3,
    pub velocity: Vec3,
}

#[derive(Clone, Copy, Debug)]
pub enum Method {
    Euler,
    SemiImplicitEuler,
    VelocityVerlet,
    RungeKutta4,
}

impl Method {
    /// Advances the state by one step. Acceleration is a function of position alone.
    pub fn advance(self, state: State, dt: f64, acceleration: impl Fn(Vec3) -> Vec3) -> State {
        let State {
            position: x,
            velocity: v,
        } = state;
        let initial = || acceleration(x);
        match self {
            Self::Euler => State {
                position: x + dt * v,
                velocity: v + dt * initial(),
            },
            Self::SemiImplicitEuler => {
                let velocity = v + dt * initial();
                State {
                    position: x + dt * velocity,
                    velocity,
                }
            }
            Self::VelocityVerlet => {
                let a = initial();
                let position = x + dt * (v + 0.5 * dt * a);
                State {
                    position,
                    velocity: v + 0.5 * dt * (a + acceleration(position)),
                }
            }
            Self::RungeKutta4 => rk4(state, 0.0, dt, |_, sample| acceleration(sample.position)),
        }
    }
}

/// Error controls for adaptive RK4 step doubling. The absolute tolerances
/// carry the units of position and velocity, respectively.
#[derive(Clone, Copy, Debug)]
pub struct AdaptiveSettings {
    pub position_tolerance: f64,
    pub velocity_tolerance: f64,
    pub relative_tolerance: f64,
    pub max_step: f64,
    pub max_attempts: usize,
}

impl Default for AdaptiveSettings {
    fn default() -> Self {
        Self {
            position_tolerance: 1e-8,
            velocity_tolerance: 1e-8,
            relative_tolerance: 1e-8,
            max_step: 0.1,
            max_attempts: 100_000,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdaptiveError {
    InvalidInput,
    NonFiniteState,
    StepUnderflow,
    AttemptLimit,
}

#[derive(Clone, Copy, Debug)]
pub struct AdaptiveResult {
    pub state: State,
    pub accepted: usize,
    pub rejected: usize,
}

/// Advances for an exact requested duration by adjusting the internal step
/// to keep estimated local error below the configured tolerances.
pub fn adaptive_rk4(
    mut state: State,
    duration: f64,
    settings: AdaptiveSettings,
    acceleration: impl Fn(Vec3) -> Vec3,
) -> Result<AdaptiveResult, AdaptiveError> {
    let s = settings;
    if ![s.position_tolerance, s.velocity_tolerance, s.max_step]
        .iter()
        .all(|&v| v.is_finite() && v > 0.0)
        || !s.relative_tolerance.is_finite()
        || s.relative_tolerance < 0.0
        || s.max_attempts == 0
        || !duration.is_finite()
        || duration < 0.0
        || !state.position.is_finite()
        || !state.velocity.is_finite()
    {
        return Err(AdaptiveError::InvalidInput);
    }

    let mut elapsed = 0.0;
    let mut h = duration.min(s.max_step);
    let mut accepted = 0;
    let mut rejected = 0;
    for _ in 0..s.max_attempts {
        if elapsed >= duration {
            break;
        }
        h = h.min(duration - elapsed);
        if elapsed + h == elapsed || h * 0.5 == 0.0 {
            return Err(AdaptiveError::StepUnderflow);
        }
        let coarse = Method::RungeKutta4.advance(state, h, &acceleration);
        let half = Method::RungeKutta4.advance(state, h * 0.5, &acceleration);
        let fine = Method::RungeKutta4.advance(half, h * 0.5, &acceleration);
        if ![coarse, fine]
            .into_iter()
            .all(|s| s.position.is_finite() && s.velocity.is_finite())
        {
            return Err(AdaptiveError::NonFiniteState);
        }
        let dx = (fine.position - coarse.position) / 15.0;
        let dv = (fine.velocity - coarse.velocity) / 15.0;
        let scaled_error = |delta: Vec3, before: Vec3, after: Vec3, absolute: f64| {
            let scale =
                Vec3::splat(absolute) + s.relative_tolerance * before.abs().max(after.abs());
            (delta.abs() / scale).max_element()
        };
        let error = scaled_error(dx, state.position, fine.position, s.position_tolerance).max(
            scaled_error(dv, state.velocity, fine.velocity, s.velocity_tolerance),
        );
        if error <= 1.0 {
            state = State {
                position: fine.position + dx,
                velocity: fine.velocity + dv,
            };
            if !state.position.is_finite() || !state.velocity.is_finite() {
                return Err(AdaptiveError::NonFiniteState);
            }
            elapsed += h;
            accepted += 1;
        } else {
            rejected += 1;
        }
        let factor = (0.9 * error.powf(-0.2)).clamp(0.2, 5.0);
        h = (h * factor).min(s.max_step);
    }
    if elapsed < duration {
        return Err(AdaptiveError::AttemptLimit);
    }
    Ok(AdaptiveResult {
        state,
        accepted,
        rejected,
    })
}

/// Two-position Verlet integrator for a fixed step and position-dependent force.
#[derive(Clone, Copy, Debug)]
pub struct PositionVerlet {
    previous: Vec3,
    current: Vec3,
    dt: f64,
}

impl PositionVerlet {
    /// Initializes the previous position using the local acceleration.
    pub fn new(state: State, dt: f64, acceleration: Vec3) -> Option<Self> {
        if !dt.is_finite()
            || dt <= 0.0
            || !state.position.is_finite()
            || !state.velocity.is_finite()
            || !acceleration.is_finite()
        {
            return None;
        }
        let previous = state.position - state.velocity * dt + acceleration * (0.5 * dt * dt);
        previous.is_finite().then_some(Self {
            previous,
            current: state.position,
            dt,
        })
    }

    pub fn position(&self) -> Vec3 {
        self.current
    }

    /// Advances one step and returns the centered velocity at the pose
    /// immediately before the advance.
    pub fn advance(&mut self, acceleration: impl Fn(Vec3) -> Vec3) -> Option<Vec3> {
        let next = self.current
            + (self.current - self.previous)
            + acceleration(self.current) * (self.dt * self.dt);
        let velocity = (next - self.previous) / (2.0 * self.dt);
        if !next.is_finite() || !velocity.is_finite() {
            return None;
        }
        self.previous = self.current;
        self.current = next;
        Some(velocity)
    }
}

/// Classical RK4 for time-, position-, and velocity-dependent acceleration.
pub fn rk4(state: State, time: f64, dt: f64, acceleration: impl Fn(f64, State) -> Vec3) -> State {
    let a1 = acceleration(time, state);
    let half = 0.5 * dt;
    let second = State {
        position: state.position + state.velocity * half,
        velocity: state.velocity + a1 * half,
    };
    let a2 = acceleration(time + half, second);
    let third = State {
        position: state.position + second.velocity * half,
        velocity: state.velocity + a2 * half,
    };
    let a3 = acceleration(time + half, third);
    let fourth = State {
        position: state.position + third.velocity * dt,
        velocity: state.velocity + a3 * dt,
    };
    let a4 = acceleration(time + dt, fourth);
    State {
        position: state.position
            + (state.velocity + (second.velocity + third.velocity) * 2.0 + fourth.velocity)
                * (dt / 6.0),
        velocity: state.velocity + (a1 + (a2 + a3) * 2.0 + a4) * (dt / 6.0),
    }
}
