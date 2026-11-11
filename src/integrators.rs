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
        let a = acceleration(x);
        match self {
            Self::Euler => State {
                position: x + dt * v,
                velocity: v + dt * a,
            },
            Self::SemiImplicitEuler => State {
                position: x + dt * (v + dt * a),
                velocity: v + dt * a,
            },
            Self::VelocityVerlet => {
                let position = x + dt * (v + 0.5 * dt * a);
                State {
                    position,
                    velocity: v + 0.5 * dt * (a + acceleration(position)),
                }
            }
            Self::RungeKutta4 => {
                let v2 = v + 0.5 * dt * a;
                let a2 = acceleration(x + 0.5 * dt * v);
                let v3 = v + 0.5 * dt * a2;
                let a3 = acceleration(x + 0.5 * dt * v2);
                let v4 = v + dt * a3;
                let a4 = acceleration(x + dt * v3);
                State {
                    position: x + (dt / 6.0) * (v + 2.0 * (v2 + v3) + v4),
                    velocity: v + (dt / 6.0) * (a + 2.0 * (a2 + a3) + a4),
                }
            }
        }
    }
}
