//! Velocity-level bilateral and unilateral constraints between rigid bodies.
use crate::world::{effective_mass, inverse_effective_mass, relative_velocity};
use crate::{BodyHandle, Quat, RigidBody, Vec3};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct JointHandle(pub(crate) BodyHandle);

/// Joint anchors are specified in each body's local space. Axis vectors and
/// target rotations are also local to the associated body.
#[derive(Clone, Copy, Debug)]
pub struct Joint {
    pub bodies: [BodyHandle; 2],
    pub anchors: [Vec3; 2],
    pub kind: JointKind,
    /// Fraction of position error corrected per substep for rigid rows.
    pub stabilization: f64,
    pub break_limit: Option<JointBreakLimit>,
}

/// Maximum solved linear force and intrinsic angular torque before a joint breaks.
#[derive(Clone, Copy, Debug)]
pub struct JointBreakLimit {
    pub force: Option<f64>,
    pub torque: Option<f64>,
}

#[derive(Clone, Copy, Debug)]
pub enum JointKind {
    Ball,
    Distance {
        length: f64,
    },
    Rope {
        max_length: f64,
    },
    Spring {
        length: f64,
        stiffness: f64,
        damping: f64,
    },
    /// Preserves the specified orientation of body B relative to A.
    Fixed {
        rotation: Quat,
    },
    /// Aligns the local axes while permitting twist about the hinge.
    Revolute {
        axes: [Vec3; 2],
        motor: Option<Motor>,
        limits: Option<HingeLimits>,
    },
    /// Allows translation along the local axis of A and fixes relative rotation.
    Prismatic {
        axis: Vec3,
        rotation: Quat,
        limits: Option<[f64; 2]>,
        motor: Option<Motor>,
    },
}

/// Angular limits in radians, measured from A's local reference tangent
/// to B's tangent about the common hinge axis. Bounds cannot cross ±π.
#[derive(Clone, Copy, Debug)]
pub struct HingeLimits {
    pub tangents: [Vec3; 2],
    pub angles: [f64; 2],
}

impl HingeLimits {
    fn valid(self, axes: [Vec3; 2]) -> bool {
        let [low, high] = self.angles;
        low > -std::f64::consts::PI
            && high < std::f64::consts::PI
            && low <= high
            && self
                .tangents
                .into_iter()
                .zip(axes)
                .all(|(t, axis)| t.is_normalized() && t.dot(axis).abs() < 1e-8)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Motor {
    pub speed: f64,
    /// Maximum force for a slider, or torque for a hinge.
    pub max_effort: f64,
}

impl Motor {
    fn valid(self) -> bool {
        self.speed.is_finite() && self.max_effort.is_finite() && self.max_effort >= 0.0
    }
}

impl Joint {
    pub fn new(bodies: [BodyHandle; 2], anchors: [Vec3; 2], kind: JointKind) -> Self {
        Self {
            bodies,
            anchors,
            kind,
            stabilization: 0.2,
            break_limit: None,
        }
    }

    pub fn breakable(mut self, limit: JointBreakLimit) -> Self {
        self.break_limit = Some(limit);
        self
    }

    pub(crate) fn requires_awake(self) -> bool {
        match self.kind {
            JointKind::Revolute { motor, .. } | JointKind::Prismatic { motor, .. } => {
                motor.is_some_and(|motor| motor.speed != 0.0 && motor.max_effort > 0.0)
            }
            JointKind::Spring { .. } => true,
            _ => false,
        }
    }

    pub(crate) fn valid(self) -> bool {
        let nonnegative = |v: f64| v.is_finite() && v >= 0.0;
        let unit_axis = |v: Vec3| (v.length_squared() - 1.0).abs() < 1e-8;
        let unit_rotation = |q: Quat| (q.length_squared() - 1.0).abs() < 1e-8;
        self.bodies[0] != self.bodies[1]
            && self.anchors.iter().all(|v| v.is_finite())
            && (0.0..=1.0).contains(&self.stabilization)
            && self.break_limit.is_none_or(|limit| {
                limit.force.is_none_or(nonnegative) && limit.torque.is_none_or(nonnegative)
            })
            && match self.kind {
                JointKind::Ball => true,
                JointKind::Distance { length } => nonnegative(length),
                JointKind::Rope { max_length } => nonnegative(max_length),
                JointKind::Spring {
                    length,
                    stiffness,
                    damping,
                } => {
                    nonnegative(length)
                        && stiffness.is_finite()
                        && stiffness > 0.0
                        && nonnegative(damping)
                }
                JointKind::Fixed { rotation } => unit_rotation(rotation),
                JointKind::Revolute {
                    axes,
                    motor,
                    limits,
                } => {
                    axes.into_iter().all(unit_axis)
                        && motor.is_none_or(Motor::valid)
                        && limits.is_none_or(|limits| limits.valid(axes))
                }
                JointKind::Prismatic {
                    axis,
                    rotation,
                    limits,
                    motor,
                } => {
                    unit_axis(axis)
                        && unit_rotation(rotation)
                        && motor.is_none_or(Motor::valid)
                        && limits
                            .is_none_or(|[lo, hi]| lo.is_finite() && hi.is_finite() && lo <= hi)
                }
            }
    }
}

#[derive(Clone, Copy)]
enum Jacobian {
    Linear([Vec3; 2]),
    Angular,
}

#[cfg_attr(feature = "rayon", derive(Clone))]
pub(crate) struct JointRow {
    axis: Vec3,
    jacobian: Jacobian,
    inverse_mass: f64,
    bias: f64,
    softness: f64,
    impulse: f64,
    bounds: [f64; 2],
}

#[derive(Clone, Copy)]
pub(crate) struct JointImpulse {
    axis: Vec3,
    angular: bool,
    impulse: f64,
}

impl JointImpulse {
    pub(crate) fn reaction(self) -> (Vec3, Vec3) {
        let impulse = self.axis * self.impulse;
        if self.angular {
            (Vec3::ZERO, impulse)
        } else {
            (impulse, Vec3::ZERO)
        }
    }
}

const FREE: [f64; 2] = [f64::NEG_INFINITY, f64::INFINITY];

struct Rows<'a> {
    bodies: [&'a RigidBody; 2],
    out: &'a mut Vec<JointRow>,
    h: f64,
    stabilization: f64,
}

impl Rows<'_> {
    fn push(&mut self, axis: Vec3, jacobian: Jacobian, bias: f64, softness: f64, bounds: [f64; 2]) {
        let [a, b] = self.bodies;
        let k = match jacobian {
            Jacobian::Linear(r) => inverse_effective_mass(a, b, r, axis),
            Jacobian::Angular => [a, b]
                .into_iter()
                .filter(|b| effective_mass(b) > 0.0)
                .map(|b| axis.dot(b.apply_inverse_inertia(axis)))
                .sum(),
        };
        if k > 0.0 {
            self.out.push(JointRow {
                axis,
                jacobian,
                inverse_mass: (k + softness).recip(),
                bias,
                softness,
                impulse: 0.0,
                bounds,
            });
        }
    }

    fn rigid(&mut self, axis: Vec3, jacobian: Jacobian, error: f64) {
        self.push(
            axis,
            jacobian,
            (error * self.stabilization / self.h).clamp(-10.0, 10.0),
            0.0,
            FREE,
        );
    }

    fn limit(&mut self, axis: Vec3, jacobian: Jacobian, error: f64) {
        // Inside the limit, allow approach at up to the speed that consumes
        // the remaining slack in one substep.
        let factor = if error > 0.0 { self.stabilization } else { 1.0 };
        self.push(
            axis,
            jacobian,
            (error * factor / self.h).min(10.0),
            0.0,
            [f64::NEG_INFINITY, 0.0],
        );
    }

    fn motor(&mut self, axis: Vec3, jacobian: Jacobian, motor: Motor) {
        let max = motor.max_effort * self.h;
        self.push(axis, jacobian, -motor.speed, 0.0, [-max, max]);
    }

    fn fixed_rotation(&mut self, target: Quat) {
        let [a, b] = self.bodies;
        let error = b.rotation * (a.rotation * target).conjugate();
        // Choose the quaternion hemisphere representing the shortest rotation.
        let error = (if error.w < 0.0 { -error } else { error }).to_scaled_axis();
        for axis in Vec3::AXES {
            self.rigid(axis, Jacobian::Angular, error.dot(axis));
        }
    }
}

pub(crate) fn prepare(joint: Joint, a: &RigidBody, b: &RigidBody, h: f64, out: &mut Vec<JointRow>) {
    let offsets = [a.rotation * joint.anchors[0], b.rotation * joint.anchors[1]];
    let delta = (b.position - a.position) + offsets[1] - offsets[0];
    let linear = Jacobian::Linear(offsets);
    let mut rows = Rows {
        bodies: [a, b],
        out,
        h,
        stabilization: joint.stabilization,
    };

    match joint.kind {
        JointKind::Ball
        | JointKind::Fixed { .. }
        | JointKind::Revolute { .. }
        | JointKind::Distance { length: 0.0 } => {
            for axis in Vec3::AXES {
                rows.rigid(axis, linear, delta.dot(axis));
            }
            match joint.kind {
                JointKind::Fixed { rotation } => rows.fixed_rotation(rotation),
                JointKind::Revolute {
                    axes,
                    motor,
                    limits,
                } => {
                    let first = a.rotation * axes[0];
                    let second = b.rotation * axes[1];
                    let (u, v) = first.any_orthonormal_pair();
                    let cross = first.cross(second);
                    let angle = cross.length().atan2(first.dot(second));
                    let correction = cross.try_normalize().unwrap_or(u) * angle;
                    for axis in [u, v] {
                        rows.rigid(axis, Jacobian::Angular, correction.dot(axis));
                    }
                    if let Some(motor) = motor {
                        rows.motor(first, Jacobian::Angular, motor);
                    }
                    if let Some(limits) = limits {
                        let ta = a.rotation * limits.tangents[0];
                        let tb = b.rotation * limits.tangents[1];
                        let correction = if angle > 1e-10 {
                            Quat::from_scaled_axis(-correction)
                        } else {
                            Quat::IDENTITY
                        };
                        let aligned = correction * tb;
                        let twist = first.dot(ta.cross(aligned)).atan2(ta.dot(aligned));
                        let [low, high] = limits.angles;
                        if low == high {
                            rows.rigid(first, Jacobian::Angular, twist - low);
                        } else {
                            rows.limit(first, Jacobian::Angular, twist - high);
                            rows.limit(-first, Jacobian::Angular, low - twist);
                        }
                    }
                }
                _ => {}
            }
        }
        JointKind::Distance { length }
        | JointKind::Rope { max_length: length }
        | JointKind::Spring { length, .. } => {
            let axis = delta.try_normalize().unwrap_or(a.rotation * Vec3::X);
            let error = delta.length() - length;
            match joint.kind {
                JointKind::Rope { .. } => rows.limit(axis, linear, error),
                JointKind::Spring {
                    stiffness, damping, ..
                } => {
                    let softness = (h * (damping + h * stiffness)).recip();
                    if softness.is_finite() {
                        rows.push(
                            axis,
                            linear,
                            error * stiffness / (damping + h * stiffness),
                            softness,
                            FREE,
                        );
                    }
                }
                _ => rows.rigid(axis, linear, error),
            }
        }
        JointKind::Prismatic {
            axis,
            rotation,
            limits,
            motor,
        } => {
            let axis = a.rotation * axis;
            let (u, v) = axis.any_orthonormal_pair();
            // The rail frame rotates with A. The effective A lever arm
            // includes the current rail displacement to differentiate that frame.
            let rail = Jacobian::Linear([offsets[0] + delta, offsets[1]]);
            for transverse in [u, v] {
                rows.rigid(transverse, rail, delta.dot(transverse));
            }
            rows.fixed_rotation(rotation);
            if let Some(motor) = motor {
                rows.motor(axis, rail, motor);
            }
            if let Some([low, high]) = limits {
                let distance = delta.dot(axis);
                if low == high {
                    rows.rigid(axis, rail, distance - low);
                } else {
                    rows.limit(axis, rail, distance - high);
                    rows.limit(-axis, rail, low - distance);
                }
            }
        }
    }
}

impl JointRow {
    pub(crate) fn reaction(&self) -> (Vec3, Vec3) {
        self.cache().reaction()
    }

    pub(crate) fn cache(&self) -> JointImpulse {
        JointImpulse {
            axis: self.axis,
            angular: matches!(self.jacobian, Jacobian::Angular),
            impulse: self.impulse,
        }
    }

    pub(crate) fn warm_start(
        &mut self,
        previous: JointImpulse,
        ratio: f64,
        a: &mut RigidBody,
        b: &mut RigidBody,
    ) {
        let agreement = self.axis.dot(previous.axis);
        if previous.angular == matches!(self.jacobian, Jacobian::Angular) && agreement > 0.98 {
            self.impulse =
                (previous.impulse * ratio * agreement).clamp(self.bounds[0], self.bounds[1]);
            self.apply(a, b, self.impulse);
        }
    }

    pub(crate) fn solve(&mut self, a: &mut RigidBody, b: &mut RigidBody) {
        let speed = match self.jacobian {
            Jacobian::Linear(r) => relative_velocity(a, b, r),
            Jacobian::Angular => b.angular_velocity() - a.angular_velocity(),
        }
        .dot(self.axis);
        let old = self.impulse;
        self.impulse = (old - (speed + self.bias + self.softness * old) * self.inverse_mass)
            .clamp(self.bounds[0], self.bounds[1]);
        self.apply(a, b, self.impulse - old);
    }

    fn apply(&self, a: &mut RigidBody, b: &mut RigidBody, amount: f64) {
        if amount == 0.0 {
            return;
        }
        let impulse = self.axis * amount;
        match self.jacobian {
            Jacobian::Linear([ra, rb]) => {
                a.solver_impulse(-impulse, ra);
                b.solver_impulse(impulse, rb);
            }
            Jacobian::Angular => {
                if effective_mass(a) > 0.0 {
                    a.angular_momentum -= impulse * a.angular_axes;
                }
                if effective_mass(b) > 0.0 {
                    b.angular_momentum += impulse * b.angular_axes;
                }
            }
        }
    }
}
