//! Bounded fixed-step accumulation for a variable-rate host loop.
#[derive(Clone, Copy, Debug)]
pub struct FixedClock {
    step: f64,
    max_steps: usize,
    remainder: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FramePlan {
    /// Fixed simulation steps the caller should perform for this frame.
    pub steps: usize,
    /// Fractional remainder, useful for interpolating rendered poses.
    pub interpolation: f64,
    /// Full steps dropped to enforce the configured catch-up limit, in seconds.
    pub dropped_time: f64,
}

impl FixedClock {
    pub fn new(step: f64, max_steps: usize) -> Option<Self> {
        (step.is_finite() && step > 0.0 && max_steps > 0).then_some(Self {
            step,
            max_steps,
            remainder: 0.0,
        })
    }

    pub fn step_size(&self) -> f64 {
        self.step
    }

    /// Discards the fractional remainder, for example on scene reset.
    pub fn reset(&mut self) {
        self.remainder = 0.0;
    }

    /// Accepts elapsed host time and plans a bounded number of simulation
    /// steps. Invalid elapsed time leaves the accumulator unchanged.
    pub fn plan(&mut self, elapsed: f64) -> Option<FramePlan> {
        let pending = self.remainder + elapsed;
        if elapsed < 0.0 || !pending.is_finite() {
            return None;
        }
        let available = (pending / self.step).floor();
        let steps = (available.min(self.max_steps as f64) as usize).min(self.max_steps);
        self.remainder = pending % self.step;
        // A quotient within roundoff of an exact integer can round up while
        // the remainder rounds to almost a full step. Consume that remainder
        // in the same frame to keep step count and accumulator consistent.
        if self.remainder > self.step * (1.0 - 4.0 * f64::EPSILON)
            && available.is_finite()
            && pending <= available * self.step
        {
            self.remainder = 0.0;
        }
        Some(FramePlan {
            steps,
            interpolation: self.remainder / self.step,
            dropped_time: (pending - self.remainder - steps as f64 * self.step).max(0.0),
        })
    }
}
