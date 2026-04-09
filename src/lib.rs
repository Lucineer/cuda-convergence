//! Convergence Detection — equilibrium processing for deliberation
//! Determines when multi-agent deliberation has reached stability.

/// Convergence state
#[derive(Debug, Clone, PartialEq)]
pub enum ConvergenceState {
    Divergent,
    Oscillating,
    Converging,
    Converged,
    Stagnant,
}

/// A confidence measurement at a point in time
#[derive(Debug, Clone)]
pub struct ConfidenceSample {
    pub round: usize,
    pub value: f64,
    pub delta: f64,
    pub velocity: f64,
}

/// Convergence detector — analyzes confidence trajectories
pub struct ConvergenceDetector {
    threshold: f64,
    window_size: usize,
    stagnation_limit: usize,
    history: Vec<ConfidenceSample>,
    oscillation_buffer: Vec<f64>,
}

impl ConvergenceDetector {
    pub fn new(threshold: f64, window: usize, stagnation: usize) -> Self {
        Self {
            threshold, window_size: window, stagnation_limit: stagnation,
            history: vec![], oscillation_buffer: vec![],
        }
    }

    /// Feed a new confidence measurement
    pub fn observe(&mut self, round: usize, confidence: f64) -> ConvergenceState {
        let delta = if let Some(last) = self.history.last() {
            confidence - last.value
        } else {
            0.0
        };
        let velocity = if self.history.len() >= 2 {
            let prev_delta = self.history.last().map(|h| h.delta).unwrap_or(0.0);
            delta - prev_delta
        } else {
            delta
        };

        self.history.push(ConfidenceSample { round, value: confidence, delta, velocity });
        self.oscillation_buffer.push(delta);
        if self.oscillation_buffer.len() > self.window_size * 2 {
            self.oscillation_buffer.remove(0);
        }

        self.evaluate()
    }

    /// Evaluate current convergence state
    fn evaluate(&self) -> ConvergenceState {
        if self.history.len() < 3 {
            return ConvergenceState::Converging;
        }

        let recent: Vec<f64> = self.history.iter()
            .rev().take(self.window_size).map(|h| h.value).collect();

        let avg = recent.iter().sum::<f64>() / recent.len() as f64;
        let variance: f64 = recent.iter().map(|v| (v - avg).powi(2)).sum::<f64>()
            / recent.len() as f64;
        let std_dev = variance.sqrt();

        // Check convergence
        if avg >= self.threshold && std_dev < 0.05 {
            return ConvergenceState::Converged;
        }

        // Check stagnation — no progress for many rounds
        let recent_deltas: Vec<f64> = self.history.iter()
            .rev().take(self.stagnation_limit).map(|h| h.delta.abs()).collect();
        if recent_deltas.len() >= self.stagnation_limit {
            let avg_delta = recent_deltas.iter().sum::<f64>() / recent_deltas.len() as f64;
            if avg_delta < 0.01 {
                return ConvergenceState::Stagnant;
            }
        }

        // Check oscillation — deltas alternate sign
        let sign_changes = self.oscillation_buffer.windows(2)
            .filter(|w| w[0] * w[1] < 0.0).count();
        let oscillation_rate = sign_changes as f64 / self.oscillation_buffer.len().max(1) as f64;
        if oscillation_rate > 0.6 && std_dev > 0.1 {
            return ConvergenceState::Oscillating;
        }

        // Check divergence
        let last = self.history.last().unwrap();
        if last.velocity < -0.1 {
            return ConvergenceState::Divergent;
        }

        ConvergenceState::Converging
    }

    /// Get the number of rounds observed
    pub fn rounds(&self) -> usize { self.history.len() }

    /// Get current confidence
    pub fn current_confidence(&self) -> f64 {
        self.history.last().map(|h| h.value).unwrap_or(0.0)
    }

    /// Predict rounds until convergence (linear extrapolation)
    pub fn predict_convergence_round(&self) -> Option<usize> {
        if self.history.len() < 3 { return None; }
        let rate = self.history.last()?.delta;
        if rate.abs() < 0.001 { return None; }
        let current = self.history.last()?.value;
        let remaining = (self.threshold - current) / rate;
        if remaining <= 0.0 { return None; }
        Some(self.history.len() + remaining.ceil() as usize)
    }
}

/// Equilibrium processor — manages the feed-in loop
pub struct EquilibriumProcessor {
    detector: ConvergenceDetector,
    feed_signals: Vec<FeedSignal>,
}

#[derive(Debug, Clone)]
pub struct FeedSignal {
    pub source: String,
    pub confidence_adjustment: f64,
    pub constraint_violation: Option<String>,
}

impl EquilibriumProcessor {
    pub fn new(threshold: f64) -> Self {
        Self {
            detector: ConvergenceDetector::new(threshold, 5, 8),
            feed_signals: vec![],
        }
    }

    /// Process a feed-in signal from user interaction or runtime feedback
    pub fn feed(&mut self, signal: FeedSignal) {
        self.feed_signals.push(signal);
    }

    /// Run one deliberation cycle with feed-in
    pub fn cycle(&mut self, round: usize, base_confidence: f64) -> ConvergenceState {
        let mut adjusted = base_confidence;
        for signal in &self.feed_signals {
            adjusted += signal.confidence_adjustment;
        }
        adjusted = adjusted.clamp(0.0, 0.99);
        self.detector.observe(round, adjusted)
    }

    pub fn state(&self) -> &ConvergenceDetector { &self.detector }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_converges_quickly() {
        let mut det = ConvergenceDetector::new(0.85, 3, 5);
        assert_eq!(det.observe(1, 0.5), ConvergenceState::Converging);
        assert_eq!(det.observe(2, 0.7), ConvergenceState::Converging);
        assert_eq!(det.observe(3, 0.85), ConvergenceState::Converged);
    }

    #[test]
    fn test_detects_stagnation() {
        let mut det = ConvergenceDetector::new(0.85, 3, 3);
        det.observe(1, 0.3);
        det.observe(2, 0.301);
        det.observe(3, 0.301);
        assert_eq!(det.evaluate(), ConvergenceState::Stagnant);
    }

    #[test]
    fn test_oscillation_detection() {
        let mut det = ConvergenceDetector::new(0.85, 3, 10);
        for i in 0..8 {
            let val = if i % 2 == 0 { 0.5 } else { 0.7 };
            det.observe(i, val);
        }
        assert_eq!(det.evaluate(), ConvergenceState::Oscillating);
    }

    #[test]
    fn test_prediction() {
        let mut det = ConvergenceDetector::new(0.85, 3, 5);
        det.observe(1, 0.5);
        det.observe(2, 0.6);
        let pred = det.predict_convergence_round();
        assert!(pred.is_some());
    }

    #[test]
    fn test_equilibrium_feed() {
        let mut eq = EquilibriumProcessor::new(0.85);
        eq.feed(FeedSignal { source: "user_click".to_string(), confidence_adjustment: 0.05, constraint_violation: None });
        let state = eq.cycle(1, 0.8);
        assert_eq!(state, ConvergenceState::Converged);
    }
}
