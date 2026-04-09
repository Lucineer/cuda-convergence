//! # cuda-convergence
//!
//! Detects when fleet deliberation reaches equilibrium.
//! Monitors confidence deltas, consensus ratios, and proposal states.
//!
//! ```rust
//! use cuda_convergence::{ConvergenceMonitor, ConvergenceEvent};
//! use cuda_equipment::Confidence;
//!
//! let mut monitor = ConvergenceMonitor::new(0.85, 5);
//! monitor.record_round(1, 0.6, 3, 1, 1);
//! monitor.record_round(2, 0.72, 3, 1, 1);
//! monitor.record_round(3, 0.88, 3, 1, 1);
//! assert!(monitor.has_converged());
//! ```

pub use cuda_equipment::{Confidence, VesselId};

use std::collections::VecDeque;

/// Why convergence was (or wasn't) reached.
#[derive(Debug, Clone, PartialEq)]
pub enum ConvergenceReason {
    ConfidenceThreshold { achieved: f64, required: f64 },
    ConsensusRatio { ratio: f64, required: f64 },
    MaxRounds { rounds: u32, max: u32 },
    ConfidenceStable { delta: f64, window: usize },
    Stall { rounds_no_change: u32 },
}

#[derive(Debug, Clone)]
pub struct ConvergenceEvent {
    pub round: u32,
    pub reason: ConvergenceReason,
    pub final_confidence: f64,
    pub consensus_ratio: f64,
}

/// Tracks convergence state across deliberation rounds.
#[derive(Debug, Clone, PartialEq)]
pub enum ConvergenceState {
    Diverging,
    Oscillating,
    Approaching,
    Converged,
    Stalled,
}

/// Snapshot of one deliberation round.
#[derive(Debug, Clone)]
pub struct RoundSnapshot {
    pub round: u32,
    pub avg_confidence: f64,
    pub confidence_delta: f64,
    pub consensus_ratio: f64,
    pub supports: usize,
    pub opposes: usize,
    pub pending: usize,
    pub total_agents: usize,
}

impl RoundSnapshot {
    pub fn agreement_fraction(&self) -> f64 {
        if self.total_agents == 0 { return 0.0; }
        self.supports as f64 / self.total_agents as f64
    }
}

/// Monitors fleet deliberation for convergence.
pub struct ConvergenceMonitor {
    threshold: f64,
    max_rounds: u32,
    stable_window: usize,
    history: VecDeque<RoundSnapshot>,
    state: ConvergenceState,
    rounds_without_change: u32,
    last_consensus: f64,
}

impl ConvergenceMonitor {
    pub fn new(threshold: f64, max_rounds: u32) -> Self {
        Self { threshold: threshold.clamp(0.0, 1.0), max_rounds,
            stable_window: 3, history: VecDeque::new(),
            state: ConvergenceState::Diverging,
            rounds_without_change: 0, last_consensus: 0.0 }
    }

    /// Record one deliberation round's outcome.
    pub fn record_round(&mut self, round: u32, avg_confidence: f64,
        supports: usize, opposes: usize, pending: usize) -> &ConvergenceState {
        let delta = self.history.back()
            .map(|prev| (avg_confidence - prev.avg_confidence).abs())
            .unwrap_or(1.0);

        let total = supports + opposes + pending;
        let consensus = if total > 0 { supports as f64 / total as f64 } else { 0.5 };

        let snap = RoundSnapshot { round, avg_confidence, confidence_delta: delta,
            consensus_ratio: consensus, supports, opposes, pending, total_agents: total };
        self.history.push_back(snap);

        // Track stall
        if (consensus - self.last_consensus).abs() < 0.01 {
            self.rounds_without_change += 1;
        } else {
            self.rounds_without_change = 0;
        }
        self.last_consensus = consensus;

        // Determine state
        self.state = if round >= self.max_rounds {
            ConvergenceState::Stalled
        } else if self.rounds_without_change >= 4 {
            ConvergenceState::Stalled
        } else if avg_confidence >= self.threshold {
            ConvergenceState::Converged
        } else if self.history.len() >= 2 {
            let recent: Vec<f64> = self.history.iter().rev().take(self.stable_window)
                .map(|s| s.avg_confidence).collect();
            let is_stable = recent.iter().zip(recent.iter().skip(1))
                .all(|(a, b)| (a - b).abs() < 0.02);
            if is_stable && recent[0] > self.threshold - 0.1 {
                ConvergenceState::Approaching
            } else if recent.windows(2).any(|w| {
                (w[0] > w[1] && w[1] > w[0]) || (w[0] < w[1] && w[1] < w[0])
            }) {
                ConvergenceState::Oscillating
            } else if delta > 0.05 {
                ConvergenceState::Diverging
            } else {
                ConvergenceState::Approaching
            }
        } else {
            ConvergenceState::Diverging
        };

        &self.state
    }

    pub fn has_converged(&self) -> bool { self.state == ConvergenceState::Converged }
    pub fn is_stalled(&self) -> bool { self.state == ConvergenceState::Stalled }
    pub fn state(&self) -> &ConvergenceState { &self.state }
    pub fn rounds(&self) -> usize { self.history.len() }
    pub fn last_snapshot(&self) -> Option<&RoundSnapshot> { self.history.back() }

    /// Generate convergence report.
    pub fn report(&self) -> Option<ConvergenceEvent> {
        let last = self.history.back()?;
        let reason = if self.state == ConvergenceState::Converged {
            if last.avg_confidence >= self.threshold {
                ConvergenceReason::ConfidenceThreshold { achieved: last.avg_confidence, required: self.threshold }
            } else {
                ConvergenceReason::ConsensusRatio { ratio: last.consensus_ratio, required: self.threshold }
            }
        } else if self.state == ConvergenceState::Stalled {
            if self.rounds_without_change >= 4 {
                ConvergenceReason::Stall { rounds_no_change: self.rounds_without_change }
            } else {
                ConvergenceReason::MaxRounds { rounds: last.round, max: self.max_rounds }
            }
        } else {
            return None;
        };
        Some(ConvergenceEvent { round: last.round, reason,
            final_confidence: last.avg_confidence, consensus_ratio: last.consensus_ratio })
    }

    /// Predict rounds to convergence based on trend.
    pub fn predict_rounds_remaining(&self) -> Option<f64> {
        if self.history.len() < 2 { return None; }
        let recent: Vec<f64> = self.history.iter().rev().take(3)
            .map(|s| s.avg_confidence).rev().collect();
        if recent.len() < 2 { return None; }
        let rate = (recent[recent.len()-1] - recent[0]) / recent.len() as f64;
        if rate <= 0.001 { return None; }
        let current = recent[recent.len()-1];
        let remaining = (self.threshold - current) / rate;
        if remaining > 0.0 { Some(remaining) } else { Some(0.0) }
    }

    pub fn avg_confidence_trend(&self) -> f64 {
        if self.history.len() < 2 { return 0.0; }
        let first = self.history.front().unwrap().avg_confidence;
        let last = self.history.back().unwrap().avg_confidence;
        last - first
    }
}

/// Equilibrium handler — processes convergence events.
pub struct EquilibriumHandler;

impl EquilibriumHandler {
    /// Determine what to do when convergence is reached.
    pub fn handle(event: &ConvergenceEvent) -> EquilibriumAction {
        match &event.reason {
            ConvergenceReason::ConfidenceThreshold { achieved, .. } => {
                if *achieved >= 0.95 {
                    EquilibriumAction::AcceptAndFinalize
                } else {
                    EquilibriumAction::AcceptWithMonitoring
                }
            }
            ConvergenceReason::ConsensusRatio { ratio, .. } => {
                if *ratio >= 0.9 {
                    EquilibriumAction::AcceptAndFinalize
                } else {
                    EquilibriumAction::AcceptWithReservation
                }
            }
            ConvergenceReason::MaxRounds { .. } => EquilibriumAction::ForceResolve,
            ConvergenceReason::Stall { .. } => EquilibriumAction::EscalateToHuman,
            ConvergenceReason::ConfidenceStable { delta, window } => {
                if *delta < 0.005 && *window >= 5 {
                    EquilibriumAction::AcceptAndFinalize
                } else {
                    EquilibriumAction::ContinueMonitoring
                }
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum EquilibriumAction {
    AcceptAndFinalize,
    AcceptWithMonitoring,
    AcceptWithReservation,
    ContinueMonitoring,
    ForceResolve,
    EscalateToHuman,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convergence_by_confidence() {
        let mut m = ConvergenceMonitor::new(0.85, 20);
        for r in 1..=6 {
            m.record_round(r, 0.7 + r as f64 * 0.03, 3, 0, 0);
        }
        assert!(m.has_converged());
    }

    #[test]
    fn test_stall_detection() {
        let mut m = ConvergenceMonitor::new(0.99, 20);
        for r in 1..=6 {
            m.record_round(r, 0.5, 2, 1, 0); // flat consensus
        }
        assert!(m.is_stalled());
    }

    #[test]
    fn test_max_rounds() {
        let mut m = ConvergenceMonitor::new(0.99, 3);
        for r in 1..=4 {
            m.record_round(r, 0.5, 1, 1, 0);
        }
        assert!(m.is_stalled());
    }

    #[test]
    fn test_prediction() {
        let mut m = ConvergenceMonitor::new(0.9, 20);
        m.record_round(1, 0.5, 2, 0, 0);
        m.record_round(2, 0.6, 2, 0, 0);
        m.record_round(3, 0.7, 2, 0, 0);
        let pred = m.predict_rounds_remaining();
        assert!(pred.is_some());
        assert!(pred.unwrap() > 0.0);
    }

    #[test]
    fn test_report() {
        let mut m = ConvergenceMonitor::new(0.8, 10);
        m.record_round(1, 0.85, 3, 0, 0);
        let report = m.report();
        assert!(report.is_some());
        assert!(matches!(report.unwrap().reason, ConvergenceReason::ConfidenceThreshold { .. }));
    }

    #[test]
    fn test_equilibrium_handler() {
        let event = ConvergenceEvent { round: 5,
            reason: ConvergenceReason::ConfidenceThreshold { achieved: 0.97, required: 0.85 },
            final_confidence: 0.97, consensus_ratio: 0.9 };
        assert_eq!(EquilibriumHandler::handle(&event), EquilibriumAction::AcceptAndFinalize);

        let stall = ConvergenceEvent { round: 8,
            reason: ConvergenceReason::Stall { rounds_no_change: 5 },
            final_confidence: 0.45, consensus_ratio: 0.5 };
        assert_eq!(EquilibriumHandler::handle(&stall), EquilibriumAction::EscalateToHuman);
    }

    #[test]
    fn test_snapshot_agreement() {
        let snap = RoundSnapshot { round: 1, avg_confidence: 0.8, confidence_delta: 0.1,
            consensus_ratio: 0.75, supports: 3, opposes: 1, pending: 0, total_agents: 4 };
        assert!((snap.agreement_fraction() - 0.75).abs() < 0.01);
    }

    #[test]
    fn test_trend() {
        let mut m = ConvergenceMonitor::new(0.9, 10);
        m.record_round(1, 0.3, 2, 0, 0);
        m.record_round(2, 0.7, 2, 0, 0);
        assert!(m.avg_confidence_trend() > 0.0);
    }
}
