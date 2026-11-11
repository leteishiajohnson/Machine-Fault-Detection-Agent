"""
anomaly_detection.py — ML-based anomaly detection engine.

Two complementary strategies are combined:

1. Statistical threshold checks
   Each signal is compared against hard-coded normal operating ranges.
   Violations are labelled with a severity level (warn / critical).

2. Isolation Forest (unsupervised ML)
   Trained on the baseline window (first N readings assumed normal),
   then scores every reading.  Anomaly score ∈ [0, 1]; higher = more anomalous.

The AnomalyDetector returns an AnomalyReport per reading window.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import IsolationForest

from .sensor_data import SensorReading, SIGNAL_NORMAL_RANGES


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class SignalAnomaly:
    """A single signal that has deviated from its normal range."""
    signal: str
    value: float
    normal_low: float
    normal_high: float
    z_score: float          # deviation in standard deviations from window mean
    severity: str           # "warn" | "critical"
    direction: str          # "high" | "low"
    description: str = ""


@dataclass
class AnomalyReport:
    """Full anomaly analysis for a window of sensor readings."""
    machine_id: str
    n_readings: int
    is_anomalous: bool
    anomaly_score: float            # 0-1, higher = more anomalous
    isolation_scores: List[float]   # per-reading IF scores
    flagged_signals: List[SignalAnomaly]
    baseline_stats: Dict[str, Dict[str, float]]   # mean/std per signal
    latest_reading: Dict[str, float]


# ---------------------------------------------------------------------------
# Severity thresholds
# ---------------------------------------------------------------------------

# Deviation ratios from normal range that trigger warn/critical
# A deviation_ratio of 1.0 means the value is exactly at the boundary.
WARN_RATIO    = 1.15   # 15% outside the normal range → warn
CRITICAL_RATIO = 1.40  # 40% outside → critical

# Isolation Forest anomaly score thresholds
IF_WARN_THRESHOLD     = 0.55
IF_CRITICAL_THRESHOLD = 0.72


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class AnomalyDetector:
    """
    Anomaly detector that combines Isolation Forest with per-signal
    threshold analysis.

    Usage:
        detector = AnomalyDetector()
        report = detector.analyze(readings)
    """

    def __init__(
        self,
        n_estimators: int = 100,
        contamination: float = 0.1,
        random_state: int = 42,
        baseline_fraction: float = 0.5,
    ):
        self._if_params = dict(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
        )
        self._baseline_fraction = baseline_fraction

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, readings: List[SensorReading]) -> AnomalyReport:
        """
        Analyze a list of sensor readings and return an AnomalyReport.
        Requires at least 4 readings; ideally 8-50.
        """
        if len(readings) < 4:
            raise ValueError(
                "AnomalyDetector requires at least 4 readings for analysis."
            )

        machine_id = readings[-1].machine_id
        signal_matrix = self._build_matrix(readings)
        baseline_stats = self._compute_baseline_stats(signal_matrix, readings)
        isolation_scores, anomaly_score = self._run_isolation_forest(signal_matrix)
        flagged = self._threshold_analysis(
            readings[-1].signals(), baseline_stats
        )

        # Overall anomaly: IF score above threshold OR any critical flag
        is_anomalous = (
            anomaly_score >= IF_WARN_THRESHOLD
            or any(f.severity == "critical" for f in flagged)
            or len(flagged) >= 2
        )

        return AnomalyReport(
            machine_id=machine_id,
            n_readings=len(readings),
            is_anomalous=is_anomalous,
            anomaly_score=round(anomaly_score, 4),
            isolation_scores=[round(s, 4) for s in isolation_scores],
            flagged_signals=flagged,
            baseline_stats=baseline_stats,
            latest_reading=readings[-1].signals(),
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _build_matrix(readings: List[SensorReading]) -> np.ndarray:
        """Convert readings list → 2-D numpy array (n_readings × n_signals)."""
        signal_names = list(readings[0].signals().keys())
        return np.array(
            [[r.signals()[s] for s in signal_names] for r in readings],
            dtype=float,
        )

    def _compute_baseline_stats(
        self,
        matrix: np.ndarray,
        readings: List[SensorReading],
    ) -> Dict[str, Dict[str, float]]:
        """Compute mean/std over the baseline (first fraction of readings)."""
        n_baseline = max(4, int(len(readings) * self._baseline_fraction))
        baseline = matrix[:n_baseline]
        signal_names = list(readings[0].signals().keys())
        stats: Dict[str, Dict[str, float]] = {}
        for i, name in enumerate(signal_names):
            col = baseline[:, i]
            stats[name] = {
                "mean": float(np.mean(col)),
                "std":  float(np.std(col)) or 1e-6,
                "min":  float(np.min(col)),
                "max":  float(np.max(col)),
            }
        return stats

    def _run_isolation_forest(
        self, matrix: np.ndarray
    ) -> Tuple[List[float], float]:
        """
        Fit and score using Isolation Forest.
        Returns (per_reading_scores, aggregate_score).
        Scores are normalised to [0, 1] where 1 = most anomalous.
        """
        clf = IsolationForest(**self._if_params)
        clf.fit(matrix)
        # decision_function: negative = anomalous; score_samples is raw
        raw_scores = clf.decision_function(matrix)
        # Normalise: invert and scale to [0, 1]
        # decision_function typically ∈ [-0.5, 0.5]
        normalised = np.clip(0.5 - raw_scores, 0.0, 1.0)
        # Aggregate score = mean of the last 40% of the window (most recent)
        n_recent = max(1, int(len(normalised) * 0.4))
        aggregate = float(np.mean(normalised[-n_recent:]))
        return list(normalised), aggregate

    def _threshold_analysis(
        self,
        latest: Dict[str, float],
        baseline_stats: Dict[str, Dict[str, float]],
    ) -> List[SignalAnomaly]:
        """Check the most recent reading against normal ranges and z-scores."""
        anomalies: List[SignalAnomaly] = []
        for signal, value in latest.items():
            if signal not in SIGNAL_NORMAL_RANGES:
                continue
            lo, hi = SIGNAL_NORMAL_RANGES[signal]
            stats = baseline_stats.get(signal, {"mean": value, "std": 1.0})
            z = (value - stats["mean"]) / (stats["std"] or 1e-6)
            range_span = hi - lo

            # Determine how far outside normal range (as ratio)
            if value > hi:
                excess = value - hi
                ratio = 1.0 + excess / max(range_span, 1e-6)
                direction = "high"
            elif value < lo:
                excess = lo - value
                ratio = 1.0 + excess / max(range_span, 1e-6)
                direction = "low"
            else:
                continue  # within normal range

            if ratio >= CRITICAL_RATIO or abs(z) >= 3.5:
                severity = "critical"
            elif ratio >= WARN_RATIO or abs(z) >= 2.0:
                severity = "warn"
            else:
                continue  # minor deviation, skip

            desc = (
                f"{signal} is {direction} at {value:.2f} "
                f"(normal: {lo}–{hi}, z={z:+.1f})"
            )
            anomalies.append(SignalAnomaly(
                signal=signal,
                value=round(value, 3),
                normal_low=lo,
                normal_high=hi,
                z_score=round(z, 2),
                severity=severity,
                direction=direction,
                description=desc,
            ))
        return anomalies


# ---------------------------------------------------------------------------
# Convenience: summarise a report as text
# ---------------------------------------------------------------------------

def summarise_anomaly_report(report: AnomalyReport) -> str:
    lines = [
        f"Machine:         {report.machine_id}",
        f"Readings analysed: {report.n_readings}",
        f"Anomaly detected:  {'YES ⚠' if report.is_anomalous else 'NO ✓'}",
        f"Anomaly score:   {report.anomaly_score:.3f}  "
        f"(IF threshold warn={IF_WARN_THRESHOLD}, critical={IF_CRITICAL_THRESHOLD})",
    ]
    if report.flagged_signals:
        lines.append("\nFlagged signals:")
        for fa in report.flagged_signals:
            tag = "🔴 CRITICAL" if fa.severity == "critical" else "🟡 WARN"
            lines.append(f"  {tag}  {fa.description}")
    else:
        lines.append("\nAll signals within normal operating ranges.")
    return "\n".join(lines)
