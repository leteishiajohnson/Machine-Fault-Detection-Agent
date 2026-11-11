"""
tests/test_anomaly_detection.py — Unit tests for the anomaly detection engine.
"""

import pytest
import numpy as np

from src.anomaly_detection import (
    AnomalyDetector,
    AnomalyReport,
    SignalAnomaly,
    summarise_anomaly_report,
    IF_WARN_THRESHOLD,
)
from src.sensor_data import SensorSimulator


def make_detector():
    return AnomalyDetector(n_estimators=50, random_state=0)


class TestAnomalyDetector:

    def test_normal_scenario_not_flagged(self):
        sim = SensorSimulator(seed=42)
        readings = sim.generate("normal", n_readings=20)
        det = make_detector()
        report = det.analyze(readings)
        # Normal scenario — should not trigger critical flags
        critical = [f for f in report.flagged_signals if f.severity == "critical"]
        assert len(critical) == 0, (
            f"Normal scenario produced critical flags: "
            f"{[f.description for f in critical]}"
        )

    def test_bearing_failure_detected(self):
        sim = SensorSimulator(seed=42)
        readings = sim.generate("bearing_failure", n_readings=12)
        det = make_detector()
        report = det.analyze(readings)
        assert report.is_anomalous, "Bearing failure should be detected as anomalous"
        flagged_names = {f.signal for f in report.flagged_signals}
        assert "vibration_mms" in flagged_names, "Vibration should be flagged"

    def test_overheating_detected(self):
        sim = SensorSimulator(seed=42)
        readings = sim.generate("overheating", n_readings=12)
        det = make_detector()
        report = det.analyze(readings)
        assert report.is_anomalous
        flagged_names = {f.signal for f in report.flagged_signals}
        assert "temperature_c" in flagged_names

    def test_cavitation_detected(self):
        sim = SensorSimulator(seed=42)
        readings = sim.generate("cavitation", n_readings=12)
        det = make_detector()
        report = det.analyze(readings)
        assert report.is_anomalous
        flagged_names = {f.signal for f in report.flagged_signals}
        assert "pressure_bar" in flagged_names

    def test_electrical_overload_detected(self):
        sim = SensorSimulator(seed=42)
        readings = sim.generate("electrical_overload", n_readings=12)
        det = make_detector()
        report = det.analyze(readings)
        assert report.is_anomalous
        flagged_names = {f.signal for f in report.flagged_signals}
        assert "current_a" in flagged_names

    def test_rotor_imbalance_detected(self):
        sim = SensorSimulator(seed=42)
        readings = sim.generate("rotor_imbalance", n_readings=12)
        det = make_detector()
        report = det.analyze(readings)
        assert report.is_anomalous
        flagged_names = {f.signal for f in report.flagged_signals}
        assert "vibration_mms" in flagged_names

    def test_insufficient_readings_raises(self):
        sim = SensorSimulator(seed=42)
        readings = sim.generate("normal", n_readings=3)
        det = make_detector()
        with pytest.raises(ValueError, match="at least 4 readings"):
            det.analyze(readings)

    def test_anomaly_score_range(self):
        sim = SensorSimulator(seed=42)
        readings = sim.generate("bearing_failure", n_readings=12)
        det = make_detector()
        report = det.analyze(readings)
        assert 0.0 <= report.anomaly_score <= 1.0

    def test_report_machine_id_preserved(self):
        sim = SensorSimulator(seed=42)
        readings = sim.generate("normal", machine_id="TEST-PUMP", n_readings=10)
        det = make_detector()
        report = det.analyze(readings)
        assert report.machine_id == "TEST-PUMP"

    def test_flagged_signal_has_correct_direction(self):
        sim = SensorSimulator(seed=42)
        readings = sim.generate("overheating", n_readings=12)
        det = make_detector()
        report = det.analyze(readings)
        temp_flags = [f for f in report.flagged_signals if f.signal == "temperature_c"]
        assert len(temp_flags) > 0
        assert temp_flags[0].direction == "high"

    def test_summarise_report_returns_string(self):
        sim = SensorSimulator(seed=42)
        readings = sim.generate("bearing_failure", n_readings=12)
        det = make_detector()
        report = det.analyze(readings)
        summary = summarise_anomaly_report(report)
        assert isinstance(summary, str)
        assert "Machine" in summary
