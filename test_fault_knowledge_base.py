"""
tests/test_fault_knowledge_base.py — Unit tests for the fault knowledge base
and signal investigator.
"""

import pytest

from src.anomaly_detection import AnomalyDetector
from src.fault_knowledge_base import (
    FAULT_KB,
    FAULT_KB_MAP,
    FaultCandidate,
    SignalInvestigator,
)
from src.sensor_data import SensorSimulator


def get_report(scenario: str, n: int = 12):
    sim = SensorSimulator(seed=42)
    readings = sim.generate(scenario, n_readings=n)
    det = AnomalyDetector(n_estimators=50, random_state=0)
    return det.analyze(readings)


class TestFaultKB:

    def test_all_fault_ids_unique(self):
        ids = [f.fault_id for f in FAULT_KB]
        assert len(ids) == len(set(ids))

    def test_normal_operation_exists(self):
        assert "NORMAL_OPERATION" in FAULT_KB_MAP

    def test_all_faults_have_actions(self):
        for fault in FAULT_KB:
            assert len(fault.recommended_actions) > 0, (
                f"{fault.fault_id} has no recommended actions"
            )

    def test_severity_values_valid(self):
        valid = {"low", "medium", "high", "critical"}
        for fault in FAULT_KB:
            assert fault.severity_level in valid, (
                f"{fault.fault_id} has invalid severity: {fault.severity_level}"
            )

    def test_urgency_values_valid(self):
        valid = {"monitor", "schedule", "immediate"}
        for fault in FAULT_KB:
            assert fault.urgency in valid, (
                f"{fault.fault_id} has invalid urgency: {fault.urgency}"
            )


class TestSignalInvestigator:

    def setup_method(self):
        self.investigator = SignalInvestigator()

    def test_normal_scenario_returns_normal_candidate(self):
        report = get_report("normal")
        # Normal scenario may be non-anomalous
        if not report.is_anomalous:
            candidates = self.investigator.investigate(report)
            assert candidates[0].fault.fault_id == "NORMAL_OPERATION"

    def test_bearing_failure_top_candidate(self):
        report = get_report("bearing_failure")
        assert report.is_anomalous
        candidates = self.investigator.investigate(report)
        top = candidates[0]
        assert top.fault.fault_id == "BEARING_FAILURE", (
            f"Expected BEARING_FAILURE, got {top.fault.fault_id} "
            f"(score: {top.match_score:.2f})"
        )

    def test_overheating_top_candidate(self):
        report = get_report("overheating")
        assert report.is_anomalous
        candidates = self.investigator.investigate(report)
        top = candidates[0]
        assert top.fault.fault_id == "OVERHEATING", (
            f"Expected OVERHEATING, got {top.fault.fault_id}"
        )

    def test_cavitation_top_candidate(self):
        report = get_report("cavitation")
        assert report.is_anomalous
        candidates = self.investigator.investigate(report)
        top = candidates[0]
        assert top.fault.fault_id == "CAVITATION", (
            f"Expected CAVITATION, got {top.fault.fault_id}"
        )

    def test_electrical_overload_top_candidate(self):
        report = get_report("electrical_overload")
        assert report.is_anomalous
        candidates = self.investigator.investigate(report)
        top = candidates[0]
        assert top.fault.fault_id == "ELECTRICAL_OVERLOAD", (
            f"Expected ELECTRICAL_OVERLOAD, got {top.fault.fault_id}"
        )

    def test_candidates_sorted_by_score(self):
        report = get_report("bearing_failure")
        candidates = self.investigator.investigate(report)
        scores = [c.match_score for c in candidates]
        assert scores == sorted(scores, reverse=True), (
            "Candidates not sorted by match score"
        )

    def test_match_score_in_range(self):
        report = get_report("bearing_failure")
        candidates = self.investigator.investigate(report)
        for c in candidates:
            assert 0.0 <= c.match_score <= 1.0, (
                f"Score out of range: {c.match_score} for {c.fault.fault_id}"
            )

    def test_evidence_summary_not_empty_for_anomaly(self):
        report = get_report("bearing_failure")
        candidates = self.investigator.investigate(report)
        top = candidates[0]
        assert top.evidence_summary  # should be non-empty
