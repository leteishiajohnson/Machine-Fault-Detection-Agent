"""
tests/test_agent.py — Integration tests for the DiagnosisAgent agentic pipeline.

These tests use the rule-based reasoner (no API key required) and verify
the full 7-step pipeline executes correctly for every scenario.
"""

import pytest

from src.agent import DiagnosisAgent, AgentRun
from src.sensor_data import SensorSimulator


def make_agent() -> DiagnosisAgent:
    """Create a rule-based agent (no LLM calls)."""
    return DiagnosisAgent(use_llm=False)


def run_agent_on(scenario: str, n: int = 12) -> AgentRun:
    sim = SensorSimulator(seed=42)
    readings = sim.generate(scenario, machine_id="TEST-01", n_readings=n)
    agent = make_agent()
    return agent.run(readings)


class TestAgentPipeline:

    FAULT_SCENARIOS = [
        ("bearing_failure",    "BEARING_FAILURE"),
        ("overheating",        "OVERHEATING"),
        ("cavitation",         "CAVITATION"),
        ("electrical_overload","ELECTRICAL_OVERLOAD"),
        ("rotor_imbalance",    "ROTOR_IMBALANCE"),
    ]

    def test_all_7_steps_run(self):
        run = run_agent_on("bearing_failure")
        step_names = [s.step_name for s in run.steps]
        expected = ["INGEST", "DETECT", "INVESTIGATE", "REASON", "DIAGNOSE", "EXPLAIN", "RECOMMEND"]
        assert step_names == expected

    def test_normal_scenario_succeeds(self):
        run = run_agent_on("normal")
        assert run.succeeded, f"Steps: {[(s.step_name, s.status) for s in run.steps]}"

    @pytest.mark.parametrize("scenario,expected_fault", FAULT_SCENARIOS)
    def test_fault_scenarios_produce_diagnosis(self, scenario, expected_fault):
        run = run_agent_on(scenario)
        assert run.diagnosis is not None, f"No diagnosis for {scenario}"
        assert run.diagnosis.fault_id == expected_fault, (
            f"Scenario '{scenario}': expected {expected_fault}, "
            f"got {run.diagnosis.fault_id}"
        )

    def test_insufficient_readings_returns_error(self):
        sim = SensorSimulator(seed=42)
        readings = sim.generate("normal", n_readings=2)
        agent = make_agent()
        run = agent.run(readings)
        # INGEST step should be error
        ingest_step = run.steps[0]
        assert ingest_step.status == "error"
        assert "Insufficient" in ingest_step.summary

    def test_empty_readings_returns_error(self):
        agent = make_agent()
        run = agent.run([])
        assert run.steps[0].status == "error"

    def test_run_has_machine_id(self):
        run = run_agent_on("overheating")
        assert run.machine_id == "TEST-01"

    def test_run_has_timestamps(self):
        run = run_agent_on("overheating")
        assert run.started_at is not None
        assert run.finished_at is not None
        assert run.finished_at >= run.started_at

    def test_run_duration_positive(self):
        run = run_agent_on("overheating")
        assert run.duration_ms >= 0

    def test_diagnosis_has_required_fields(self):
        run = run_agent_on("bearing_failure")
        dx = run.diagnosis
        assert dx is not None
        assert dx.fault_id
        assert dx.fault_name
        assert dx.confidence in ("high", "medium", "low", "uncertain")
        assert dx.severity in ("low", "medium", "high", "critical")
        assert dx.urgency in ("monitor", "schedule", "immediate")
        assert isinstance(dx.recommended_actions, list)
        assert len(dx.recommended_actions) > 0
        assert isinstance(dx.evidence, list)

    def test_fault_candidates_populated(self):
        run = run_agent_on("bearing_failure")
        assert len(run.fault_candidates) > 0
        assert run.fault_candidates[0].fault.fault_id == "BEARING_FAILURE"

    def test_anomaly_report_populated(self):
        run = run_agent_on("overheating")
        assert run.anomaly_report is not None
        assert run.anomaly_report.is_anomalous

    def test_on_step_callback_called(self):
        called_steps = []
        agent = DiagnosisAgent(use_llm=False, on_step=lambda s: called_steps.append(s.step_name))
        sim = SensorSimulator(seed=42)
        readings = sim.generate("cavitation", n_readings=12)
        agent.run(readings)
        assert len(called_steps) == 7

    def test_to_dict_serialisable(self):
        import json
        run = run_agent_on("electrical_overload")
        dx = run.diagnosis
        assert dx is not None
        d = dx.to_dict()
        # Must be JSON serialisable
        serialised = json.dumps(d, default=str)
        assert isinstance(serialised, str)

    def test_rule_based_normal_confidence_high(self):
        """For normal scenario, rule-based should give high confidence."""
        sim = SensorSimulator(seed=42)
        readings = sim.generate("normal", n_readings=20)
        agent = make_agent()
        run = agent.run(readings)
        if run.diagnosis and run.diagnosis.fault_id == "NORMAL_OPERATION":
            assert run.diagnosis.confidence == "high"


class TestAgentCSVInput:

    CSV_BEARING = (
        "machine_id,timestamp,temperature_c,vibration_mms,pressure_bar,current_a,rpm\n"
        "PUMP-01,2024-01-15T08:00:00+00:00,71.2,2.3,5.1,22.5,1482\n"
        "PUMP-01,2024-01-15T08:00:10+00:00,71.8,2.5,5.2,22.7,1480\n"
        "PUMP-01,2024-01-15T08:00:20+00:00,72.0,2.4,5.0,22.6,1479\n"
        "PUMP-01,2024-01-15T08:00:30+00:00,74.5,3.1,5.1,23.1,1478\n"
        "PUMP-01,2024-01-15T08:00:40+00:00,85.1,7.2,4.9,24.1,1472\n"
        "PUMP-01,2024-01-15T08:00:50+00:00,94.3,10.5,4.8,24.8,1468\n"
        "PUMP-01,2024-01-15T08:01:00+00:00,103.8,13.1,4.7,25.2,1463\n"
        "PUMP-01,2024-01-15T08:01:10+00:00,109.4,14.7,4.6,25.7,1459\n"
    )

    def test_csv_input_produces_diagnosis(self):
        from src.sensor_data import SensorDataLoader
        readings = SensorDataLoader.from_csv_string(self.CSV_BEARING)
        agent = make_agent()
        run = agent.run(readings)
        assert run.diagnosis is not None
        assert run.diagnosis.fault_id in ("BEARING_FAILURE", "OVERHEATING")
