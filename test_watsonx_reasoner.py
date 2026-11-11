"""
tests/test_watsonx_reasoner.py — Tests for the IBM watsonx.ai integration.

Tests cover:
  - WatsonxReasoner with mocked ibm-watsonx-ai SDK
  - JSON response parsing and fallback
  - Config status helper (safe — never exposes credentials)
  - Agent fallback behaviour when watsonx is not configured
  - Agent fallback behaviour when watsonx call raises an exception
  - Live connection test (skipped unless credentials are configured)
"""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from src.anomaly_detection import AnomalyDetector
from src.fault_knowledge_base import SignalInvestigator
from src.sensor_data import SensorSimulator
from src.watsonx_reasoner import (
    DEFAULT_WATSONX_MODEL,
    _extract_json,
    _sanitise_error,
    get_watsonx_config_status,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_report_and_candidates(scenario: str = "bearing_failure"):
    """Generate a real AnomalyReport + FaultCandidates for the given scenario."""
    sim = SensorSimulator(seed=42)
    readings = sim.generate(scenario, n_readings=12)
    detector = AnomalyDetector(n_estimators=50, random_state=0)
    report = detector.analyze(readings)
    investigator = SignalInvestigator()
    candidates = investigator.investigate(report)
    return report, candidates


VALID_RESPONSE_JSON = {
    "fault_id": "BEARING_FAILURE",
    "fault_name": "Bearing Failure",
    "confidence": "high",
    "severity": "high",
    "urgency": "immediate",
    "explanation": "Vibration is critically elevated indicating bearing wear.",
    "evidence": ["vibration_mms is high at 12.50"],
    "recommended_actions": ["Stop machine immediately.", "Inspect bearings."],
    "differential_diagnosis": ["Rotor Imbalance"],
    "uncertainty_notes": "",
}


# ---------------------------------------------------------------------------
# Tests: _extract_json
# ---------------------------------------------------------------------------

class TestExtractJson:

    def test_parses_direct_json(self):
        text = json.dumps(VALID_RESPONSE_JSON)
        result = _extract_json(text)
        assert result is not None
        assert result["fault_id"] == "BEARING_FAILURE"

    def test_parses_fenced_json(self):
        text = "Here is your result:\n```json\n" + json.dumps(VALID_RESPONSE_JSON) + "\n```\n"
        result = _extract_json(text)
        assert result is not None
        assert result["confidence"] == "high"

    def test_parses_embedded_json(self):
        text = "Some preamble text\n" + json.dumps(VALID_RESPONSE_JSON) + "\nSome trailing text."
        result = _extract_json(text)
        assert result is not None

    def test_returns_none_for_invalid(self):
        result = _extract_json("This is just plain text with no JSON.")
        assert result is None

    def test_returns_none_for_empty(self):
        result = _extract_json("")
        assert result is None


# ---------------------------------------------------------------------------
# Tests: _sanitise_error
# ---------------------------------------------------------------------------

class TestSanitiseError:

    def test_removes_bearer_token(self):
        msg = "HTTP 403: Bearer eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJhcGlrZXkifQ.sig"
        result = _sanitise_error(msg)
        assert "Bearer" in result
        assert "eyJhbGciOiJSUzI1NiJ9" not in result
        assert "[REDACTED]" in result

    def test_removes_long_tokens(self):
        # Simulate a 40-char API key embedded in an error message
        key = "A" * 45
        msg = f"Authentication failed with key={key}"
        result = _sanitise_error(msg)
        assert key not in result

    def test_preserves_short_words(self):
        msg = "Connection refused at https://us-south.ml.cloud.ibm.com"
        result = _sanitise_error(msg)
        # Short words preserved
        assert "Connection" in result


# ---------------------------------------------------------------------------
# Tests: get_watsonx_config_status (safe — checks env var presence only)
# ---------------------------------------------------------------------------

class TestGetWatsonxConfigStatus:

    def test_saas_not_configured_when_no_env(self):
        with patch.dict(os.environ, {}, clear=False):
            # Remove watsonx keys if present
            clean = {
                k: v for k, v in os.environ.items()
                if not k.startswith("WATSONX")
            }
            with patch.dict(os.environ, clean, clear=True):
                cfg = get_watsonx_config_status()
                assert cfg["mode"] == "SaaS"
                assert cfg["configured"] is False
                assert cfg["api_key_set"] is False
                assert cfg["project_id_set"] is False

    def test_saas_configured_when_key_and_project_set(self):
        env = {
            "WATSONX_API_KEY": "fake-key",
            "WATSONX_PROJECT_ID": "fake-project",
        }
        with patch.dict(os.environ, env):
            cfg = get_watsonx_config_status()
            assert cfg["mode"] == "SaaS"
            assert cfg["configured"] is True
            assert cfg["api_key_set"] is True
            assert cfg["project_id_set"] is True
            # Values should NOT be present in the status dict
            assert "fake-key" not in str(cfg)
            assert "fake-project" not in str(cfg)

    def test_cpd_mode_activated_by_cpd_url(self):
        env = {
            "WATSONX_CPD_URL": "https://cpd.example.com",
            "WATSONX_CPD_USERNAME": "mysecretuser",
            "WATSONX_CPD_PASSWORD": "mysecretpass",
            "WATSONX_PROJECT_ID": "proj-id",
        }
        with patch.dict(os.environ, env):
            cfg = get_watsonx_config_status()
            assert cfg["mode"] == "CP4D"
            assert cfg["configured"] is True
            # The actual credential VALUES must not appear in the status dict
            cfg_str = str(cfg)
            assert "mysecretuser" not in cfg_str
            assert "mysecretpass" not in cfg_str
            assert "proj-id" not in cfg_str
            # But the status keys should be booleans
            assert cfg["cpd_username_set"] is True
            assert cfg["cpd_password_set"] is True
            assert cfg["project_id_set"] is True

    def test_cpd_mode_not_configured_without_project(self):
        env = {
            "WATSONX_CPD_URL": "https://cpd.example.com",
            "WATSONX_CPD_USERNAME": "user",
            "WATSONX_CPD_PASSWORD": "pass",
            # WATSONX_PROJECT_ID intentionally absent
        }
        # Make sure WATSONX_PROJECT_ID is not inherited from environment
        base_env = {k: v for k, v in os.environ.items() if k != "WATSONX_PROJECT_ID"}
        with patch.dict(os.environ, {**base_env, **env}, clear=True):
            cfg = get_watsonx_config_status()
            assert cfg["mode"] == "CP4D"
            assert cfg["configured"] is False

    def test_default_model_id_present(self):
        clean = {k: v for k, v in os.environ.items() if not k.startswith("WATSONX")}
        with patch.dict(os.environ, clean, clear=True):
            cfg = get_watsonx_config_status()
            assert "model_id" in cfg
            assert cfg["model_id"] == DEFAULT_WATSONX_MODEL


# ---------------------------------------------------------------------------
# Tests: WatsonxReasoner with mocked ibm-watsonx-ai SDK
# ---------------------------------------------------------------------------

class TestWatsonxReasonerMocked:
    """Test WatsonxReasoner behaviour with a mock SDK — no real API calls."""

    def _make_reasoner_with_mock(self, mock_response: str):
        """
        Patch ibm_watsonx_ai.Credentials and ModelInference so no real SDK
        calls are made. Returns (reasoner, mock_model).
        """
        from src.watsonx_reasoner import WatsonxReasoner

        mock_model = MagicMock()
        mock_model.generate_text.return_value = mock_response

        with (
            patch("ibm_watsonx_ai.Credentials") as mock_creds,
            patch("ibm_watsonx_ai.foundation_models.ModelInference") as mock_mi,
        ):
            mock_mi.return_value = mock_model
            mock_creds.return_value = MagicMock()
            env = {
                "WATSONX_API_KEY": "test-key",
                "WATSONX_PROJECT_ID": "test-project",
            }
            with patch.dict(os.environ, env):
                reasoner = WatsonxReasoner()
                reasoner._model = mock_model  # inject mock directly
        return reasoner, mock_model

    def test_diagnose_returns_diagnosis_result(self):
        reasoner, mock_model = self._make_reasoner_with_mock(
            json.dumps(VALID_RESPONSE_JSON)
        )
        report, candidates = make_report_and_candidates("bearing_failure")
        result = reasoner.diagnose(report, candidates)

        assert result.fault_id == "BEARING_FAILURE"
        assert result.fault_name == "Bearing Failure"
        assert result.confidence == "high"
        assert result.severity == "high"
        assert result.urgency == "immediate"
        assert len(result.recommended_actions) > 0
        assert isinstance(result.evidence, list)

    def test_diagnose_handles_fenced_json(self):
        fenced = "```json\n" + json.dumps(VALID_RESPONSE_JSON) + "\n```"
        reasoner, mock_model = self._make_reasoner_with_mock(fenced)
        report, candidates = make_report_and_candidates("bearing_failure")
        result = reasoner.diagnose(report, candidates)
        assert result.fault_id == "BEARING_FAILURE"

    def test_diagnose_handles_unparseable_response(self):
        """When Granite returns unparseable text, fallback diagnosis is used."""
        reasoner, mock_model = self._make_reasoner_with_mock(
            "I cannot produce a diagnosis for this input."
        )
        report, candidates = make_report_and_candidates("bearing_failure")
        result = reasoner.diagnose(report, candidates)
        # Should return a fallback, not raise
        assert result.fault_id == "PARSE_ERROR"
        assert result.confidence == "uncertain"
        assert "parse error" in result.uncertainty_notes.lower()

    def test_diagnose_machine_id_preserved(self):
        reasoner, _ = self._make_reasoner_with_mock(json.dumps(VALID_RESPONSE_JSON))
        report, candidates = make_report_and_candidates("bearing_failure")
        result = reasoner.diagnose(report, candidates)
        assert result.machine_id == report.machine_id

    def test_model_id_property(self):
        from src.watsonx_reasoner import WatsonxReasoner

        mock_model = MagicMock()

        with (
            patch("ibm_watsonx_ai.Credentials"),
            patch("ibm_watsonx_ai.foundation_models.ModelInference", return_value=mock_model),
        ):
            env = {
                "WATSONX_API_KEY": "k",
                "WATSONX_PROJECT_ID": "p",
                "WATSONX_MODEL_ID": "ibm/granite-3-2b-instruct",
            }
            with patch.dict(os.environ, env):
                reasoner = WatsonxReasoner()
                assert reasoner.model_id == "ibm/granite-3-2b-instruct"


# ---------------------------------------------------------------------------
# Tests: Agent fallback behaviour
# ---------------------------------------------------------------------------

class TestAgentFallbackBehaviour:
    """
    Verify the agent correctly falls back to the rule-based reasoner
    when watsonx.ai is not configured or when the call fails.
    """

    def test_agent_uses_rule_based_when_no_credentials(self):
        """Without credentials, agent should still produce a valid diagnosis."""
        from src.agent import DiagnosisAgent

        sim = SensorSimulator(seed=42)
        readings = sim.generate("bearing_failure", n_readings=12)

        clean = {k: v for k, v in os.environ.items() if not k.startswith("WATSONX")}
        clean = {k: v for k, v in clean.items() if k != "OPENAI_API_KEY"}

        with patch.dict(os.environ, clean, clear=True):
            agent = DiagnosisAgent(use_llm=True)
            run = agent.run(readings)

        assert run.diagnosis is not None
        assert run.diagnosis.fault_id == "BEARING_FAILURE"
        reason_step = next(s for s in run.steps if s.step_name == "REASON")
        # Should be rule-based fallback since no credentials
        assert "rule-based" in reason_step.details.get("mode", "")
        assert reason_step.status == "warn"

    def test_agent_falls_back_when_watsonx_raises(self):
        """Agent falls back to rule-based when WatsonxReasoner raises an exception."""
        from src.agent import DiagnosisAgent

        sim = SensorSimulator(seed=42)
        readings = sim.generate("bearing_failure", n_readings=12)

        env = {
            "WATSONX_API_KEY": "bad-key",
            "WATSONX_PROJECT_ID": "bad-project",
        }

        # WatsonxReasoner is imported locally inside _step_reason;
        # patch at the module where it is defined.
        with patch.dict(os.environ, env):
            with patch(
                "src.watsonx_reasoner.WatsonxReasoner",
                side_effect=Exception("simulated connection failure"),
            ):
                agent = DiagnosisAgent(use_llm=True)
                run = agent.run(readings)

        assert run.diagnosis is not None
        reason_step = next(s for s in run.steps if s.step_name == "REASON")
        # Should have fallen back
        assert "fallback" in reason_step.details.get("mode", "")

    def test_agent_rule_based_mode_never_calls_watsonx(self):
        """use_llm=False must not attempt any LLM calls whatsoever."""
        from src.agent import DiagnosisAgent

        sim = SensorSimulator(seed=42)
        readings = sim.generate("overheating", n_readings=12)

        # WatsonxReasoner is imported locally; patch at its source module.
        with patch("src.watsonx_reasoner.WatsonxReasoner") as mock_wx:
            agent = DiagnosisAgent(use_llm=False)
            run = agent.run(readings)
            mock_wx.assert_not_called()

        assert run.diagnosis is not None
        reason_step = next(s for s in run.steps if s.step_name == "REASON")
        assert reason_step.details.get("mode") == "rule-based"
        assert reason_step.status == "ok"

    def test_all_scenarios_succeed_with_fallback(self):
        """All fault scenarios should produce correct diagnoses using rule-based fallback."""
        from src.agent import DiagnosisAgent

        scenarios_and_faults = [
            ("bearing_failure",     "BEARING_FAILURE"),
            ("overheating",         "OVERHEATING"),
            ("cavitation",          "CAVITATION"),
            ("electrical_overload", "ELECTRICAL_OVERLOAD"),
            ("rotor_imbalance",     "ROTOR_IMBALANCE"),
        ]

        clean = {k: v for k, v in os.environ.items() if not k.startswith("WATSONX")}
        clean = {k: v for k, v in clean.items() if k != "OPENAI_API_KEY"}

        with patch.dict(os.environ, clean, clear=True):
            sim = SensorSimulator(seed=42)
            for scenario, expected_fault in scenarios_and_faults:
                readings = sim.generate(scenario, n_readings=12)
                agent = DiagnosisAgent(use_llm=True)
                run = agent.run(readings)
                assert run.diagnosis is not None, f"No diagnosis for {scenario}"
                assert run.diagnosis.fault_id == expected_fault, (
                    f"{scenario}: expected {expected_fault}, got {run.diagnosis.fault_id}"
                )


# ---------------------------------------------------------------------------
# Tests: Live watsonx.ai connection (skipped unless credentials are set)
# ---------------------------------------------------------------------------

class TestWatsonxLiveConnection:
    """
    These tests make REAL requests to IBM watsonx.ai.
    They are skipped unless WATSONX_API_KEY and WATSONX_PROJECT_ID are set.

    To run them:
        WATSONX_API_KEY=<key> WATSONX_PROJECT_ID=<id> pytest -k live
    """

    @pytest.fixture(autouse=True)
    def skip_if_not_configured(self):
        cfg = get_watsonx_config_status()
        if not cfg["configured"]:
            pytest.skip(
                "watsonx.ai credentials not configured. "
                "Set WATSONX_API_KEY and WATSONX_PROJECT_ID to run live tests."
            )

    def test_live_connection_succeeds(self):
        from src.watsonx_reasoner import test_watsonx_connection
        result = test_watsonx_connection()
        # If credentials are set but connection fails, this is a real failure
        assert result["success"], f"Live connection failed: {result['message']}"
        assert result["model_id"]

    def test_live_diagnosis_bearing_failure(self):
        from src.watsonx_reasoner import WatsonxReasoner
        reasoner = WatsonxReasoner()
        report, candidates = make_report_and_candidates("bearing_failure")
        result = reasoner.diagnose(report, candidates)

        assert result.fault_id  # any non-empty string
        assert result.confidence in ("high", "medium", "low", "uncertain")
        assert result.severity in ("low", "medium", "high", "critical", "unknown")
        assert result.urgency in ("monitor", "schedule", "immediate")
        assert isinstance(result.explanation, str)
        assert len(result.explanation) > 10
        assert isinstance(result.recommended_actions, list)
        assert len(result.recommended_actions) > 0

    def test_live_diagnosis_result_not_empty(self):
        from src.watsonx_reasoner import WatsonxReasoner
        reasoner = WatsonxReasoner()
        report, candidates = make_report_and_candidates("electrical_overload")
        result = reasoner.diagnose(report, candidates)
        assert result.fault_name
        assert result.explanation
