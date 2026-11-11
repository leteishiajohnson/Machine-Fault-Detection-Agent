"""
watsonx_reasoner.py — IBM watsonx.ai integration for the Machine Fault Diagnosis Agent.

Supports both IBM Cloud SaaS and IBM Cloud Pak for Data (CP4D) authentication.

IBM Cloud SaaS environment variables:
    WATSONX_API_KEY     — IBM Cloud IAM API key (required)
    WATSONX_USERNAME    — IBM Cloud IAM username / account email (optional, used by some SDK versions)
    WATSONX_PROJECT_ID  — watsonx.ai project ID (required)
    WATSONX_URL         — Regional endpoint, e.g. https://us-south.ml.cloud.ibm.com
    WATSONX_MODEL_ID    — Granite model ID (default: ibm/granite-3-8b-instruct)
    WATSONX_VERSION     — API version string passed to the SDK (e.g. 5.2)

IBM Cloud Pak for Data (CP4D) environment variables:
    WATSONX_CPD_URL      — CP4D instance URL, e.g. https://cpd.example.com
    WATSONX_CPD_USERNAME — CP4D username
    WATSONX_CPD_PASSWORD — CP4D password (or use WATSONX_CPD_APIKEY)
    WATSONX_CPD_APIKEY   — CP4D API key (alternative to password)
    WATSONX_PROJECT_ID   — watsonx.ai project ID (required)
    WATSONX_MODEL_ID     — Granite model ID (default: ibm/granite-3-8b-instruct)
    WATSONX_VERSION      — CP4D platform version string (e.g. 5.2)

To use CP4D mode, set WATSONX_CPD_URL (the presence of this variable activates CP4D auth).
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import List, Optional

from .anomaly_detection import AnomalyReport
from .fault_knowledge_base import FaultCandidate
from .llm_reasoner import DiagnosisResult, _build_system_prompt, _build_user_prompt


# ---------------------------------------------------------------------------
# Default model
# ---------------------------------------------------------------------------

# ibm/granite-3-8b-instruct is generally available on IBM Cloud watsonx.ai SaaS
# as of 2025. Granite 3 models support instruction-following and JSON output.
DEFAULT_WATSONX_MODEL = "ibm/granite-3-8b-instruct"


# ---------------------------------------------------------------------------
# Connection status helper (safe — never exposes credentials)
# ---------------------------------------------------------------------------

def get_watsonx_config_status() -> dict:
    """
    Returns a dict describing which watsonx.ai environment variables are set.
    Values are boolean (set / not set) — credentials are NEVER included.
    """
    cpd_url = os.environ.get("WATSONX_CPD_URL", "")
    is_cpd = bool(cpd_url)
    version = os.environ.get("WATSONX_VERSION", "")

    if is_cpd:
        return {
            "mode": "CP4D",
            "cpd_url_set": bool(cpd_url),
            "cpd_username_set": bool(os.environ.get("WATSONX_CPD_USERNAME")),
            "cpd_password_set": bool(os.environ.get("WATSONX_CPD_PASSWORD")),
            "cpd_apikey_set": bool(os.environ.get("WATSONX_CPD_APIKEY")),
            "project_id_set": bool(os.environ.get("WATSONX_PROJECT_ID")),
            "model_id": os.environ.get("WATSONX_MODEL_ID", DEFAULT_WATSONX_MODEL),
            "version": version or "4.8",
            "configured": bool(
                cpd_url
                and os.environ.get("WATSONX_PROJECT_ID")
                and (
                    os.environ.get("WATSONX_CPD_USERNAME")
                    and (
                        os.environ.get("WATSONX_CPD_PASSWORD")
                        or os.environ.get("WATSONX_CPD_APIKEY")
                    )
                )
            ),
        }
    else:
        return {
            "mode": "SaaS",
            "api_key_set": bool(os.environ.get("WATSONX_API_KEY")),
            "username_set": bool(os.environ.get("WATSONX_USERNAME")),
            "project_id_set": bool(os.environ.get("WATSONX_PROJECT_ID")),
            "url_set": bool(os.environ.get("WATSONX_URL")),
            "url": os.environ.get("WATSONX_URL", "https://us-south.ml.cloud.ibm.com"),
            "model_id": os.environ.get("WATSONX_MODEL_ID", DEFAULT_WATSONX_MODEL),
            "version": version,
            "configured": bool(
                os.environ.get("WATSONX_API_KEY")
                and os.environ.get("WATSONX_PROJECT_ID")
            ),
        }


# ---------------------------------------------------------------------------
# WatsonxReasoner
# ---------------------------------------------------------------------------

class WatsonxReasoner:
    """
    Calls IBM Granite through the official ibm-watsonx-ai SDK to produce a
    structured DiagnosisResult.

    Falls back gracefully when credentials are missing or the service is
    unavailable.  Never logs or exposes credentials.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        username: Optional[str] = None,
        project_id: Optional[str] = None,
        url: Optional[str] = None,
        model_id: Optional[str] = None,
        version: Optional[str] = None,
        # CP4D overrides
        cpd_url: Optional[str] = None,
        cpd_username: Optional[str] = None,
        cpd_password: Optional[str] = None,
        cpd_apikey: Optional[str] = None,
    ):
        from ibm_watsonx_ai import Credentials  # type: ignore
        from ibm_watsonx_ai.foundation_models import ModelInference  # type: ignore

        self._model_id = (
            model_id
            or os.environ.get("WATSONX_MODEL_ID", DEFAULT_WATSONX_MODEL)
        )

        # WATSONX_VERSION — used as the platform/API version for both SaaS and CP4D
        resolved_version = version or os.environ.get("WATSONX_VERSION", "")

        # Determine auth mode
        resolved_cpd_url = cpd_url or os.environ.get("WATSONX_CPD_URL", "")

        if resolved_cpd_url:
            # ---- CP4D authentication ----------------------------------------
            resolved_username = cpd_username or os.environ.get("WATSONX_CPD_USERNAME", "")
            resolved_password = cpd_password or os.environ.get("WATSONX_CPD_PASSWORD", "")
            resolved_cpd_apikey = cpd_apikey or os.environ.get("WATSONX_CPD_APIKEY", "")
            cpd_version = resolved_version or "4.8"   # fallback for CP4D

            if resolved_cpd_apikey:
                credentials = Credentials(
                    url=resolved_cpd_url,
                    api_key=resolved_cpd_apikey,
                    version=cpd_version,
                )
            else:
                credentials = Credentials(
                    url=resolved_cpd_url,
                    username=resolved_username,
                    password=resolved_password,
                    instance_id="openshift",
                    version=cpd_version,
                )
        else:
            # ---- IBM Cloud SaaS authentication ------------------------------
            resolved_key = api_key or os.environ.get("WATSONX_API_KEY", "")
            resolved_username = username or os.environ.get("WATSONX_USERNAME", "")
            resolved_url = url or os.environ.get(
                "WATSONX_URL", "https://us-south.ml.cloud.ibm.com"
            )
            # Build kwargs — only include optional fields when they are set
            cred_kwargs: dict = {"url": resolved_url, "api_key": resolved_key}
            if resolved_username:
                cred_kwargs["username"] = resolved_username
            if resolved_version:
                cred_kwargs["version"] = resolved_version
            credentials = Credentials(**cred_kwargs)

        resolved_project_id = project_id or os.environ.get("WATSONX_PROJECT_ID", "")

        self._model = ModelInference(
            model_id=self._model_id,
            credentials=credentials,
            project_id=resolved_project_id,
            params={
                "max_new_tokens": 1024,
                "temperature": 0.1,
                "repetition_penalty": 1.05,
            },
        )

    def diagnose(
        self,
        report: AnomalyReport,
        candidates: List[FaultCandidate],
    ) -> DiagnosisResult:
        """Call IBM Granite through watsonx.ai and return a structured DiagnosisResult."""
        system_prompt = _build_system_prompt()
        user_prompt = _build_user_prompt(report, candidates)

        # Combine system + user as a single prompt for Granite (instruct format)
        full_prompt = (
            f"<|system|>\n{system_prompt}\n<|user|>\n{user_prompt}\n<|assistant|>\n"
        )

        response = self._model.generate_text(prompt=full_prompt)

        raw = response if isinstance(response, str) else str(response)
        return _parse_watsonx_response(raw, report)

    @property
    def model_id(self) -> str:
        return self._model_id


# ---------------------------------------------------------------------------
# Response parser (shared utility)
# ---------------------------------------------------------------------------

def _parse_watsonx_response(raw: str, report: AnomalyReport) -> DiagnosisResult:
    """Parse a Granite JSON response into a DiagnosisResult."""
    data = _extract_json(raw)
    if data is None:
        data = _fallback_diagnosis(report, error=raw[:200])

    return DiagnosisResult(
        machine_id=report.machine_id,
        timestamp=datetime.now(tz=timezone.utc),
        fault_id=str(data.get("fault_id", "UNKNOWN")),
        fault_name=str(data.get("fault_name", "Unknown")),
        confidence=str(data.get("confidence", "uncertain")),
        severity=str(data.get("severity", "unknown")),
        urgency=str(data.get("urgency", "monitor")),
        explanation=str(data.get("explanation", "")),
        evidence=list(data.get("evidence", [])),
        recommended_actions=list(data.get("recommended_actions", [])),
        differential_diagnosis=list(data.get("differential_diagnosis", [])),
        uncertainty_notes=str(data.get("uncertainty_notes", "")),
        raw_llm_response=raw,
    )


def _extract_json(text: str) -> Optional[dict]:
    """Try several strategies to extract a JSON object from a model response."""
    # Strategy 1: direct parse
    try:
        return json.loads(text.strip())
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 2: extract from markdown code fence
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except (json.JSONDecodeError, ValueError):
            pass

    # Strategy 3: find the first { ... } block
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except (json.JSONDecodeError, ValueError):
            pass

    return None


def _fallback_diagnosis(report: AnomalyReport, error: str = "") -> dict:
    """Safe fallback dict when the model response cannot be parsed."""
    return {
        "fault_id":   "PARSE_ERROR",
        "fault_name": "Diagnosis unavailable",
        "confidence": "uncertain",
        "severity":   "unknown",
        "urgency":    "schedule",
        "explanation": (
            "The model response could not be parsed. "
            "Manual review of sensor data is recommended."
        ),
        "evidence":    [fa.description for fa in report.flagged_signals],
        "recommended_actions": [
            "Review sensor data manually.",
            "Re-run diagnosis.",
        ],
        "differential_diagnosis": [],
        "uncertainty_notes": f"Parse error: {error}",
    }


# ---------------------------------------------------------------------------
# Connectivity test (safe — never reveals credentials)
# ---------------------------------------------------------------------------

def test_watsonx_connection() -> dict:
    """
    Attempt a minimal watsonx.ai inference to verify connectivity.

    Returns a dict with keys:
        success (bool), model_id (str), message (str)

    Never includes credentials in the returned dict or in any exception message.
    """
    try:
        reasoner = WatsonxReasoner()
        # Minimal prompt to test token generation
        response = reasoner._model.generate_text(
            prompt="Respond with the single word: OK"
        )
        return {
            "success": True,
            "model_id": reasoner.model_id,
            "message": f"Connection successful. Model responded.",
        }
    except ImportError:
        return {
            "success": False,
            "model_id": os.environ.get("WATSONX_MODEL_ID", DEFAULT_WATSONX_MODEL),
            "message": (
                "ibm-watsonx-ai package is not installed. "
                "Run: pip install ibm-watsonx-ai"
            ),
        }
    except Exception as exc:
        # Sanitise the error: strip anything that looks like a key/token
        safe_msg = _sanitise_error(str(exc))
        return {
            "success": False,
            "model_id": os.environ.get("WATSONX_MODEL_ID", DEFAULT_WATSONX_MODEL),
            "message": f"Connection failed: {safe_msg}",
        }


def _sanitise_error(msg: str) -> str:
    """Remove anything resembling an API key or bearer token from an error string."""
    # Remove Bearer tokens
    msg = re.sub(r"Bearer\s+[A-Za-z0-9\-._~+/]+=*", "Bearer [REDACTED]", msg)
    # Remove anything that looks like an IAM key (starts with common prefixes)
    msg = re.sub(r"\b[A-Za-z0-9]{40,}\b", "[REDACTED]", msg)
    return msg
