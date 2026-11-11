"""
llm_reasoner.py — LLM reasoning and explanation layer.

Uses the OpenAI-compatible API (works with OpenAI GPT-4/3.5 and
IBM watsonx.ai OpenAI-compat endpoint).

The reasoner receives:
  - Latest sensor reading values with units
  - Anomaly detection results (flagged signals, IF score)
  - Ranked fault candidates from the knowledge base

It returns a structured DiagnosisResult containing:
  - Diagnosed fault (may differ from highest-scoring KB candidate)
  - Confidence level
  - Plain-English explanation of reasoning
  - Supporting evidence cited from sensor data
  - Severity assessment
  - Recommended maintenance actions
  - Uncertainty notes (when data is insufficient)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from .anomaly_detection import AnomalyReport, SIGNAL_NORMAL_RANGES
from .fault_knowledge_base import FaultCandidate
from .sensor_data import SIGNAL_UNITS


# ---------------------------------------------------------------------------
# Diagnosis result dataclass
# ---------------------------------------------------------------------------

@dataclass
class DiagnosisResult:
    machine_id: str
    timestamp: datetime
    fault_id: str
    fault_name: str
    confidence: str             # "high" | "medium" | "low" | "uncertain"
    severity: str               # "low" | "medium" | "high" | "critical"
    urgency: str                # "monitor" | "schedule" | "immediate"
    explanation: str            # narrative explanation from LLM
    evidence: List[str]         # cited sensor evidence
    recommended_actions: List[str]
    differential_diagnosis: List[str]  # other considered faults
    uncertainty_notes: str      # any caveats about data quality / ambiguity
    raw_llm_response: str = ""  # for debugging

    def to_dict(self) -> dict:
        d = {
            "machine_id": self.machine_id,
            "timestamp":  self.timestamp.isoformat(),
            "fault_id":   self.fault_id,
            "fault_name": self.fault_name,
            "confidence": self.confidence,
            "severity":   self.severity,
            "urgency":    self.urgency,
            "explanation": self.explanation,
            "evidence":   self.evidence,
            "recommended_actions": self.recommended_actions,
            "differential_diagnosis": self.differential_diagnosis,
            "uncertainty_notes": self.uncertainty_notes,
        }
        return d


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_system_prompt() -> str:
    return (
        "You are an expert industrial maintenance engineer with 20 years of experience "
        "diagnosing faults in rotating machinery, pumps, and motors using sensor data.\n\n"
        "You receive structured sensor data, anomaly detection results, and candidate faults "
        "identified by a signal analysis engine. Your task is to reason carefully about the "
        "evidence and produce a precise, well-supported machine fault diagnosis.\n\n"
        "Rules:\n"
        "1. Base your diagnosis on the sensor evidence provided — do NOT invent readings.\n"
        "2. If the evidence is ambiguous or insufficient, say so explicitly and lower confidence.\n"
        "3. Consider alternative explanations (differential diagnosis).\n"
        "4. Your response MUST be a single valid JSON object — no prose, no markdown fences.\n"
        "5. All recommendations must be actionable and specific to the identified fault.\n"
        "6. Severity levels: low | medium | high | critical\n"
        "7. Confidence levels: high | medium | low | uncertain\n"
        "8. Urgency levels: monitor | schedule | immediate\n"
    )


def _build_user_prompt(
    report: AnomalyReport,
    candidates: List[FaultCandidate],
) -> str:
    # --- Signal readings section ------------------------------------------
    signal_lines = []
    for sig, val in report.latest_reading.items():
        lo, hi = SIGNAL_NORMAL_RANGES.get(sig, (None, None))
        unit = SIGNAL_UNITS.get(sig, "")
        range_str = f" [normal: {lo}–{hi} {unit}]" if lo is not None else ""
        signal_lines.append(f"  {sig}: {val:.2f} {unit}{range_str}")

    # --- Anomaly section ---------------------------------------------------
    anomaly_lines = [
        f"  Anomaly detected: {'YES' if report.is_anomalous else 'NO'}",
        f"  Anomaly score (0-1, higher=worse): {report.anomaly_score:.3f}",
        f"  Number of readings in window: {report.n_readings}",
    ]
    if report.flagged_signals:
        anomaly_lines.append("  Flagged signals:")
        for fa in report.flagged_signals:
            anomaly_lines.append(f"    - {fa.description} [{fa.severity.upper()}]")
    else:
        anomaly_lines.append("  No signals individually outside thresholds.")

    # --- Candidate faults section -----------------------------------------
    cand_lines = []
    top_n = min(3, len(candidates))
    for i, c in enumerate(candidates[:top_n], 1):
        cand_lines.append(
            f"  {i}. {c.fault.name} (score: {c.match_score:.2f})"
        )
        if c.matched_signals:
            cand_lines.append(
                f"     Matched signals: {', '.join(c.matched_signals)}"
            )
        cand_lines.append(
            f"     Description: {c.fault.description[:200]}"
        )

    prompt = (
        "=== MACHINE SENSOR DATA ===\n"
        f"Machine ID: {report.machine_id}\n"
        "Latest sensor readings:\n"
        + "\n".join(signal_lines)
        + "\n\n=== ANOMALY DETECTION RESULTS ===\n"
        + "\n".join(anomaly_lines)
        + "\n\n=== CANDIDATE FAULTS (ranked by signal match) ===\n"
        + ("\n".join(cand_lines) if cand_lines else "  None identified.")
        + "\n\n=== YOUR TASK ===\n"
        "Analyse the above sensor evidence and candidate faults. "
        "Provide your diagnosis as a JSON object with EXACTLY these keys:\n"
        "{\n"
        '  "fault_id": "<FAULT_ID or NORMAL_OPERATION>",\n'
        '  "fault_name": "<human readable name>",\n'
        '  "confidence": "<high|medium|low|uncertain>",\n'
        '  "severity": "<low|medium|high|critical>",\n'
        '  "urgency": "<monitor|schedule|immediate>",\n'
        '  "explanation": "<2-4 sentences explaining your reasoning>",\n'
        '  "evidence": ["<cited sensor observation 1>", "<cited sensor observation 2>", ...],\n'
        '  "recommended_actions": ["<action 1>", "<action 2>", ...],\n'
        '  "differential_diagnosis": ["<alternative fault 1>", "<alternative fault 2>"],\n'
        '  "uncertainty_notes": "<any caveats, or empty string if none>"\n'
        "}\n"
    )
    return prompt


# ---------------------------------------------------------------------------
# LLM Reasoner
# ---------------------------------------------------------------------------

class LLMReasoner:
    """
    Calls an LLM (OpenAI or watsonx OpenAI-compatible) to produce a
    structured DiagnosisResult from the anomaly report and fault candidates.

    Environment variables:
      OPENAI_API_KEY     — required for OpenAI
      OPENAI_API_BASE    — override for watsonx / Azure / local LLMs
      LLM_MODEL          — model name (default: gpt-4o-mini)
    """

    DEFAULT_MODEL = "gpt-4o-mini"

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        from openai import OpenAI  # deferred import

        resolved_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        resolved_base = base_url or os.environ.get("OPENAI_API_BASE") or None
        self._model = model or os.environ.get("LLM_MODEL", self.DEFAULT_MODEL)

        self._client = OpenAI(
            api_key=resolved_key,
            base_url=resolved_base,
        )

    def diagnose(
        self,
        report: AnomalyReport,
        candidates: List[FaultCandidate],
    ) -> DiagnosisResult:
        """Call the LLM and return a structured DiagnosisResult."""
        system_prompt = _build_system_prompt()
        user_prompt = _build_user_prompt(report, candidates)

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.1,      # low temperature for consistency
            max_tokens=1024,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content or "{}"
        return self._parse_response(raw, report)

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse_response(
        self, raw: str, report: AnomalyReport
    ) -> DiagnosisResult:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Try to extract JSON from fenced code block if model wrapped it
            import re
            m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw, re.DOTALL)
            if m:
                data = json.loads(m.group(1))
            else:
                data = self._fallback_diagnosis(report, error=raw[:200])

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

    @staticmethod
    def _fallback_diagnosis(report: AnomalyReport, error: str = "") -> dict:
        """Return a safe fallback when the LLM response cannot be parsed."""
        return {
            "fault_id":   "PARSE_ERROR",
            "fault_name": "Diagnosis unavailable",
            "confidence": "uncertain",
            "severity":   "unknown",
            "urgency":    "schedule",
            "explanation": (
                "The LLM response could not be parsed. "
                "Manual review of sensor data is recommended."
            ),
            "evidence":    [fa.description for fa in report.flagged_signals],
            "recommended_actions": ["Review sensor data manually.", "Re-run diagnosis."],
            "differential_diagnosis": [],
            "uncertainty_notes": f"Parse error: {error}",
        }


# ---------------------------------------------------------------------------
# Fallback: rule-based reasoner (no LLM required)
# ---------------------------------------------------------------------------

class RuleBasedReasoner:
    """
    A deterministic fallback reasoner that produces a DiagnosisResult
    directly from the knowledge base candidates without calling an LLM.
    Useful for testing, offline use, or when no API key is available.
    """

    def diagnose(
        self,
        report: AnomalyReport,
        candidates: List[FaultCandidate],
    ) -> DiagnosisResult:
        if not report.is_anomalous or not candidates:
            top = candidates[0] if candidates else None
            fault_id = "NORMAL_OPERATION"
            fault_name = "Normal Operation"
            confidence = "high"
            severity = "low"
            urgency = "monitor"
            explanation = (
                "All sensor readings are within normal operating ranges. "
                "No anomalies were detected by the isolation forest or "
                "threshold analysis. The machine appears to be functioning correctly."
            )
            evidence: List[str] = []
            actions = ["Continue normal monitoring schedule."]
            differential: List[str] = []
            uncertainty = ""
        else:
            top_cand = candidates[0]
            fault = top_cand.fault
            fault_id = fault.fault_id
            fault_name = fault.name
            score = top_cand.match_score

            if score >= 0.70:
                confidence = "high"
            elif score >= 0.45:
                confidence = "medium"
            elif score >= 0.25:
                confidence = "low"
            else:
                confidence = "uncertain"

            severity = fault.severity_level
            urgency = fault.urgency

            # Build explanation from KB data
            explanation = (
                f"Signal analysis identified {fault_name} as the most likely fault "
                f"with a match score of {score:.0%}. {fault.description} "
                f"{fault.typical_progression}"
            )
            evidence = [fa.description for fa in report.flagged_signals]
            actions = fault.recommended_actions
            differential = [
                c.fault.name for c in candidates[1:3]
                if c.fault.fault_id != "NORMAL_OPERATION"
            ]
            uncertainty = (
                "" if confidence in ("high", "medium")
                else (
                    "Low match score — the sensor pattern does not strongly match "
                    "any single fault. Multiple faults or a novel failure mode are possible. "
                    "Further investigation by a qualified technician is recommended."
                )
            )

        return DiagnosisResult(
            machine_id=report.machine_id,
            timestamp=datetime.now(tz=timezone.utc),
            fault_id=fault_id,
            fault_name=fault_name,
            confidence=confidence,
            severity=severity,
            urgency=urgency,
            explanation=explanation,
            evidence=evidence,
            recommended_actions=actions,
            differential_diagnosis=differential,
            uncertainty_notes=uncertainty,
        )
