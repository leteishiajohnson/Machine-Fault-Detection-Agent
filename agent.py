"""
agent.py — Agentic orchestrator for the Machine Fault Diagnosis pipeline.

The DiagnosisAgent executes a structured, step-by-step agentic workflow:

  Step 1: INGEST        — Load and validate sensor readings
  Step 2: DETECT        — Run anomaly detection (Isolation Forest + thresholds)
  Step 3: INVESTIGATE   — Correlate signals with fault knowledge base
  Step 4: REASON        — IBM Granite / LLM / rule-based analyses evidence
  Step 5: DIAGNOSE      — Finalise structured diagnosis
  Step 6: EXPLAIN       — Format human-readable explanation
  Step 7: RECOMMEND     — Surface prioritised maintenance actions

Each step emits a StepResult that records inputs, outputs, timing, and
any step-level notes. The full execution trace is preserved in AgentRun.

Reasoner priority (when use_llm=True):
  1. IBM watsonx.ai (WatsonxReasoner) — if WATSONX_API_KEY + WATSONX_PROJECT_ID set
  2. OpenAI-compatible (LLMReasoner)  — if OPENAI_API_KEY set
  3. Rule-based fallback              — always available, no API key needed
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

from .anomaly_detection import AnomalyDetector, AnomalyReport, summarise_anomaly_report
from .fault_knowledge_base import FaultCandidate, SignalInvestigator
from .llm_reasoner import DiagnosisResult, LLMReasoner, RuleBasedReasoner
from .sensor_data import SensorReading


# ---------------------------------------------------------------------------
# Step result
# ---------------------------------------------------------------------------

@dataclass
class StepResult:
    step_name: str
    status: str              # "ok" | "warn" | "error" | "skipped"
    duration_ms: float
    summary: str
    details: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Full agent run
# ---------------------------------------------------------------------------

@dataclass
class AgentRun:
    run_id: str
    machine_id: str
    started_at: datetime
    finished_at: Optional[datetime]
    steps: List[StepResult]
    anomaly_report: Optional[AnomalyReport]
    fault_candidates: List[FaultCandidate]
    diagnosis: Optional[DiagnosisResult]

    @property
    def succeeded(self) -> bool:
        return all(s.status in ("ok", "warn", "skipped") for s in self.steps)

    @property
    def duration_ms(self) -> float:
        if self.finished_at and self.started_at:
            return (self.finished_at - self.started_at).total_seconds() * 1000
        return 0.0


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class DiagnosisAgent:
    """
    Agentic orchestrator for machine fault diagnosis.

    Parameters
    ----------
    use_llm : bool
        If True, try watsonx.ai then OpenAI then rule-based fallback.
        If False, use the deterministic rule-based reasoner only.
    llm_api_key : str, optional
        Override OPENAI_API_KEY env var (used for OpenAI-compat path).
    llm_base_url : str, optional
        Override base URL for OpenAI-compat endpoint.
    llm_model : str, optional
        Model name override for OpenAI-compat path.
    on_step : callable, optional
        Callback invoked after each step: on_step(step_result).
        Useful for streaming progress to a UI.
    """

    def __init__(
        self,
        use_llm: bool = True,
        llm_api_key: Optional[str] = None,
        llm_base_url: Optional[str] = None,
        llm_model: Optional[str] = None,
        on_step: Optional[Callable[[StepResult], None]] = None,
    ):
        self._detector = AnomalyDetector()
        self._investigator = SignalInvestigator()
        self._on_step = on_step

        self._use_llm = use_llm
        self._llm_kwargs = dict(
            api_key=llm_api_key,
            base_url=llm_base_url,
            model=llm_model,
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, readings: List[SensorReading]) -> AgentRun:
        """
        Execute the full 7-step diagnostic pipeline.

        Returns an AgentRun containing the step trace, anomaly report,
        fault candidates, and final diagnosis.
        """
        run_id = f"RUN-{datetime.now(tz=timezone.utc).strftime('%Y%m%d%H%M%S')}"
        machine_id = readings[-1].machine_id if readings else "UNKNOWN"

        run = AgentRun(
            run_id=run_id,
            machine_id=machine_id,
            started_at=datetime.now(tz=timezone.utc),
            finished_at=None,
            steps=[],
            anomaly_report=None,
            fault_candidates=[],
            diagnosis=None,
        )

        # ---- Step 1: INGEST -----------------------------------------------
        step = self._step_ingest(readings)
        run.steps.append(step)
        self._emit(step)
        if step.status == "error":
            run.finished_at = datetime.now(tz=timezone.utc)
            return run

        # ---- Step 2: DETECT -----------------------------------------------
        step, report = self._step_detect(readings)
        run.steps.append(step)
        self._emit(step)
        run.anomaly_report = report
        if step.status == "error":
            run.finished_at = datetime.now(tz=timezone.utc)
            return run

        # ---- Step 3: INVESTIGATE ------------------------------------------
        step, candidates = self._step_investigate(report)
        run.steps.append(step)
        self._emit(step)
        run.fault_candidates = candidates

        # ---- Step 4: REASON -----------------------------------------------
        step, diagnosis = self._step_reason(report, candidates)
        run.steps.append(step)
        self._emit(step)
        run.diagnosis = diagnosis

        # ---- Step 5: DIAGNOSE (validate & finalise) -----------------------
        step = self._step_diagnose(diagnosis)
        run.steps.append(step)
        self._emit(step)

        # ---- Step 6: EXPLAIN ----------------------------------------------
        step = self._step_explain(diagnosis, report)
        run.steps.append(step)
        self._emit(step)

        # ---- Step 7: RECOMMEND --------------------------------------------
        step = self._step_recommend(diagnosis)
        run.steps.append(step)
        self._emit(step)

        run.finished_at = datetime.now(tz=timezone.utc)
        return run

    # ------------------------------------------------------------------
    # Individual steps
    # ------------------------------------------------------------------

    def _step_ingest(self, readings: List[SensorReading]) -> StepResult:
        t0 = time.monotonic()
        try:
            if not readings:
                return StepResult(
                    step_name="INGEST",
                    status="error",
                    duration_ms=0,
                    summary="No sensor readings provided.",
                    details={},
                )
            if len(readings) < 4:
                return StepResult(
                    step_name="INGEST",
                    status="error",
                    duration_ms=_ms(t0),
                    summary=f"Insufficient data: {len(readings)} readings (minimum 4 required).",
                    details={"n_readings": len(readings)},
                )
            machine_ids = {r.machine_id for r in readings}
            return StepResult(
                step_name="INGEST",
                status="ok",
                duration_ms=_ms(t0),
                summary=(
                    f"Ingested {len(readings)} readings for machine(s): "
                    f"{', '.join(machine_ids)}"
                ),
                details={
                    "n_readings": len(readings),
                    "machine_ids": list(machine_ids),
                    "time_range": {
                        "start": readings[0].timestamp.isoformat(),
                        "end":   readings[-1].timestamp.isoformat(),
                    },
                },
            )
        except Exception as exc:
            return StepResult(
                step_name="INGEST",
                status="error",
                duration_ms=_ms(t0),
                summary=f"Ingest error: {exc}",
                details={"error": str(exc)},
            )

    def _step_detect(
        self, readings: List[SensorReading]
    ) -> tuple[StepResult, Optional[AnomalyReport]]:
        t0 = time.monotonic()
        try:
            report = self._detector.analyze(readings)
            n_flags = len(report.flagged_signals)
            status = "warn" if report.is_anomalous else "ok"
            summary = summarise_anomaly_report(report)
            return (
                StepResult(
                    step_name="DETECT",
                    status=status,
                    duration_ms=_ms(t0),
                    summary=summary,
                    details={
                        "is_anomalous": report.is_anomalous,
                        "anomaly_score": report.anomaly_score,
                        "n_flagged_signals": n_flags,
                        "flagged": [
                            {
                                "signal": fa.signal,
                                "value": fa.value,
                                "severity": fa.severity,
                                "direction": fa.direction,
                                "z_score": fa.z_score,
                            }
                            for fa in report.flagged_signals
                        ],
                    },
                ),
                report,
            )
        except Exception as exc:
            return (
                StepResult(
                    step_name="DETECT",
                    status="error",
                    duration_ms=_ms(t0),
                    summary=f"Detection error: {exc}",
                    details={"error": str(exc)},
                ),
                None,
            )

    def _step_investigate(
        self, report: AnomalyReport
    ) -> tuple[StepResult, List[FaultCandidate]]:
        t0 = time.monotonic()
        try:
            candidates = self._investigator.investigate(report)
            top = candidates[0] if candidates else None
            summary = (
                f"Identified {len(candidates)} candidate fault(s). "
                + (
                    f"Top candidate: {top.fault.name} "
                    f"(match score: {top.match_score:.0%})"
                    if top else "No candidates."
                )
            )
            return (
                StepResult(
                    step_name="INVESTIGATE",
                    status="ok",
                    duration_ms=_ms(t0),
                    summary=summary,
                    details={
                        "n_candidates": len(candidates),
                        "candidates": [
                            {
                                "fault_id":    c.fault.fault_id,
                                "fault_name":  c.fault.name,
                                "match_score": c.match_score,
                                "matched_signals": c.matched_signals,
                            }
                            for c in candidates
                        ],
                    },
                ),
                candidates,
            )
        except Exception as exc:
            return (
                StepResult(
                    step_name="INVESTIGATE",
                    status="error",
                    duration_ms=_ms(t0),
                    summary=f"Investigation error: {exc}",
                    details={"error": str(exc)},
                ),
                [],
            )

    def _step_reason(
        self,
        report: AnomalyReport,
        candidates: List[FaultCandidate],
    ) -> tuple[StepResult, Optional[DiagnosisResult]]:
        import os
        t0 = time.monotonic()
        try:
            if not self._use_llm:
                reasoner: object = RuleBasedReasoner()
                mode_label = "rule-based"
                diagnosis = reasoner.diagnose(report, candidates)
                return (
                    StepResult(
                        step_name="REASON",
                        status="ok",
                        duration_ms=_ms(t0),
                        summary=(
                            f"[{mode_label}] Reasoned diagnosis: {diagnosis.fault_name} "
                            f"(confidence: {diagnosis.confidence})"
                        ),
                        details={
                            "mode": mode_label,
                            "fault_id":   diagnosis.fault_id,
                            "confidence": diagnosis.confidence,
                        },
                    ),
                    diagnosis,
                )

            # Try watsonx.ai first (preferred when WATSONX_API_KEY + PROJECT_ID set)
            watsonx_key = os.environ.get("WATSONX_API_KEY", "")
            watsonx_cpd = os.environ.get("WATSONX_CPD_URL", "")
            watsonx_project = os.environ.get("WATSONX_PROJECT_ID", "")
            has_watsonx = (watsonx_project and (watsonx_key or watsonx_cpd))

            if has_watsonx:
                try:
                    from .watsonx_reasoner import WatsonxReasoner
                    wx_reasoner = WatsonxReasoner()
                    diagnosis = wx_reasoner.diagnose(report, candidates)
                    return (
                        StepResult(
                            step_name="REASON",
                            status="ok",
                            duration_ms=_ms(t0),
                            summary=(
                                f"[IBM Granite/{wx_reasoner.model_id}] "
                                f"Reasoned diagnosis: {diagnosis.fault_name} "
                                f"(confidence: {diagnosis.confidence})"
                            ),
                            details={
                                "mode": "watsonx",
                                "model_id": wx_reasoner.model_id,
                                "fault_id":   diagnosis.fault_id,
                                "confidence": diagnosis.confidence,
                            },
                        ),
                        diagnosis,
                    )
                except Exception as wx_exc:
                    # watsonx failed — continue to OpenAI fallback
                    pass

            # Try OpenAI-compatible path
            openai_key = self._llm_kwargs.get("api_key") or os.environ.get("OPENAI_API_KEY", "")
            if openai_key:
                try:
                    reasoner_llm = LLMReasoner(**self._llm_kwargs)
                    diagnosis = reasoner_llm.diagnose(report, candidates)
                    return (
                        StepResult(
                            step_name="REASON",
                            status="ok",
                            duration_ms=_ms(t0),
                            summary=(
                                f"[LLM] Reasoned diagnosis: {diagnosis.fault_name} "
                                f"(confidence: {diagnosis.confidence})"
                            ),
                            details={
                                "mode": "openai",
                                "fault_id":   diagnosis.fault_id,
                                "confidence": diagnosis.confidence,
                            },
                        ),
                        diagnosis,
                    )
                except Exception:
                    pass

            # Final fallback: rule-based
            diagnosis = RuleBasedReasoner().diagnose(report, candidates)
            # Determine why we fell back
            if has_watsonx:
                fallback_reason = "watsonx.ai call failed"
            elif openai_key:
                fallback_reason = "OpenAI call failed"
            else:
                fallback_reason = "no LLM credentials configured"

            return (
                StepResult(
                    step_name="REASON",
                    status="warn",
                    duration_ms=_ms(t0),
                    summary=(
                        f"[rule-based fallback; {fallback_reason}] "
                        f"Diagnosis: {diagnosis.fault_name}"
                    ),
                    details={
                        "mode":  "rule-based-fallback",
                        "reason": fallback_reason,
                        "fault_id":   diagnosis.fault_id,
                        "confidence": diagnosis.confidence,
                    },
                ),
                diagnosis,
            )
        except Exception as exc2:
            return (
                StepResult(
                    step_name="REASON",
                    status="error",
                    duration_ms=_ms(t0),
                    summary=f"Reasoning error: {exc2}",
                    details={"error": str(exc2)},
                ),
                None,
            )

    def _step_diagnose(
        self, diagnosis: Optional[DiagnosisResult]
    ) -> StepResult:
        t0 = time.monotonic()
        if diagnosis is None:
            return StepResult(
                step_name="DIAGNOSE",
                status="error",
                duration_ms=_ms(t0),
                summary="No diagnosis available — reasoning step failed.",
                details={},
            )
        return StepResult(
            step_name="DIAGNOSE",
            status="ok",
            duration_ms=_ms(t0),
            summary=(
                f"Diagnosis finalised: {diagnosis.fault_name} | "
                f"Severity: {diagnosis.severity.upper()} | "
                f"Urgency: {diagnosis.urgency.upper()}"
            ),
            details=diagnosis.to_dict(),
        )

    def _step_explain(
        self,
        diagnosis: Optional[DiagnosisResult],
        report: AnomalyReport,
    ) -> StepResult:
        t0 = time.monotonic()
        if diagnosis is None:
            return StepResult(
                step_name="EXPLAIN",
                status="skipped",
                duration_ms=_ms(t0),
                summary="Explanation skipped — no diagnosis available.",
                details={},
            )
        explanation_text = _format_explanation(diagnosis, report)
        return StepResult(
            step_name="EXPLAIN",
            status="ok",
            duration_ms=_ms(t0),
            summary="Natural-language explanation generated.",
            details={"explanation": explanation_text},
        )

    def _step_recommend(
        self, diagnosis: Optional[DiagnosisResult]
    ) -> StepResult:
        t0 = time.monotonic()
        if diagnosis is None:
            return StepResult(
                step_name="RECOMMEND",
                status="skipped",
                duration_ms=_ms(t0),
                summary="Recommendations skipped — no diagnosis available.",
                details={},
            )
        actions = diagnosis.recommended_actions
        return StepResult(
            step_name="RECOMMEND",
            status="ok",
            duration_ms=_ms(t0),
            summary=f"Generated {len(actions)} maintenance recommendation(s).",
            details={"actions": actions},
        )

    # ------------------------------------------------------------------

    def _emit(self, step: StepResult) -> None:
        if self._on_step:
            self._on_step(step)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ms(t0: float) -> float:
    return round((time.monotonic() - t0) * 1000, 2)


def _format_explanation(
    diagnosis: DiagnosisResult, report: AnomalyReport
) -> str:
    """Format a readable multi-section explanation string."""
    lines = [
        f"MACHINE: {diagnosis.machine_id}",
        f"DIAGNOSIS: {diagnosis.fault_name}",
        f"CONFIDENCE: {diagnosis.confidence.upper()} | "
        f"SEVERITY: {diagnosis.severity.upper()} | "
        f"URGENCY: {diagnosis.urgency.upper()}",
        "",
        "EXPLANATION:",
        diagnosis.explanation,
    ]

    if diagnosis.evidence:
        lines += ["", "SUPPORTING EVIDENCE:"]
        for i, ev in enumerate(diagnosis.evidence, 1):
            lines.append(f"  {i}. {ev}")

    if diagnosis.differential_diagnosis:
        lines += ["", "DIFFERENTIAL DIAGNOSIS (other considered faults):"]
        for alt in diagnosis.differential_diagnosis:
            lines.append(f"  • {alt}")

    if diagnosis.uncertainty_notes:
        lines += ["", f"⚠ UNCERTAINTY: {diagnosis.uncertainty_notes}"]

    lines += ["", "RECOMMENDED ACTIONS:"]
    for i, act in enumerate(diagnosis.recommended_actions, 1):
        lines.append(f"  {i}. {act}")

    return "\n".join(lines)
