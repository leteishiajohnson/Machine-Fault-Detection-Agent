"""
fault_knowledge_base.py — Structured fault knowledge base and signal investigator.

The knowledge base encodes domain expertise about industrial pump/motor faults:
- Which signals are elevated or depressed for each fault
- Severity and urgency classifications
- Recommended maintenance actions

The SignalInvestigator:
- Correlates anomalous signals with known fault signatures
- Ranks candidate faults by match score
- Returns structured evidence for the LLM reasoning layer
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .anomaly_detection import AnomalyReport, SignalAnomaly


# ---------------------------------------------------------------------------
# Fault definitions
# ---------------------------------------------------------------------------

@dataclass
class FaultDefinition:
    fault_id: str
    name: str
    description: str
    # signal_name → ("high"|"low"|"any", weight 0-1)
    # weight represents how strongly this signal change is diagnostic for this fault
    signal_signatures: Dict[str, Tuple[str, float]]
    severity_level: str          # "low" | "medium" | "high" | "critical"
    urgency: str                 # "monitor" | "schedule" | "immediate"
    recommended_actions: List[str]
    typical_progression: str     # natural language description of how the fault develops


FAULT_KB: List[FaultDefinition] = [
    FaultDefinition(
        fault_id="BEARING_FAILURE",
        name="Bearing Failure",
        description=(
            "Mechanical wear or fatigue in rolling-element bearings. "
            "Manifests as elevated vibration (often with a characteristic frequency) "
            "and progressive temperature rise due to friction."
        ),
        signal_signatures={
            "vibration_mms":   ("high", 0.90),
            "temperature_c":   ("high", 0.85),  # temp rise strongly co-diagnostic
            "rpm":             ("low",  0.40),
        },
        severity_level="high",
        urgency="immediate",
        recommended_actions=[
            "Stop machine and inspect bearings immediately.",
            "Measure bearing clearance and check for pitting or spalling.",
            "Replace worn bearings; re-grease with correct specification lubricant.",
            "Re-align shaft after bearing replacement.",
            "Monitor vibration spectrum for 24 hours post-maintenance.",
        ],
        typical_progression=(
            "Vibration rises gradually then sharply; temperature follows "
            "shortly after. If unchecked, leads to catastrophic seizure."
        ),
    ),
    FaultDefinition(
        fault_id="OVERHEATING",
        name="Overheating",
        description=(
            "Excessive thermal build-up in motor windings or housing, typically "
            "caused by insufficient cooling, blocked ventilation, or prolonged overload."
        ),
        signal_signatures={
            "temperature_c":   ("high", 0.95),
            # Pure overheating does not produce dramatically elevated current.
            # A critically-high current reading MISMATCHES this fault's expectation
            # and shifts the diagnosis to ELECTRICAL_OVERLOAD.
            "current_a":       ("low",  0.25),
            # Pure overheating does NOT produce high vibration; a high-vibration reading
            # incurs a direction-mismatch penalty and shifts diagnosis to BEARING_FAILURE.
            "vibration_mms":   ("low",  0.30),
        },
        severity_level="high",
        urgency="immediate",
        recommended_actions=[
            "Shut down machine to prevent insulation damage.",
            "Inspect and clean ventilation ducts and cooling fins.",
            "Verify ambient temperature and cooling system operation.",
            "Check motor load profile — possible overloading.",
            "Test winding insulation resistance after cooldown.",
            "Review thermal protection relay settings.",
        ],
        typical_progression=(
            "Temperature climbs steadily; vibration remains relatively normal. "
            "Winding insulation degrades if sustained, risking motor burnout."
        ),
    ),
    FaultDefinition(
        fault_id="CAVITATION",
        name="Pump Cavitation",
        description=(
            "Formation and collapse of vapour bubbles inside the pump casing, "
            "caused by localised low pressure. Produces characteristic noise and "
            "erodes impeller surfaces."
        ),
        signal_signatures={
            "pressure_bar":    ("low",  0.90),
            "vibration_mms":   ("high", 0.80),
            "current_a":       ("low",  0.45),
        },
        severity_level="high",
        urgency="schedule",
        recommended_actions=[
            "Increase suction-side pressure (raise sump level or reduce suction losses).",
            "Inspect suction strainer / filter for blockage.",
            "Check NPSH available vs. NPSH required.",
            "Reduce pump speed or throttle delivery valve temporarily.",
            "Inspect impeller for erosion damage; replace if necessary.",
        ],
        typical_progression=(
            "Pressure drops on suction side; vibration and noise increase rapidly. "
            "Extended cavitation causes impeller pitting and eventual flow loss."
        ),
    ),
    FaultDefinition(
        fault_id="ELECTRICAL_OVERLOAD",
        name="Electrical Overload",
        description=(
            "Motor current exceeds rated value, indicating mechanical overloading, "
            "phase imbalance, or electrical supply issues."
        ),
        signal_signatures={
            "current_a":       ("high", 0.95),
            "temperature_c":   ("high", 0.45),  # secondary; current is primary differentiator
            "rpm":             ("low",  0.50),
        },
        severity_level="high",
        urgency="immediate",
        recommended_actions=[
            "Check and correct mechanical load — potential obstruction or jamming.",
            "Verify supply voltage and check for phase imbalance.",
            "Inspect motor windings for shorts or ground faults.",
            "Verify overload relay settings match motor nameplate FLA.",
            "Review duty cycle — machine may be cycling too frequently.",
        ],
        typical_progression=(
            "Current surges quickly; temperature rises in motor windings. "
            "Sustained overload trips thermal protection or burns windings."
        ),
    ),
    FaultDefinition(
        fault_id="ROTOR_IMBALANCE",
        name="Rotor Imbalance",
        description=(
            "Uneven mass distribution in the rotating assembly (rotor, impeller, "
            "coupling) causing 1× running-speed vibration that grows with speed. "
            "Temperature is typically normal; elevated temp suggests bearing damage instead."
        ),
        signal_signatures={
            "vibration_mms":   ("high", 0.90),
            "rpm":             ("any",  0.20),
            # Expects temperature to be normal; if it is high, direction mismatch
            # penalises this candidate and promotes BEARING_FAILURE instead.
            "temperature_c":   ("low",  0.30),
        },
        severity_level="medium",
        urgency="schedule",
        recommended_actions=[
            "Perform dynamic balancing of rotor/impeller assembly.",
            "Inspect for material build-up on impeller (deposits, corrosion).",
            "Check coupling for wear or damage.",
            "Verify shaft straightness with dial gauge.",
            "Re-check alignment after any balancing work.",
        ],
        typical_progression=(
            "Vibration at 1× RPM builds progressively; bearing loads increase. "
            "Eventually causes bearing fatigue and shaft fatigue cracks if unaddressed."
        ),
    ),
    FaultDefinition(
        fault_id="NORMAL_OPERATION",
        name="Normal Operation",
        description="All signals are within expected operating ranges. No fault detected.",
        signal_signatures={},
        severity_level="low",
        urgency="monitor",
        recommended_actions=[
            "Continue normal monitoring schedule.",
            "Log reading for trend analysis.",
        ],
        typical_progression="Machine is operating within design parameters.",
    ),
]

FAULT_KB_MAP: Dict[str, FaultDefinition] = {f.fault_id: f for f in FAULT_KB}


# ---------------------------------------------------------------------------
# Candidate fault result
# ---------------------------------------------------------------------------

@dataclass
class FaultCandidate:
    fault: FaultDefinition
    match_score: float          # 0-1, higher = better match
    matched_signals: List[str]  # which signals matched
    unmatched_signals: List[str]
    evidence_summary: str


# ---------------------------------------------------------------------------
# Signal Investigator
# ---------------------------------------------------------------------------

class SignalInvestigator:
    """
    Correlates anomalous signals from the AnomalyReport with the fault
    knowledge base and returns ranked FaultCandidate objects.
    """

    def investigate(self, report: AnomalyReport) -> List[FaultCandidate]:
        """
        Returns a ranked list of FaultCandidate objects (most likely first).
        Always includes NORMAL_OPERATION as the null hypothesis.
        """
        if not report.is_anomalous:
            return [self._normal_candidate()]

        flagged_map: Dict[str, SignalAnomaly] = {
            fa.signal: fa for fa in report.flagged_signals
        }
        candidates: List[FaultCandidate] = []

        for fault in FAULT_KB:
            if fault.fault_id == "NORMAL_OPERATION":
                continue
            score, matched, unmatched = self._score_fault(fault, flagged_map)
            if score > 0:
                evidence = self._build_evidence(fault, matched, flagged_map)
                candidates.append(FaultCandidate(
                    fault=fault,
                    match_score=round(score, 4),
                    matched_signals=matched,
                    unmatched_signals=unmatched,
                    evidence_summary=evidence,
                ))

        # Sort by match score descending
        candidates.sort(key=lambda c: c.match_score, reverse=True)

        # If no fault matched at all, insert normal as fallback
        if not candidates:
            candidates.append(self._normal_candidate())

        return candidates

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _score_fault(
        self,
        fault: FaultDefinition,
        flagged: Dict[str, SignalAnomaly],
    ) -> Tuple[float, List[str], List[str]]:
        """
        Compute a match score for a fault given the flagged signals.

        Score = weighted sum of matched signals / total possible weighted sum
        Penalised slightly for each expected signal that did NOT appear.
        """
        if not fault.signal_signatures:
            return 0.0, [], []

        total_weight = sum(w for _, w in fault.signal_signatures.values())
        earned = 0.0
        matched, unmatched = [], []

        for signal, (direction, weight) in fault.signal_signatures.items():
            if signal in flagged:
                fa = flagged[signal]
                # Check direction match
                if direction == "any":
                    dir_match = True
                else:
                    dir_match = fa.direction == direction
                if dir_match:
                    # Scale by severity: critical gives full weight, warn gives 0.7
                    sev_multiplier = 1.0 if fa.severity == "critical" else 0.7
                    earned += weight * sev_multiplier
                    matched.append(signal)
                else:
                    # Direction mismatch — slight penalty
                    earned -= weight * 0.2
                    unmatched.append(signal)
            else:
                unmatched.append(signal)

        score = max(0.0, earned / (total_weight or 1.0))
        return score, matched, unmatched

    @staticmethod
    def _build_evidence(
        fault: FaultDefinition,
        matched_signals: List[str],
        flagged: Dict[str, SignalAnomaly],
    ) -> str:
        parts = []
        for sig in matched_signals:
            if sig in flagged:
                parts.append(flagged[sig].description)
        return "; ".join(parts) if parts else "Pattern match based on signal combination."

    @staticmethod
    def _normal_candidate() -> FaultCandidate:
        normal = FAULT_KB_MAP["NORMAL_OPERATION"]
        return FaultCandidate(
            fault=normal,
            match_score=1.0,
            matched_signals=[],
            unmatched_signals=[],
            evidence_summary="All signals within normal operating parameters.",
        )
