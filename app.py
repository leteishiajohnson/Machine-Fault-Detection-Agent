"""
app.py — Streamlit application for the Machine Fault Diagnosis Agent.

Run with:
    streamlit run app.py

Pages:
    1. Dashboard           — Machine health overview
    2. Sensor Monitoring   — Live/simulated sensor charts + CSV upload
    3. AI Diagnosis Chat   — Conversational fault diagnosis
    4. Fault Detection     — Anomaly results and fault candidates
    5. Maintenance         — Recommendations and safety warnings
    6. Agent Trace         — 7-step pipeline execution log
    7. Settings            — watsonx.ai configuration status
"""

from __future__ import annotations

import io
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

# ---------------------------------------------------------------------------
# Path setup — allow running from any directory
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent

# Add the directory that contains app.py to sys.path so that `src` is importable.
# Also check for a `machine_fault_agent/` sub-directory in case the repo was
# cloned and the user is running from the repo root rather than the project folder.
def _find_src_root() -> Path:
    """Return the directory that directly contains the `src/` package."""
    # 1. Same directory as app.py  (normal case)
    if (_HERE / "src").is_dir():
        return _HERE
    # 2. machine_fault_agent/ subdirectory  (running from repo root)
    candidate = _HERE / "machine_fault_agent"
    if (candidate / "src").is_dir():
        return candidate
    # 3. Parent directory  (app.py was moved up one level)
    parent = _HERE.parent
    if (parent / "src").is_dir():
        return parent
    # Fallback — use app.py's own directory and let the import fail with a
    # clear message rather than a silent wrong-path error.
    return _HERE

_SRC_ROOT = _find_src_root()
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

# ---------------------------------------------------------------------------
# Load .env if present
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    # Look for .env next to app.py first, then next to the src/ root
    _env_path = _HERE / ".env" if (_HERE / ".env").exists() else _SRC_ROOT / ".env"
    load_dotenv(_env_path)
except ImportError:
    pass

import streamlit as st

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from src.agent import DiagnosisAgent, AgentRun
from src.anomaly_detection import AnomalyDetector, SIGNAL_NORMAL_RANGES
from src.fault_knowledge_base import FAULT_KB, SignalInvestigator
from src.llm_reasoner import DiagnosisResult, RuleBasedReasoner
from src.sensor_data import (
    SensorDataLoader,
    SensorReading,
    SensorSimulator,
    SIGNAL_UNITS,
)
from src.watsonx_reasoner import get_watsonx_config_status

# ---------------------------------------------------------------------------
# Page configuration (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Machine Fault Diagnosis Agent",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS overrides — polished, industrial look
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Global typography */
    html, body, [class*="css"] { font-family: 'Segoe UI', system-ui, sans-serif; }

    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #0f1923; }
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span { color: #c9d1d9 !important; }

    /* Status badges */
    .badge-ok    { background:#1a7a4a; color:#fff; padding:2px 10px; border-radius:12px; font-size:0.8rem; }
    .badge-warn  { background:#b88a00; color:#fff; padding:2px 10px; border-radius:12px; font-size:0.8rem; }
    .badge-crit  { background:#c0392b; color:#fff; padding:2px 10px; border-radius:12px; font-size:0.8rem; }
    .badge-info  { background:#2471a3; color:#fff; padding:2px 10px; border-radius:12px; font-size:0.8rem; }

    /* Step timeline */
    .step-ok     { border-left: 4px solid #27ae60; padding-left: 10px; margin-bottom: 6px; }
    .step-warn   { border-left: 4px solid #f39c12; padding-left: 10px; margin-bottom: 6px; }
    .step-error  { border-left: 4px solid #e74c3c; padding-left: 10px; margin-bottom: 6px; }
    .step-skip   { border-left: 4px solid #95a5a6; padding-left: 10px; margin-bottom: 6px; }

    /* Metric cards — reduce the oversized default font */
    div[data-testid="metric-container"] {
        background: #f7f8fa;
        border-radius: 8px;
        padding: 8px 10px;
    }
    /* Label (top line) */
    div[data-testid="metric-container"] label {
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #57606a !important;
    }
    /* Value (big number / text) */
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] > div {
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        line-height: 1.3 !important;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    /* Delta text */
    div[data-testid="metric-container"] div[data-testid="stMetricDelta"] {
        font-size: 0.75rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEVERITY_EMOJI = {
    "low":      "🟢",
    "medium":   "🟡",
    "high":     "🔴",
    "critical": "💥",
    "unknown":  "⚪",
}
URGENCY_EMOJI = {
    "monitor":   "👁️",
    "schedule":  "📅",
    "immediate": "🚨",
}
STATUS_COLOR = {
    "ok":      "#27ae60",
    "warn":    "#f39c12",
    "error":   "#e74c3c",
    "skipped": "#95a5a6",
}

SIMULATED_NOTE = (
    "⚠️ **Simulated data** — these readings are synthetically generated "
    "for demonstration purposes and do not represent a real machine."
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
def _init_state() -> None:
    defaults = {
        "readings": None,          # List[SensorReading] currently loaded
        "agent_run": None,         # AgentRun from last execution
        "scenario": "bearing_failure",
        "machine_id": "PUMP-01",
        "chat_history": [],        # list of {"role": "user"|"assistant", "content": str}
        "data_source": "simulated",  # "simulated" | "csv"
        "sim_seed": 42,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
def render_sidebar() -> str:
    with st.sidebar:
        st.markdown("## ⚙️ Machine Fault\nDiagnosis Agent")
        st.divider()
        page = st.radio(
            "Navigate",
            options=[
                "🏠 Dashboard",
                "📡 Sensor Monitoring",
                "💬 AI Diagnosis Chat",
                "🔍 Fault Detection",
                "🔧 Maintenance",
                "🗂️ Agent Trace",
                "⚙️ Settings",
            ],
            label_visibility="collapsed",
        )
        st.divider()

        # Quick machine selector
        st.markdown("**Machine**")
        st.session_state.machine_id = st.text_input(
            "Machine ID",
            value=st.session_state.machine_id,
            label_visibility="collapsed",
        )

        # Connection indicator
        cfg = get_watsonx_config_status()
        if cfg["configured"]:
            model_short = cfg["model_id"].replace("ibm/", "")
            st.markdown(
                f'<span class="badge-ok">✓ Granite · {model_short}</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span class="badge-warn">⚠ Rule-based mode</span>',
                unsafe_allow_html=True,
            )
        st.caption("See Settings to configure watsonx.ai")

    return page


# ---------------------------------------------------------------------------
# Helper: get or generate readings
# ---------------------------------------------------------------------------
def _get_readings() -> Optional[List[SensorReading]]:
    return st.session_state.get("readings")


def _load_simulated(scenario: str, machine_id: str, seed: int = 42) -> List[SensorReading]:
    sim = SensorSimulator(seed=seed)
    return sim.generate(scenario, machine_id=machine_id, n_readings=20)


def _run_agent(readings: List[SensorReading]) -> AgentRun:
    """Run the full diagnosis pipeline; cache result in session state."""
    agent = DiagnosisAgent(use_llm=True)
    run = agent.run(readings)
    st.session_state.agent_run = run
    return run


# ===========================================================================
# PAGE 1 — Dashboard
# ===========================================================================
def page_dashboard() -> None:
    st.title("⚙️ Machine Fault Diagnosis Agent")
    st.caption("IBM Granite · watsonx.ai · Real-time industrial fault diagnosis")
    st.divider()

    # Load / trigger data
    col_ctrl, col_info = st.columns([2, 3])
    with col_ctrl:
        st.markdown("### Quick Diagnosis")
        scenario = st.selectbox(
            "Fault scenario",
            SensorSimulator.SCENARIOS,
            index=SensorSimulator.SCENARIOS.index(st.session_state.scenario),
            key="dash_scenario",
        )
        if st.button("▶ Run Diagnosis", type="primary", use_container_width=True):
            with st.spinner("Running agent pipeline…"):
                readings = _load_simulated(scenario, st.session_state.machine_id)
                st.session_state.readings = readings
                st.session_state.scenario = scenario
                st.session_state.data_source = "simulated"
                run = _run_agent(readings)
                st.session_state.agent_run = run

    run: Optional[AgentRun] = st.session_state.agent_run

    with col_info:
        if run:
            dx = run.diagnosis
            sev = dx.severity if dx else "unknown"
            urg = dx.urgency if dx else "monitor"
            st.markdown("### Machine Status")
            s1, s2, s3 = st.columns(3)
            s1.metric("Machine", run.machine_id)
            s2.metric("Severity", f"{SEVERITY_EMOJI.get(sev, '⚪')} {sev.upper()}")
            s3.metric("Urgency", f"{URGENCY_EMOJI.get(urg, '')} {urg.upper()}")
        else:
            st.info("Select a scenario and click **Run Diagnosis** to begin.")

    if run is None:
        # Show a summary of what the agent can do
        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.markdown(
            "#### 🔍 Anomaly Detection\n"
            "Isolation Forest ML + statistical threshold analysis across 5 sensor signals."
        )
        c2.markdown(
            "#### 🧠 IBM Granite AI\n"
            "Structured reasoning via IBM watsonx.ai with rule-based fallback when offline."
        )
        c3.markdown(
            "#### 🔧 Maintenance Guidance\n"
            "Prioritised, fault-specific maintenance actions and safety precautions."
        )
        return

    dx = run.diagnosis
    st.divider()

    # KPI row
    col1, col2, col3, col4 = st.columns(4)
    if dx:
        col1.metric("Fault Diagnosed", dx.fault_name)
        col2.metric("Confidence", dx.confidence.upper())
        col3.metric("Agent Steps", len(run.steps))
        col4.metric("Duration", f"{run.duration_ms:.0f} ms")
    else:
        col1.metric("Status", "No diagnosis")

    # Sensor snapshot
    st.markdown("### Sensor Snapshot (latest reading)")
    if run.anomaly_report:
        latest = run.anomaly_report.latest_reading
        flagged = {fa.signal: fa for fa in run.anomaly_report.flagged_signals}
        cols = st.columns(len(latest))
        for i, (sig, val) in enumerate(latest.items()):
            unit = SIGNAL_UNITS.get(sig, "")
            lo, hi = SIGNAL_NORMAL_RANGES.get(sig, (None, None))
            fa = flagged.get(sig)
            delta_str = None
            if fa:
                delta_str = f"{'▲' if fa.direction == 'high' else '▼'} {fa.severity.upper()}"
            cols[i].metric(
                label=sig.replace("_", " ").title(),
                value=f"{val:.1f} {unit}",
                delta=delta_str,
                delta_color="inverse" if fa else "normal",
            )

    if st.session_state.data_source == "simulated":
        st.caption(SIMULATED_NOTE)

    # Quick recommendation preview
    if dx and dx.recommended_actions:
        st.markdown("### Top Recommendation")
        st.info(f"**{dx.recommended_actions[0]}**")
        if len(dx.recommended_actions) > 1:
            st.caption(f"+ {len(dx.recommended_actions) - 1} more — see Maintenance page")


# ===========================================================================
# PAGE 2 — Sensor Monitoring
# ===========================================================================
def page_sensor_monitoring() -> None:
    import pandas as pd

    st.title("📡 Sensor Monitoring")
    st.caption("Live-simulated sensor readings with anomaly threshold overlays")

    tab_sim, tab_csv = st.tabs(["🔁 Simulated Data", "📂 CSV Upload"])

    # ---- Simulated Data Tab ------------------------------------------------
    with tab_sim:
        col_l, col_r = st.columns([1, 3])
        with col_l:
            scenario = st.selectbox(
                "Fault Scenario",
                SensorSimulator.SCENARIOS,
                key="mon_scenario",
            )
            machine_id = st.text_input("Machine ID", value="PUMP-01", key="mon_mid")
            n_readings = st.slider("Number of readings", 8, 40, 20, key="mon_n")
            seed = st.number_input("Simulation seed", value=42, key="mon_seed")

            if st.button("Load Readings", use_container_width=True, key="mon_load"):
                readings = _load_simulated(scenario, machine_id, seed=int(seed))
                st.session_state.readings = readings
                st.session_state.data_source = "simulated"
                st.session_state.scenario = scenario
                st.session_state.machine_id = machine_id

        with col_r:
            readings = _get_readings()
            if readings is None:
                st.info("Click **Load Readings** to generate simulated sensor data.")
            else:
                _render_sensor_charts(readings, source="simulated")

    # ---- CSV Upload Tab ----------------------------------------------------
    with tab_csv:
        st.markdown(
            "Upload a CSV file with columns: "
            "`machine_id`, `timestamp`, `temperature_c`, `vibration_mms`, "
            "`pressure_bar`, `current_a`, `rpm`"
        )
        uploaded = st.file_uploader("Choose CSV file", type=["csv"], key="csv_upload")
        if uploaded:
            try:
                text = uploaded.read().decode("utf-8")
                readings = SensorDataLoader.from_csv_string(text)
                st.session_state.readings = readings
                st.session_state.data_source = "csv"
                st.success(f"Loaded {len(readings)} readings from {uploaded.name}")
                _render_sensor_charts(readings, source="csv")
            except Exception as exc:
                st.error(f"Failed to parse CSV: {exc}")

        # Sample CSV download
        sample_csv = _make_sample_csv()
        st.download_button(
            "⬇ Download sample CSV",
            data=sample_csv,
            file_name="sample_sensor_data.csv",
            mime="text/csv",
        )


def _render_sensor_charts(readings: List[SensorReading], source: str) -> None:
    import pandas as pd

    if source == "simulated":
        st.caption(SIMULATED_NOTE)

    df = pd.DataFrame([
        {
            "timestamp": r.timestamp,
            "temperature_c": r.temperature_c,
            "vibration_mms": r.vibration_mms,
            "pressure_bar":  r.pressure_bar,
            "current_a":     r.current_a,
            "rpm":           r.rpm,
        }
        for r in readings
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["reading #"] = df.index + 1

    signals = [
        ("vibration_mms",  "Vibration (mm/s)",       (0.5, 4.5),   "🔴"),
        ("temperature_c",  "Temperature (°C)",        (60.0, 85.0), "🟠"),
        ("current_a",      "Motor Current (A)",       (18.0, 28.0), "🟡"),
        ("pressure_bar",   "Discharge Pressure (bar)",(3.5, 7.0),   "🔵"),
        ("rpm",            "Rotational Speed (RPM)",  (1440, 1510), "🟢"),
    ]

    for sig, label, (lo, hi), dot in signals:
        with st.expander(f"{dot} {label}", expanded=(sig in ("vibration_mms", "temperature_c"))):
            chart_df = df[["reading #", sig]].rename(columns={sig: label})
            st.line_chart(chart_df.set_index("reading #"), height=180)

            latest_val = df[sig].iloc[-1]
            mean_val   = df[sig].mean()
            max_val    = df[sig].max()

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Latest", f"{latest_val:.2f}")
            m2.metric("Mean",   f"{mean_val:.2f}")
            m3.metric("Max",    f"{max_val:.2f}")
            status = "✅ Normal" if lo <= latest_val <= hi else "⚠️ ANOMALY"
            m4.metric("Status", status)
            st.caption(f"Normal range: {lo} – {hi} {SIGNAL_UNITS.get(sig, '')}")


def _make_sample_csv() -> str:
    sim = SensorSimulator(seed=1)
    readings = sim.generate("bearing_failure", machine_id="PUMP-SAMPLE", n_readings=10)
    lines = ["machine_id,timestamp,temperature_c,vibration_mms,pressure_bar,current_a,rpm"]
    for r in readings:
        lines.append(
            f"{r.machine_id},{r.timestamp.isoformat()},"
            f"{r.temperature_c:.2f},{r.vibration_mms:.2f},"
            f"{r.pressure_bar:.2f},{r.current_a:.2f},{r.rpm:.2f}"
        )
    return "\n".join(lines)


# ===========================================================================
# PAGE 3 — AI Diagnosis Chat
# ===========================================================================
def page_chat() -> None:
    st.title("💬 AI Diagnosis Chat")
    cfg = get_watsonx_config_status()
    if cfg["configured"]:
        model_name = cfg["model_id"]
        st.caption(f"Powered by **IBM Granite** via watsonx.ai · Model: `{model_name}`")
    else:
        st.caption("Running in **rule-based mode** — configure watsonx.ai for AI-powered responses")

    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Suggested questions
    if not st.session_state.chat_history:
        st.markdown("**Suggested questions:**")
        suggestions = [
            "Why is my CNC machine vibrating too much?",
            "What does high temperature in the motor mean?",
            "How do I fix pump cavitation?",
            "Is electrical overload dangerous?",
            "What are the signs of bearing failure?",
        ]
        cols = st.columns(len(suggestions))
        for i, q in enumerate(suggestions):
            if cols[i].button(q, key=f"sugg_{i}", use_container_width=True):
                _handle_chat_message(q)
                st.rerun()

    # Chat input
    if prompt := st.chat_input("Ask about machine faults, sensor readings, or maintenance…"):
        _handle_chat_message(prompt)
        st.rerun()


def _handle_chat_message(user_msg: str) -> None:
    """Process a user chat message and generate an agent-backed response."""
    st.session_state.chat_history.append({"role": "user", "content": user_msg})

    # Use current readings if available, else use a default scenario
    readings = _get_readings()
    if readings is None:
        # Infer scenario from question keywords
        scenario = _infer_scenario(user_msg)
        readings = _load_simulated(scenario, st.session_state.machine_id)

    try:
        run = _run_agent(readings)
        response = _format_chat_response(run, user_msg)
    except Exception as exc:
        response = (
            f"I encountered an issue running the diagnostic pipeline: {exc}\n\n"
            "Please check the sensor data and try again."
        )

    st.session_state.chat_history.append({"role": "assistant", "content": response})


def _infer_scenario(text: str) -> str:
    """Map question keywords to a relevant fault scenario."""
    text_lower = text.lower()
    if any(w in text_lower for w in ["vibrat", "imbalance", "shake", "cnc"]):
        return "rotor_imbalance"
    if any(w in text_lower for w in ["bearing", "seiz", "grease"]):
        return "bearing_failure"
    if any(w in text_lower for w in ["heat", "hot", "temperature", "overheat", "cool"]):
        return "overheating"
    if any(w in text_lower for w in ["cavitat", "pressure", "pump", "bubble"]):
        return "cavitation"
    if any(w in text_lower for w in ["current", "electric", "overload", "amp", "motor"]):
        return "electrical_overload"
    return "bearing_failure"


def _format_chat_response(run: AgentRun, question: str) -> str:
    """Format a diagnostic result as a conversational response."""
    dx = run.diagnosis
    if dx is None:
        return "The diagnostic pipeline could not produce a result. Please check the sensor data."

    sev_emo = SEVERITY_EMOJI.get(dx.severity, "⚪")
    urg_emo = URGENCY_EMOJI.get(dx.urgency, "")

    # Detect which reasoner was used
    reason_step = next((s for s in run.steps if s.step_name == "REASON"), None)
    mode = reason_step.details.get("mode", "rule-based") if reason_step else "rule-based"
    model_tag = ""
    if mode == "watsonx":
        model_id = reason_step.details.get("model_id", "Granite")
        model_tag = f"\n\n*Analysis powered by IBM Granite (`{model_id}`) via watsonx.ai.*"
    elif mode == "openai":
        model_tag = "\n\n*Analysis powered by OpenAI LLM.*"
    else:
        model_tag = "\n\n*Analysis performed by rule-based engine (watsonx.ai not configured).*"

    lines = [
        f"## {sev_emo} Diagnosis: {dx.fault_name}",
        f"",
        f"**Confidence:** {dx.confidence.upper()} &nbsp;|&nbsp; "
        f"**Severity:** {dx.severity.upper()} &nbsp;|&nbsp; "
        f"**Urgency:** {urg_emo} {dx.urgency.upper()}",
        f"",
        f"### Explanation",
        dx.explanation,
    ]

    if dx.evidence:
        lines += ["", "### Sensor Evidence"]
        for e in dx.evidence:
            lines.append(f"- {e}")

    if dx.differential_diagnosis:
        lines += ["", "### Other Considered Faults"]
        for d in dx.differential_diagnosis:
            lines.append(f"- {d}")

    if dx.uncertainty_notes:
        lines += ["", f"⚠️ **Uncertainty:** {dx.uncertainty_notes}"]

    lines += ["", "### Recommended Actions"]
    for i, a in enumerate(dx.recommended_actions, 1):
        lines.append(f"{i}. {a}")

    lines.append(model_tag)
    return "\n".join(lines)


# ===========================================================================
# PAGE 4 — Fault Detection
# ===========================================================================
def page_fault_detection() -> None:
    st.title("🔍 Fault Detection")
    st.caption("Anomaly detection results and fault candidate ranking")

    run: Optional[AgentRun] = st.session_state.agent_run
    if run is None:
        st.info("Run a diagnosis from the **Dashboard** or **Sensor Monitoring** page first.")
        return

    report = run.anomaly_report
    if report is None:
        st.warning("Anomaly report not available. The detection step may have failed.")
        return

    # ---- Anomaly summary ---------------------------------------------------
    st.markdown("### Anomaly Detection Result")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Anomaly Detected", "YES ⚠️" if report.is_anomalous else "NO ✅")
    col2.metric("Anomaly Score", f"{report.anomaly_score:.3f}",
                help="0 = normal, 1 = most anomalous (Isolation Forest + thresholds)")
    col3.metric("Readings Analysed", report.n_readings)
    col4.metric("Flagged Signals", len(report.flagged_signals))

    # Anomaly score bar
    score_pct = int(report.anomaly_score * 100)
    score_color = "#27ae60" if score_pct < 40 else "#f39c12" if score_pct < 65 else "#e74c3c"
    st.markdown(
        f"""
        <div style="background:#e0e0e0; border-radius:8px; height:22px; margin:8px 0;">
          <div style="background:{score_color}; width:{score_pct}%; height:22px;
                      border-radius:8px; display:flex; align-items:center;
                      padding-left:8px; color:white; font-weight:bold; font-size:13px;">
            {score_pct}%
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---- Flagged signals table ---------------------------------------------
    if report.flagged_signals:
        st.markdown("### Flagged Signals")
        import pandas as pd
        rows = []
        for fa in report.flagged_signals:
            lo, hi = SIGNAL_NORMAL_RANGES.get(fa.signal, (None, None))
            unit = SIGNAL_UNITS.get(fa.signal, "")
            rows.append({
                "Signal":    fa.signal.replace("_", " ").title(),
                "Value":     f"{fa.value:.2f} {unit}",
                "Direction": f"{'▲ HIGH' if fa.direction == 'high' else '▼ LOW'}",
                "Normal Range": f"{lo} – {hi} {unit}",
                "Z-Score":   f"{fa.z_score:+.1f}σ",
                "Severity":  fa.severity.upper(),
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.success("No individual signals outside normal operating ranges.")

    # ---- Fault candidates --------------------------------------------------
    st.markdown("### Fault Candidates")
    st.caption(
        "Match scores are based on how well the flagged signals align with known fault patterns. "
        "They are **not** statistical probabilities."
    )

    if run.fault_candidates:
        import pandas as pd
        rows = []
        for c in run.fault_candidates:
            rows.append({
                "Fault":        c.fault.name,
                "Match Score":  f"{c.match_score:.0%}",
                "Matched":      ", ".join(c.matched_signals) or "—",
                "Severity":     c.fault.severity_level.upper(),
                "Urgency":      c.fault.urgency.upper(),
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Visual score bars
        st.markdown("#### Score Comparison")
        for c in run.fault_candidates[:5]:
            pct = int(c.match_score * 100)
            color = "#e74c3c" if c.fault.severity_level in ("high", "critical") else "#3498db"
            st.markdown(
                f"<div style='display:flex; align-items:center; margin:4px 0;'>"
                f"<span style='width:220px; font-size:0.85rem;'>{c.fault.name}</span>"
                f"<div style='background:#e0e0e0; border-radius:6px; flex:1; height:18px;'>"
                f"<div style='background:{color}; width:{pct}%; height:18px; border-radius:6px;'></div>"
                f"</div>"
                f"<span style='width:50px; text-align:right; font-size:0.85rem;'>{pct}%</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
    else:
        st.info("No fault candidates identified.")


# ===========================================================================
# PAGE 5 — Maintenance Recommendations
# ===========================================================================
def page_maintenance() -> None:
    st.title("🔧 Maintenance Recommendations")

    run: Optional[AgentRun] = st.session_state.agent_run
    if run is None:
        st.info("Run a diagnosis from the **Dashboard** first.")
        return

    dx = run.diagnosis
    if dx is None:
        st.warning("No diagnosis available.")
        return

    # ---- Safety disclaimer -------------------------------------------------
    st.error(
        "⛔ **SAFETY NOTICE** — This application provides decision-support information only. "
        "It does **not** and **cannot** automatically control, start, or stop any machinery. "
        "Always follow your site's lock-out/tag-out procedures. "
        "A qualified maintenance engineer must review and authorise all work."
    )

    # ---- Diagnosis summary -------------------------------------------------
    sev_emo = SEVERITY_EMOJI.get(dx.severity, "⚪")
    urg_emo = URGENCY_EMOJI.get(dx.urgency, "")
    col1, col2, col3 = st.columns(3)
    col1.metric("Fault", dx.fault_name)
    col2.metric("Severity", f"{sev_emo} {dx.severity.upper()}")
    col3.metric("Urgency", f"{urg_emo} {dx.urgency.upper()}")

    if dx.urgency == "immediate":
        st.error(
            "🚨 **IMMEDIATE ACTION REQUIRED** — Continued operation risks equipment damage "
            "or personal injury. Stop the machine and isolate it before inspection."
        )
    elif dx.urgency == "schedule":
        st.warning(
            "📅 **Schedule maintenance** — Arrange inspection within your next planned "
            "maintenance window. Monitor closely in the meantime."
        )
    else:
        st.success(
            "👁️ **Monitor** — No immediate action required. "
            "Continue routine monitoring schedule."
        )

    # ---- Recommended actions -----------------------------------------------
    st.markdown("### Recommended Actions")
    for i, action in enumerate(dx.recommended_actions, 1):
        is_safety = any(
            w in action.lower()
            for w in ["stop", "shut", "isolat", "lock", "safe", "inspect", "halt", "power off"]
        )
        icon = "⛔" if is_safety else f"{i}."
        st.markdown(f"**{icon}** {action}")

    # ---- Explanation -------------------------------------------------------
    with st.expander("📖 Fault Explanation", expanded=True):
        st.markdown(dx.explanation)

    # ---- Evidence ----------------------------------------------------------
    if dx.evidence:
        with st.expander("📊 Supporting Evidence"):
            for e in dx.evidence:
                st.markdown(f"- {e}")

    # ---- Uncertainty -------------------------------------------------------
    if dx.uncertainty_notes:
        st.warning(f"⚠️ **Uncertainty:** {dx.uncertainty_notes}")

    # ---- When to call a technician -----------------------------------------
    st.markdown("### When to Call a Qualified Technician")
    st.info(
        "You should engage a qualified maintenance technician or engineer when:\n\n"
        "- The fault severity is **HIGH** or **CRITICAL**\n"
        "- Urgency is marked as **IMMEDIATE** or **SCHEDULE**\n"
        "- You are uncertain about any diagnosis or recommended action\n"
        "- Any safety-related work is required (electrical isolation, pressure systems)\n"
        "- The fault recurs after maintenance\n"
        "- Multiple faults are detected simultaneously\n\n"
        "This application is a diagnostic aid, not a substitute for professional judgement."
    )

    # ---- KB info for this fault --------------------------------------------
    fault_def = next(
        (f for f in FAULT_KB if f.fault_id == dx.fault_id), None
    )
    if fault_def and fault_def.typical_progression:
        with st.expander("📈 Typical Fault Progression"):
            st.markdown(fault_def.typical_progression)


# ===========================================================================
# PAGE 6 — Agent Execution Trace
# ===========================================================================
def page_agent_trace() -> None:
    st.title("🗂️ Agent Execution Trace")
    st.caption("7-step diagnostic pipeline: INGEST → DETECT → INVESTIGATE → REASON → DIAGNOSE → EXPLAIN → RECOMMEND")

    run: Optional[AgentRun] = st.session_state.agent_run
    if run is None:
        st.info("Run a diagnosis from the **Dashboard** first.")
        return

    # Run metadata
    col1, col2, col3 = st.columns(3)
    col1.metric("Run ID", run.run_id)
    col2.metric("Total Duration", f"{run.duration_ms:.0f} ms")
    col3.metric("Overall Status", "✅ Passed" if run.succeeded else "❌ Failed")

    st.markdown(f"**Machine:** {run.machine_id} &nbsp;|&nbsp; "
                f"**Started:** {run.started_at.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    st.divider()

    # Pipeline steps
    STEP_ICONS = {
        "INGEST":      "📥",
        "DETECT":      "🔎",
        "INVESTIGATE": "🔬",
        "REASON":      "🧠",
        "DIAGNOSE":    "📋",
        "EXPLAIN":     "📖",
        "RECOMMEND":   "🔧",
    }

    for step in run.steps:
        icon = STEP_ICONS.get(step.step_name, "•")
        color = STATUS_COLOR.get(step.status, "#95a5a6")
        css_class = f"step-{step.status}"

        with st.container():
            st.markdown(
                f"<div class='{css_class}' style='padding:8px 0;'>"
                f"<strong>{icon} {step.step_name}</strong> "
                f"&nbsp;·&nbsp; <span style='color:{color};font-weight:bold;'>{step.status.upper()}</span>"
                f"&nbsp;·&nbsp; <span style='color:#666;'>{step.duration_ms:.0f} ms</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.markdown(f"*{step.summary}*")

            # Show step details in expander
            if step.details:
                with st.expander(f"Details — {step.step_name}", expanded=False):
                    _render_step_details(step.step_name, step.details)

            st.markdown("---")

    # Visual pipeline progress bar
    st.markdown("### Pipeline Progress")
    total = len(run.steps)
    ok_count = sum(1 for s in run.steps if s.status in ("ok", "warn", "skipped"))
    pct = int((ok_count / total) * 100) if total else 0
    st.progress(pct / 100, text=f"{ok_count}/{total} steps completed")


def _render_step_details(step_name: str, details: dict) -> None:
    """Render step-specific details in a readable format."""
    if step_name == "INGEST":
        st.write(f"**Readings:** {details.get('n_readings', '?')}")
        st.write(f"**Machines:** {', '.join(details.get('machine_ids', []))}")
        tr = details.get("time_range", {})
        if tr:
            st.write(f"**Time range:** {tr.get('start', '?')} → {tr.get('end', '?')}")

    elif step_name == "DETECT":
        st.write(f"**Anomalous:** {details.get('is_anomalous', '?')}")
        st.write(f"**Anomaly score:** {details.get('anomaly_score', '?'):.4f}")
        flagged = details.get("flagged", [])
        if flagged:
            import pandas as pd
            df = pd.DataFrame(flagged)
            st.dataframe(df, use_container_width=True, hide_index=True)

    elif step_name == "INVESTIGATE":
        st.write(f"**Candidates found:** {details.get('n_candidates', '?')}")
        candidates = details.get("candidates", [])
        if candidates:
            import pandas as pd
            df = pd.DataFrame([
                {
                    "Fault": c["fault_name"],
                    "Score": f"{c['match_score']:.0%}",
                    "Signals": ", ".join(c.get("matched_signals", [])),
                }
                for c in candidates
            ])
            st.dataframe(df, use_container_width=True, hide_index=True)

    elif step_name == "REASON":
        st.write(f"**Mode:** {details.get('mode', '?')}")
        if details.get("model_id"):
            st.write(f"**Model:** {details['model_id']}")
        if details.get("reason"):
            st.write(f"**Fallback reason:** {details['reason']}")

    elif step_name in ("DIAGNOSE", "EXPLAIN"):
        for k, v in details.items():
            if k in ("explanation", "actions"):
                if isinstance(v, list):
                    for item in v:
                        st.write(f"- {item}")
                else:
                    st.write(v)
            elif k not in ("evidence", "recommended_actions", "differential_diagnosis"):
                st.write(f"**{k}:** {v}")

    elif step_name == "RECOMMEND":
        actions = details.get("actions", [])
        for i, a in enumerate(actions, 1):
            st.write(f"{i}. {a}")

    else:
        st.json(details)


# ===========================================================================
# PAGE 7 — Settings
# ===========================================================================
def page_settings() -> None:
    st.title("⚙️ Settings")
    st.caption("Application configuration status — credentials are never displayed here")

    # ---- watsonx.ai status -------------------------------------------------
    st.markdown("### IBM watsonx.ai Configuration")
    cfg = get_watsonx_config_status()

    mode = cfg.get("mode", "SaaS")
    configured = cfg.get("configured", False)

    if configured:
        st.success(f"✅ **{mode} mode** — watsonx.ai is configured and ready")
    else:
        st.warning(
            f"⚠️ **{mode} mode** — watsonx.ai is NOT fully configured. "
            "The application will use the rule-based reasoner."
        )

    # Status table — no credential values ever shown
    import pandas as pd
    if mode == "SaaS":
        rows = [
            ("WATSONX_API_KEY",    "IBM Cloud IAM API key",          "✅ Set" if cfg.get("api_key_set") else "❌ Not set"),
            ("WATSONX_USERNAME",   "IBM Cloud IAM username",         "✅ Set" if cfg.get("username_set") else "— not set (optional)"),
            ("WATSONX_PROJECT_ID", "watsonx.ai project ID",          "✅ Set" if cfg.get("project_id_set") else "❌ Not set"),
            ("WATSONX_URL",        "Regional endpoint URL",          cfg.get("url", "https://us-south.ml.cloud.ibm.com")),
            ("WATSONX_MODEL_ID",   "Granite model ID",               cfg.get("model_id", "ibm/granite-3-8b-instruct")),
            ("WATSONX_VERSION",    "API version (optional)",         cfg.get("version") or "not set"),
        ]
    else:
        rows = [
            ("WATSONX_CPD_URL",      "CP4D instance URL",   "✅ Set" if cfg.get("cpd_url_set") else "❌ Not set"),
            ("WATSONX_CPD_USERNAME", "CP4D username",       "✅ Set" if cfg.get("cpd_username_set") else "❌ Not set"),
            ("WATSONX_CPD_PASSWORD", "CP4D password",       "✅ Set" if cfg.get("cpd_password_set") else "❌ Not set"),
            ("WATSONX_PROJECT_ID",   "watsonx.ai project",  "✅ Set" if cfg.get("project_id_set") else "❌ Not set"),
            ("WATSONX_MODEL_ID",     "Granite model ID",    cfg.get("model_id", "ibm/granite-3-8b-instruct")),
            ("WATSONX_VERSION",      "CP4D platform version",        cfg.get("version", "4.8")),
        ]

    df = pd.DataFrame(rows, columns=["Variable", "Purpose", "Status"])
    st.dataframe(df, use_container_width=True, hide_index=True)

    # ---- Connection test ---------------------------------------------------
    st.markdown("### Connection Test")
    st.caption(
        "Sends a minimal test request to verify the watsonx.ai endpoint is reachable. "
        "This does NOT display any credentials."
    )
    if not configured:
        st.info(
            "Configure `WATSONX_API_KEY` and `WATSONX_PROJECT_ID` in your `.env` file "
            "before testing the connection."
        )
    else:
        if st.button("🔌 Test Connection", type="primary"):
            with st.spinner("Testing IBM watsonx.ai connection…"):
                from src.watsonx_reasoner import test_watsonx_connection
                result = test_watsonx_connection()
            if result["success"]:
                st.success(f"✅ {result['message']} (Model: `{result['model_id']}`)")
            else:
                st.error(f"❌ {result['message']}")

    # ---- How to configure --------------------------------------------------
    with st.expander("📋 How to configure watsonx.ai"):
        st.markdown(
            """
            1. **Copy the example file:**
               ```bash
               cp .env.example .env
               ```

            2. **Open `.env`** and fill in your credentials:
               - `WATSONX_API_KEY` — IBM Cloud IAM API key
                 ([Get one here](https://cloud.ibm.com/iam/apikeys))
               - `WATSONX_PROJECT_ID` — from your watsonx.ai project settings
               - `WATSONX_URL` — your regional endpoint
                 (default: `https://us-south.ml.cloud.ibm.com`)

            3. **Restart** `streamlit run app.py`

            4. **Verify** using the connection test above.

            ⚠️ Never commit your `.env` file. It is included in `.gitignore`.
            """
        )

    # ---- Reasoner priority info -------------------------------------------
    with st.expander("🧠 Reasoner priority"):
        st.markdown(
            """
            The agent selects a reasoner in this order:

            | Priority | Reasoner | Requires |
            |---|---|---|
            | 1 | **IBM Granite (watsonx.ai)** | `WATSONX_API_KEY` + `WATSONX_PROJECT_ID` |
            | 2 | **OpenAI-compatible LLM** | `OPENAI_API_KEY` |
            | 3 | **Rule-based fallback** | Nothing — always available |

            The rule-based fallback uses deterministic signal-matching against a fault
            knowledge base. It produces correct diagnoses for all known fault scenarios
            even without any API credentials.
            """
        )

    # ---- Application info ---------------------------------------------------
    st.markdown("### Application Info")
    c1, c2 = st.columns(2)
    c1.markdown(
        "**Machine Fault Diagnosis Agent**  \n"
        "Version: 2.0.0  \n"
        "Python: " + sys.version.split()[0]
    )
    c2.markdown(
        "**Backend modules:**  \n"
        "- `src/agent.py` — 7-step pipeline  \n"
        "- `src/watsonx_reasoner.py` — IBM Granite  \n"
        "- `src/anomaly_detection.py` — Isolation Forest  \n"
        "- `src/fault_knowledge_base.py` — 5 fault types  "
    )


# ===========================================================================
# Main router
# ===========================================================================
def main() -> None:
    page = render_sidebar()

    if page.startswith("🏠"):
        page_dashboard()
    elif page.startswith("📡"):
        page_sensor_monitoring()
    elif page.startswith("💬"):
        page_chat()
    elif page.startswith("🔍"):
        page_fault_detection()
    elif page.startswith("🔧"):
        page_maintenance()
    elif page.startswith("🗂️"):
        page_agent_trace()
    elif page.startswith("⚙️"):
        page_settings()


if __name__ == "__main__":
    main()
