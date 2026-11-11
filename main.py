"""
main.py — CLI demo runner for the Machine Fault Diagnosis Agent.

Usage:
  python main.py                          # run all 6 demo scenarios
  python main.py --scenario bearing_failure
  python main.py --scenario normal
  python main.py --csv path/to/data.csv   # diagnose from a CSV file
  python main.py --no-llm                 # use rule-based reasoner (no API key needed)

Environment variables:
  OPENAI_API_KEY   — your OpenAI key (or watsonx-compat API key)
  OPENAI_API_BASE  — base URL override (e.g. watsonx endpoint)
  LLM_MODEL        — model name (default: gpt-4o-mini)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Rich for pretty CLI output
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich import box
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

# Add parent dir to path so we can run: python main.py from project root
sys.path.insert(0, str(Path(__file__).parent))

from src.agent import DiagnosisAgent, AgentRun, StepResult, _format_explanation
from src.sensor_data import SensorSimulator, SensorDataLoader

console = Console() if HAS_RICH else None


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

SEVERITY_COLOR = {
    "low":      "green",
    "medium":   "yellow",
    "high":     "red",
    "critical": "bold red",
    "unknown":  "dim",
}

STEP_STATUS_COLOR = {
    "ok":      "green",
    "warn":    "yellow",
    "error":   "bold red",
    "skipped": "dim",
}


def print_header(text: str) -> None:
    if HAS_RICH:
        console.print(Panel(
            f"[bold cyan]{text}[/bold cyan]",
            border_style="cyan",
            padding=(0, 2),
        ))
    else:
        print(f"\n{'='*60}\n  {text}\n{'='*60}")


def print_step_table(run: AgentRun) -> None:
    if not HAS_RICH:
        for s in run.steps:
            print(f"  [{s.status.upper()}] {s.step_name:12} ({s.duration_ms:.0f}ms)  {s.summary[:80]}")
        return

    table = Table(
        title="Agent Pipeline Steps",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold white",
        title_style="bold",
        expand=True,
    )
    table.add_column("Step",     style="bold white",  no_wrap=True, width=14)
    table.add_column("Status",   no_wrap=True,        width=10)
    table.add_column("Time ms",  justify="right",     width=9)
    table.add_column("Summary")

    for s in run.steps:
        color = STEP_STATUS_COLOR.get(s.status, "white")
        table.add_row(
            s.step_name,
            f"[{color}]{s.status.upper()}[/{color}]",
            f"{s.duration_ms:.0f}",
            s.summary[:120],
        )
    console.print(table)


def print_diagnosis(run: AgentRun) -> None:
    dx = run.diagnosis
    if dx is None:
        if HAS_RICH:
            console.print("[bold red]No diagnosis produced.[/bold red]")
        else:
            print("No diagnosis produced.")
        return

    sev_color = SEVERITY_COLOR.get(dx.severity, "white")

    if HAS_RICH:
        # Title panel
        console.print(Panel(
            f"[bold]{dx.fault_name}[/bold]\n"
            f"[dim]{dx.fault_id}[/dim]",
            title="[bold]DIAGNOSIS[/bold]",
            border_style=sev_color,
            padding=(0, 2),
        ))

        # Key metrics table
        meta = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
        meta.add_column("Key",   style="dim")
        meta.add_column("Value", style="bold")
        meta.add_row("Machine",    dx.machine_id)
        meta.add_row("Timestamp",  dx.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"))
        meta.add_row("Confidence", f"[bold]{dx.confidence.upper()}[/bold]")
        meta.add_row("Severity",   f"[{sev_color}]{dx.severity.upper()}[/{sev_color}]")
        meta.add_row("Urgency",    dx.urgency.upper())
        console.print(meta)

        # Explanation
        console.print(Panel(
            dx.explanation,
            title="Reasoning",
            border_style="blue",
            padding=(0, 2),
        ))

        # Evidence
        if dx.evidence:
            ev_text = "\n".join(f"  • {e}" for e in dx.evidence)
            console.print(Panel(
                ev_text,
                title="Supporting Evidence",
                border_style="yellow",
                padding=(0, 2),
            ))

        # Differential
        if dx.differential_diagnosis:
            diff_text = "\n".join(f"  • {d}" for d in dx.differential_diagnosis)
            console.print(Panel(
                diff_text,
                title="Differential Diagnosis",
                border_style="dim",
                padding=(0, 1),
            ))

        # Uncertainty
        if dx.uncertainty_notes:
            console.print(Panel(
                f"[yellow]{dx.uncertainty_notes}[/yellow]",
                title="⚠ Uncertainty",
                border_style="yellow",
                padding=(0, 2),
            ))

        # Recommendations
        act_text = "\n".join(
            f"  [bold cyan]{i}.[/bold cyan] {a}"
            for i, a in enumerate(dx.recommended_actions, 1)
        )
        console.print(Panel(
            act_text,
            title="Recommended Actions",
            border_style="green",
            padding=(0, 2),
        ))

    else:
        print(f"\nDIAGNOSIS: {dx.fault_name} ({dx.fault_id})")
        print(f"  Machine   : {dx.machine_id}")
        print(f"  Confidence: {dx.confidence}")
        print(f"  Severity  : {dx.severity}")
        print(f"  Urgency   : {dx.urgency}")
        print(f"\nExplanation:\n  {dx.explanation}")
        if dx.evidence:
            print("\nEvidence:")
            for e in dx.evidence:
                print(f"  - {e}")
        print("\nRecommended Actions:")
        for i, a in enumerate(dx.recommended_actions, 1):
            print(f"  {i}. {a}")
        if dx.uncertainty_notes:
            print(f"\nUncertainty: {dx.uncertainty_notes}")


def save_report(run: AgentRun, output_dir: Path) -> Path:
    """Save a JSON report to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = run.started_at.strftime("%Y%m%d_%H%M%S")
    filename = output_dir / f"report_{run.machine_id}_{ts}.json"

    dx = run.diagnosis
    report_data = {
        "run_id":    run.run_id,
        "machine_id": run.machine_id,
        "started_at": run.started_at.isoformat(),
        "duration_ms": run.duration_ms,
        "succeeded": run.succeeded,
        "diagnosis": dx.to_dict() if dx else None,
        "steps": [
            {
                "step":        s.step_name,
                "status":      s.status,
                "duration_ms": s.duration_ms,
                "summary":     s.summary,
            }
            for s in run.steps
        ],
        "fault_candidates": [
            {
                "fault_id":    c.fault.fault_id,
                "fault_name":  c.fault.name,
                "match_score": c.match_score,
            }
            for c in run.fault_candidates
        ],
    }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, default=str)
    return filename


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_scenario(
    scenario: str,
    agent: DiagnosisAgent,
    output_dir: Path,
    machine_id: str = "PUMP-01",
) -> AgentRun:
    simulator = SensorSimulator()
    readings = simulator.generate(scenario, machine_id=machine_id, n_readings=12)
    print_header(f"Scenario: {scenario.upper().replace('_', ' ')}  |  Machine: {machine_id}")
    run = agent.run(readings)
    print_step_table(run)
    print_diagnosis(run)
    report_path = save_report(run, output_dir)
    if HAS_RICH:
        console.print(f"\n[dim]Report saved → {report_path}[/dim]\n")
    else:
        print(f"\nReport saved → {report_path}\n")
    return run


def run_csv(
    csv_path: str,
    agent: DiagnosisAgent,
    output_dir: Path,
) -> AgentRun:
    with open(csv_path, "r", encoding="utf-8") as f:
        text = f.read()
    readings = SensorDataLoader.from_csv_string(text)
    machine_id = readings[-1].machine_id if readings else "CSV-MACHINE"
    print_header(f"CSV Input: {csv_path}  |  Machine: {machine_id}")
    run = agent.run(readings)
    print_step_table(run)
    print_diagnosis(run)
    report_path = save_report(run, output_dir)
    if HAS_RICH:
        console.print(f"\n[dim]Report saved → {report_path}[/dim]\n")
    else:
        print(f"\nReport saved → {report_path}\n")
    return run


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Machine Fault Diagnosis Agent — demo runner"
    )
    parser.add_argument(
        "--scenario",
        choices=SensorSimulator.SCENARIOS,
        default=None,
        help="Run a single scenario (default: run all)",
    )
    parser.add_argument(
        "--csv",
        default=None,
        metavar="PATH",
        help="Diagnose readings from a CSV file",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        default=False,
        help="Use rule-based reasoner (no API key required)",
    )
    parser.add_argument(
        "--output-dir",
        default="reports",
        metavar="DIR",
        help="Directory for JSON reports (default: reports/)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="LLM model name override (e.g. gpt-4o, gpt-4o-mini)",
    )
    args = parser.parse_args()

    use_llm = not args.no_llm and bool(os.environ.get("OPENAI_API_KEY"))
    if not use_llm and not args.no_llm:
        if HAS_RICH:
            console.print(
                "[yellow]⚠ OPENAI_API_KEY not set — using rule-based reasoner.[/yellow]\n"
                "[dim]Set OPENAI_API_KEY (or pass --no-llm) to suppress this warning.[/dim]\n"
            )
        else:
            print("WARNING: OPENAI_API_KEY not set — using rule-based reasoner.\n")

    output_dir = Path(args.output_dir)

    agent = DiagnosisAgent(
        use_llm=use_llm,
        llm_model=args.model,
    )

    if args.csv:
        run_csv(args.csv, agent, output_dir)
        return

    scenarios = [args.scenario] if args.scenario else SensorSimulator.SCENARIOS
    machines  = ["PUMP-01", "MOTOR-02", "COMP-03", "FAN-04", "PUMP-05", "MOTOR-06"]

    if HAS_RICH:
        console.rule("[bold cyan]Machine Fault Diagnosis Agent[/bold cyan]")
        console.print(
            f"[dim]Reasoner: {'LLM (' + (args.model or 'gpt-4o-mini') + ')' if use_llm else 'Rule-based'}[/dim]"
        )
    else:
        print("Machine Fault Diagnosis Agent")

    for i, scenario in enumerate(scenarios):
        mid = machines[i % len(machines)]
        run_scenario(scenario, agent, output_dir, machine_id=mid)

    if HAS_RICH:
        console.rule("[bold green]All scenarios complete[/bold green]")
    else:
        print("\n=== All scenarios complete ===")


if __name__ == "__main__":
    main()
