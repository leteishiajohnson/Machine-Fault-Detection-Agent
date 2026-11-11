"""
sensor_data.py — Sensor data ingestion, validation, and simulation.

Provides:
  - SensorReading: typed dataclass for a single multi-signal snapshot
  - SensorDataLoader: load from CSV or dict
  - SensorSimulator: generate realistic synthetic scenarios for testing
"""

from __future__ import annotations

import csv
import io
import math
import random
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Signal definitions — normal operating ranges for an industrial pump/motor
# ---------------------------------------------------------------------------
SIGNAL_NORMAL_RANGES: Dict[str, tuple[float, float]] = {
    "temperature_c":    (60.0,  85.0),   # °C  — bearing/housing temp
    "vibration_mms":    (0.5,   4.5),    # mm/s RMS — casing vibration
    "pressure_bar":     (3.5,   7.0),    # bar — discharge pressure
    "current_a":        (18.0, 28.0),    # A   — motor phase current
    "rpm":              (1440, 1510),    # RPM — rotational speed
}

SIGNAL_UNITS: Dict[str, str] = {
    "temperature_c":  "°C",
    "vibration_mms":  "mm/s",
    "pressure_bar":   "bar",
    "current_a":      "A",
    "rpm":            "RPM",
}


# ---------------------------------------------------------------------------
# Core dataclass
# ---------------------------------------------------------------------------
@dataclass
class SensorReading:
    """A single point-in-time sensor snapshot from one machine."""

    machine_id: str
    timestamp: datetime
    temperature_c: float
    vibration_mms: float
    pressure_bar: float
    current_a: float
    rpm: float
    metadata: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, float | str]:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d

    def signals(self) -> Dict[str, float]:
        """Return only the numeric sensor signals."""
        return {
            "temperature_c": self.temperature_c,
            "vibration_mms": self.vibration_mms,
            "pressure_bar":  self.pressure_bar,
            "current_a":     self.current_a,
            "rpm":           self.rpm,
        }


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
class SensorDataLoader:
    """Load SensorReadings from CSV text or list-of-dicts."""

    REQUIRED_COLUMNS = {
        "machine_id", "timestamp",
        "temperature_c", "vibration_mms",
        "pressure_bar", "current_a", "rpm",
    }

    @classmethod
    def from_csv_string(cls, csv_text: str) -> List[SensorReading]:
        reader = csv.DictReader(io.StringIO(csv_text.strip()))
        missing = cls.REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")
        readings = []
        for row in reader:
            readings.append(cls._row_to_reading(row))
        return readings

    @classmethod
    def from_dict_list(cls, rows: List[Dict]) -> List[SensorReading]:
        return [cls._row_to_reading(r) for r in rows]

    @staticmethod
    def _row_to_reading(row: Dict) -> SensorReading:
        ts_raw = row.get("timestamp", "")
        try:
            ts = datetime.fromisoformat(str(ts_raw))
        except ValueError:
            ts = datetime.now(tz=timezone.utc)
        return SensorReading(
            machine_id=str(row.get("machine_id", "UNKNOWN")),
            timestamp=ts,
            temperature_c=float(row["temperature_c"]),
            vibration_mms=float(row["vibration_mms"]),
            pressure_bar=float(row["pressure_bar"]),
            current_a=float(row["current_a"]),
            rpm=float(row["rpm"]),
            metadata={
                k: str(v) for k, v in row.items()
                if k not in SensorDataLoader.REQUIRED_COLUMNS
            },
        )


# ---------------------------------------------------------------------------
# Simulator — generates realistic labelled test scenarios
# ---------------------------------------------------------------------------
class SensorSimulator:
    """
    Generate synthetic multi-reading sensor windows for known fault scenarios.
    Useful for integration testing and demo runs.
    """

    SCENARIOS = [
        "normal",
        "bearing_failure",
        "overheating",
        "cavitation",
        "electrical_overload",
        "rotor_imbalance",
    ]

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)

    def generate(
        self,
        scenario: str,
        machine_id: str = "PUMP-01",
        n_readings: int = 10,
    ) -> List[SensorReading]:
        """
        Return a list of SensorReadings representing the specified scenario.
        The last few readings carry the fault signature.
        """
        if scenario not in self.SCENARIOS:
            raise ValueError(
                f"Unknown scenario '{scenario}'. "
                f"Valid: {self.SCENARIOS}"
            )
        base = self._baseline(n_readings)
        inject = getattr(self, f"_inject_{scenario}")
        readings = inject(base, machine_id)
        return readings

    # -- helpers -----------------------------------------------------------

    def _jitter(self, value: float, pct: float = 0.02) -> float:
        return value * (1.0 + self._rng.uniform(-pct, pct))

    def _baseline(self, n: int) -> List[Dict[str, float]]:
        now = datetime.now(tz=timezone.utc).timestamp()
        rows = []
        for i in range(n):
            rows.append({
                "temperature_c": self._jitter(72.0),
                "vibration_mms": self._jitter(2.5),
                "pressure_bar":  self._jitter(5.2),
                "current_a":     self._jitter(23.0),
                "rpm":           self._jitter(1480.0),
                "_ts":           now + i * 10,
            })
        return rows

    def _make_readings(
        self, rows: List[Dict], machine_id: str
    ) -> List[SensorReading]:
        out = []
        for r in rows:
            ts = datetime.fromtimestamp(r.pop("_ts"), tz=timezone.utc)
            out.append(SensorReading(machine_id=machine_id, timestamp=ts, **r))
        return out

    # -- scenario injectors ------------------------------------------------

    def _inject_normal(self, rows, machine_id):
        return self._make_readings(rows, machine_id)

    def _inject_bearing_failure(self, rows, machine_id):
        for r in rows[-4:]:
            r["temperature_c"] = self._jitter(108.0, 0.03)   # hot
            r["vibration_mms"] = self._jitter(12.5, 0.05)    # very high
            r["rpm"]           = self._jitter(1460.0, 0.02)  # slight drop
        return self._make_readings(rows, machine_id)

    def _inject_overheating(self, rows, machine_id):
        for r in rows[-5:]:
            r["temperature_c"] = self._jitter(118.0, 0.02)   # very hot
            r["current_a"]     = self._jitter(25.5, 0.02)    # slightly elevated
        return self._make_readings(rows, machine_id)

    def _inject_cavitation(self, rows, machine_id):
        for r in rows[-4:]:
            r["pressure_bar"]  = self._jitter(1.8, 0.05)     # pressure collapse
            r["vibration_mms"] = self._jitter(9.0, 0.06)     # cavitation noise
            r["current_a"]     = self._jitter(21.0, 0.03)    # slightly low
        return self._make_readings(rows, machine_id)

    def _inject_electrical_overload(self, rows, machine_id):
        for r in rows[-3:]:
            r["current_a"]     = self._jitter(41.0, 0.03)    # surge
            r["temperature_c"] = self._jitter(92.0, 0.02)    # motor heat
            r["rpm"]           = self._jitter(1455.0, 0.02)  # slight drop
        return self._make_readings(rows, machine_id)

    def _inject_rotor_imbalance(self, rows, machine_id):
        for i, r in enumerate(rows[-5:], 1):
            r["vibration_mms"] = self._jitter(7.5 + i * 0.4, 0.05)  # growing
            r["rpm"]           = self._jitter(1476.0, 0.02)
        return self._make_readings(rows, machine_id)
