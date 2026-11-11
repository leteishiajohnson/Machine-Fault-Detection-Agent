"""
tests/test_sensor_data.py — Unit tests for sensor data ingestion and simulation.
"""

import pytest
from datetime import timezone

from src.sensor_data import (
    SensorSimulator,
    SensorDataLoader,
    SensorReading,
    SIGNAL_NORMAL_RANGES,
    SIGNAL_UNITS,
)


class TestSensorSimulator:

    def setup_method(self):
        self.sim = SensorSimulator(seed=42)

    def test_normal_scenario_returns_readings(self):
        readings = self.sim.generate("normal", n_readings=10)
        assert len(readings) == 10

    def test_all_scenarios_succeed(self):
        for scenario in SensorSimulator.SCENARIOS:
            readings = self.sim.generate(scenario, n_readings=10)
            assert len(readings) == 10, f"Failed for scenario: {scenario}"

    def test_readings_have_required_fields(self):
        readings = self.sim.generate("bearing_failure", n_readings=5)
        for r in readings:
            assert isinstance(r, SensorReading)
            assert r.machine_id
            assert r.timestamp is not None
            assert isinstance(r.temperature_c, float)
            assert isinstance(r.vibration_mms, float)
            assert isinstance(r.pressure_bar, float)
            assert isinstance(r.current_a, float)
            assert isinstance(r.rpm, float)

    def test_bearing_failure_has_elevated_vibration(self):
        readings = self.sim.generate("bearing_failure", n_readings=12)
        last = readings[-1]
        # Should be well above normal max of 4.5
        assert last.vibration_mms > 8.0, (
            f"Expected high vibration, got {last.vibration_mms}"
        )

    def test_overheating_has_elevated_temperature(self):
        readings = self.sim.generate("overheating", n_readings=12)
        last = readings[-1]
        assert last.temperature_c > 100.0, (
            f"Expected high temperature, got {last.temperature_c}"
        )

    def test_cavitation_has_low_pressure(self):
        readings = self.sim.generate("cavitation", n_readings=12)
        last = readings[-1]
        assert last.pressure_bar < 3.0, (
            f"Expected low pressure, got {last.pressure_bar}"
        )

    def test_electrical_overload_has_high_current(self):
        readings = self.sim.generate("electrical_overload", n_readings=12)
        last = readings[-1]
        assert last.current_a > 35.0, (
            f"Expected high current, got {last.current_a}"
        )

    def test_invalid_scenario_raises(self):
        with pytest.raises(ValueError, match="Unknown scenario"):
            self.sim.generate("nonexistent_fault")

    def test_signals_dict_contains_all_keys(self):
        readings = self.sim.generate("normal", n_readings=5)
        signals = readings[0].signals()
        expected_keys = set(SIGNAL_NORMAL_RANGES.keys())
        assert expected_keys == set(signals.keys())


class TestSensorDataLoader:

    SAMPLE_CSV = (
        "machine_id,timestamp,temperature_c,vibration_mms,pressure_bar,current_a,rpm\n"
        "PUMP-01,2024-01-01T00:00:00+00:00,72.0,2.5,5.2,23.0,1480\n"
        "PUMP-01,2024-01-01T00:00:10+00:00,73.0,2.6,5.1,23.1,1479\n"
    )

    def test_from_csv_string_returns_readings(self):
        readings = SensorDataLoader.from_csv_string(self.SAMPLE_CSV)
        assert len(readings) == 2

    def test_from_csv_string_parses_values(self):
        readings = SensorDataLoader.from_csv_string(self.SAMPLE_CSV)
        r = readings[0]
        assert r.machine_id == "PUMP-01"
        assert r.temperature_c == 72.0
        assert r.vibration_mms == 2.5

    def test_from_csv_missing_column_raises(self):
        bad_csv = "machine_id,timestamp,temperature_c\nPUMP-01,2024-01-01T00:00:00,72.0\n"
        with pytest.raises(ValueError, match="missing required columns"):
            SensorDataLoader.from_csv_string(bad_csv)

    def test_from_dict_list(self):
        rows = [
            {
                "machine_id": "M1",
                "timestamp":  "2024-01-01T00:00:00+00:00",
                "temperature_c": "80.0",
                "vibration_mms": "3.0",
                "pressure_bar":  "5.0",
                "current_a":     "22.0",
                "rpm":           "1480",
            }
        ]
        readings = SensorDataLoader.from_dict_list(rows)
        assert len(readings) == 1
        assert readings[0].temperature_c == 80.0

    def test_to_dict_round_trip(self):
        readings = SensorDataLoader.from_csv_string(self.SAMPLE_CSV)
        d = readings[0].to_dict()
        assert "temperature_c" in d
        assert "machine_id" in d
        assert isinstance(d["timestamp"], str)
