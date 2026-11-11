# Machine Fault Diagnosis Agent

An AI-powered industrial machine fault diagnosis application built with **IBM Granite** (via **IBM watsonx.ai**), scikit-learn Anomaly Detection, and a polished **Streamlit** interface.

---

## Overview

The Machine Fault Diagnosis Agent continuously monitors industrial machinery sensor readings, detects anomalies, correlates signals with known fault patterns, and generates human-readable diagnoses using IBM Granite large language models. It supports a rule-based fallback engine for operation without API credentials.

```
Sensor Data
    → Ingest & Validate
    → Anomaly Detection (Isolation Forest + statistical thresholds)
    → Fault Investigation (signal-to-fault correlation)
    → IBM Granite Reasoning (via watsonx.ai)
    → Structured Diagnosis
    → Human-readable Explanation
    → Maintenance & Safety Recommendations
```

---

## Features

| Feature | Description |
|---|---|
| **7-Step Agent Pipeline** | INGEST → DETECT → INVESTIGATE → REASON → DIAGNOSE → EXPLAIN → RECOMMEND |
| **IBM Granite AI** | Structured fault reasoning via watsonx.ai using `ibm/granite-3-8b-instruct` |
| **Anomaly Detection** | Isolation Forest ML + per-signal statistical threshold analysis |
| **Fault Knowledge Base** | 5 fault types: Bearing Failure, Overheating, Cavitation, Electrical Overload, Rotor Imbalance |
| **Streamlit UI** | 7-page application: Dashboard, Sensor Monitoring, AI Chat, Fault Detection, Maintenance, Agent Trace, Settings |
| **CSV Upload** | Load real sensor data from CSV for diagnosis |
| **Simulated Data** | Built-in sensor simulator for all 6 fault scenarios |
| **Rule-based Fallback** | Full functionality with no API credentials required |
| **CLI Entry Point** | `python main.py` for command-line batch diagnosis |

---

## Application Architecture

```
machine_fault_agent/
├── app.py                        ← Streamlit application (7 pages)
├── main.py                       ← CLI entry point
├── requirements.txt
├── .env.example                  ← Copy to .env and fill credentials
├── .gitignore
└── src/
    ├── agent.py                  ← DiagnosisAgent — 7-step pipeline
    ├── sensor_data.py            ← SensorReading, SensorSimulator, SensorDataLoader
    ├── anomaly_detection.py      ← AnomalyDetector (Isolation Forest)
    ├── fault_knowledge_base.py   ← FaultDefinition, SignalInvestigator, FAULT_KB
    ├── llm_reasoner.py           ← LLMReasoner (OpenAI-compat), RuleBasedReasoner
    └── watsonx_reasoner.py       ← WatsonxReasoner (IBM Granite, SaaS + CP4D)
```

### Reasoner Priority

When `use_llm=True` (default), the agent selects a reasoner in this order:

| Priority | Reasoner | Requires |
|---|---|---|
| 1 | **IBM Granite (watsonx.ai)** | `WATSONX_API_KEY` + `WATSONX_PROJECT_ID` |
| 2 | **OpenAI-compatible LLM** | `OPENAI_API_KEY` |
| 3 | **Rule-based fallback** | Nothing — always available |

---

## Technologies Used

- **IBM watsonx.ai** — Foundation model inference platform
- **IBM Granite** (`ibm/granite-3-8b-instruct`) — Instruction-tuned LLM for structured reasoning
- **ibm-watsonx-ai** — Official IBM Python SDK (`>= 1.0.0`)
- **scikit-learn** — Isolation Forest for unsupervised anomaly detection
- **Streamlit** — Interactive web application framework
- **pandas / numpy** — Sensor data processing
- **python-dotenv** — Secure environment variable loading

---

## Installation

### Prerequisites

- Python 3.10 or later
- pip

### Steps

```bash
# 1. Clone or navigate to the project directory
cd machine_fault_agent

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Environment Configuration

### IBM watsonx.ai (SaaS — recommended)

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Open `.env` and fill in your IBM Cloud credentials:
   ```env
   WATSONX_API_KEY=your-ibm-cloud-api-key
   WATSONX_USERNAME=your-ibm-cloud-email@example.com
   WATSONX_PROJECT_ID=your-watsonx-project-id
   WATSONX_URL=https://us-south.ml.cloud.ibm.com
   WATSONX_MODEL_ID=ibm/granite-3-8b-instruct
   WATSONX_VERSION=5.2
   ```

   - **API key**: [IBM Cloud → Manage → IAM → API keys](https://cloud.ibm.com/iam/apikeys)
   - **Project ID**: From your [watsonx.ai project settings](https://dataplatform.cloud.ibm.com/projects/?context=wx)
   - **Regional URL options**:
     | Region | URL |
     |---|---|
     | Dallas (default) | `https://us-south.ml.cloud.ibm.com` |
     | Frankfurt | `https://eu-de.ml.cloud.ibm.com` |
     | London | `https://eu-gb.ml.cloud.ibm.com` |
     | Tokyo | `https://jp-tok.ml.cloud.ibm.com` |
     | Sydney | `https://au-syd.ml.cloud.ibm.com` |

### IBM Cloud Pak for Data (CP4D — on-premises)

```env
WATSONX_CPD_URL=https://your-cpd-instance.example.com
WATSONX_CPD_USERNAME=your-username
WATSONX_CPD_PASSWORD=your-password
WATSONX_PROJECT_ID=your-project-id
WATSONX_MODEL_ID=ibm/granite-3-8b-instruct
```

> The presence of `WATSONX_CPD_URL` activates CP4D authentication mode automatically.

### No credentials (rule-based mode)

The application works fully without any API credentials. Leave `.env` unconfigured and the rule-based reasoner will be used automatically.

> ⚠️ **Security**: Never commit your `.env` file. It is listed in `.gitignore`.

---

## Running the Application

### Streamlit Web Application

```bash
streamlit run app.py
```

The application opens at `http://localhost:8501` with 7 pages:

| Page | Description |
|---|---|
| **🏠 Dashboard** | Machine health overview, quick diagnosis, KPI cards |
| **📡 Sensor Monitoring** | Live-simulated charts, CSV upload, signal analysis |
| **💬 AI Diagnosis Chat** | Conversational fault Q&A powered by IBM Granite |
| **🔍 Fault Detection** | Anomaly scores, flagged signals, fault candidate ranking |
| **🔧 Maintenance** | Prioritised actions, safety warnings, technician guidance |
| **🗂️ Agent Trace** | Full 7-step pipeline execution log with timings |
| **⚙️ Settings** | watsonx.ai status, connection test, configuration guide |

### CLI Entry Point

```bash
# Run all 6 demo scenarios
python main.py

# Run a specific scenario
python main.py --scenario bearing_failure

# Use rule-based mode (no API key needed)
python main.py --no-llm

# Diagnose from a CSV file
python main.py --csv path/to/sensor_data.csv

# All options
python main.py --help
```

---

## Connecting IBM watsonx.ai

1. **Configure credentials** in `.env` as described above.
2. **Restart** the Streamlit app:
   ```bash
   streamlit run app.py
   ```
3. **Verify** the connection in the **⚙️ Settings** page using the **Test Connection** button.
4. The sidebar will show: `✓ Granite · granite-3-8b-instruct` when connected.

If the connection test fails:
- Check that `WATSONX_API_KEY` has permission to access the project.
- Confirm `WATSONX_PROJECT_ID` matches a project in your watsonx.ai workspace.
- Verify the regional `WATSONX_URL` matches your IBM Cloud region.

---

## Testing

### Run all tests

```bash
pytest tests/ -v
```

### Run only the new watsonx tests (mocked — no credentials required)

```bash
pytest tests/test_watsonx_reasoner.py -v
```

### Run live watsonx.ai tests (requires real credentials)

```bash
WATSONX_API_KEY=<key> WATSONX_PROJECT_ID=<id> pytest tests/ -k live -v
```

### Test summary

| Test file | Tests | What is covered |
|---|---|---|
| `test_sensor_data.py` | 14 | Sensor reading, simulation, CSV loading |
| `test_anomaly_detection.py` | 11 | Isolation Forest, threshold analysis, all scenarios |
| `test_fault_knowledge_base.py` | 13 | KB structure, signal investigator, candidate ranking |
| `test_agent.py` | 19 | Full 7-step pipeline, all scenarios, callbacks |
| `test_watsonx_reasoner.py` | 22 + 3 live | Mocked SDK, JSON parsing, fallback, config status |

**Total: 79 passing, 3 skipped (live tests)** — all run in under 90 seconds.

---

## Sensor Signals

The application monitors these signals for an industrial pump/motor:

| Signal | Unit | Normal Range | Description |
|---|---|---|---|
| `temperature_c` | °C | 60 – 85 | Bearing/housing temperature |
| `vibration_mms` | mm/s | 0.5 – 4.5 | Casing vibration (RMS) |
| `pressure_bar` | bar | 3.5 – 7.0 | Discharge pressure |
| `current_a` | A | 18 – 28 | Motor phase current |
| `rpm` | RPM | 1440 – 1510 | Rotational speed |

## Fault Types

| Fault | Primary Signals | Urgency |
|---|---|---|
| Bearing Failure | Vibration ↑, Temperature ↑ | Immediate |
| Overheating | Temperature ↑↑ | Immediate |
| Pump Cavitation | Pressure ↓, Vibration ↑ | Schedule |
| Electrical Overload | Current ↑↑ | Immediate |
| Rotor Imbalance | Vibration ↑ | Schedule |

---

## CSV Data Format

Upload sensor data in this format:

```csv
machine_id,timestamp,temperature_c,vibration_mms,pressure_bar,current_a,rpm
PUMP-01,2024-01-15T08:00:00+00:00,72.0,2.5,5.2,23.0,1480
PUMP-01,2024-01-15T08:00:10+00:00,73.0,2.6,5.1,23.1,1479
```

Minimum **4 readings** required. Download a sample from the Sensor Monitoring page.

---

## Limitations and Future Scope

### Current Limitations

- **Pump/motor domain only** — signal ranges and fault types are calibrated for an industrial pump/motor. Other machine types would require different signal definitions.
- **No real-time hardware integration** — the application reads CSV files or simulated data. Integration with OPC-UA, MQTT, or SCADA requires additional development.
- **Granite response format** — the model is prompted to return JSON. Occasional formatting variations are handled by the parser, but complex responses may fall back to the rule-based engine.
- **Single machine per run** — the pipeline processes one machine window at a time.
- **No historical trending** — readings are not persisted between sessions.

### Future Scope

- Real-time sensor integration via MQTT or OPC-UA
- Multi-machine fleet monitoring dashboard
- Historical trend analysis and maintenance scheduling
- Fine-tuned Granite model on industrial fault data
- Custom alert thresholds per machine type
- PDF report export
- User authentication for multi-tenant deployments

---

## Safety Notice

> ⛔ This application provides **decision-support information only**.
> It does **not** and **cannot** automatically control, start, or stop any machinery.
> Always follow your site's lock-out/tag-out (LOTO) procedures.
> A qualified maintenance engineer must review and authorise all work.
> Never rely solely on AI-generated diagnoses for safety-critical decisions.
