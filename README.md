# Pulso Python Backend

FastAPI backend for live PPG measurements, persisted recordings, and PPG quality-analysis feature extraction.

## Run

```bash
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8080
```

The backend loads `.env` automatically. `DATABASE_URL` or `POSTGRES_DSN` enables persistence.
If a full DSN is not set, the app can build one from `POSTGRES_HOST`, `POSTGRES_PORT`,
`POSTGRES_PASSWORD`, `POSTGRES_USER`, and `POSTGRES_DB`.
`CORS_ORIGINS` can override the comma-separated frontend origins allowed by the API;
by default `http://localhost:3000` and `http://127.0.0.1:3000` are allowed.

On startup the app creates or migrates `users`, `projects`, `project_users`, `measurements`,
`recordings`, `recordings_samples`, and `quality_analyses`.

## API v4.2

Public IDs are prefixed:

```text
dev_ for devices
mes_ for measurements
rec_ for recordings
usr_ for users
prj_ for projects
qlt_ for quality analyses
```

All HTTP responses use the v4 envelope:

```json
{
  "data": {},
  "$meta": {
    "status": "success",
    "timestamp": "2026-04-28T11:51:20.778832Z"
  }
}
```

Errors use explicit codes:

```json
{
  "data": null,
  "$meta": {
    "status": "error",
    "timestamp": "2026-04-28T11:51:20.778832Z",
    "error": {
      "http_status": 404,
      "code": "recording_not_found",
      "message": "Recording not found"
    }
  }
}
```

Project-user assignment errors include `assignment_not_found` and `project_user_already_exists`.

## HTTP Endpoints

```text
GET    /api/health

GET    /api/devices?connection_status=connected|disconnected&limit=100&offset=0
GET    /api/devices/{device_id}/metrics
DELETE /api/devices/{device_id}

POST   /api/devices/{device_id}/measurements
POST   /api/measurements/{measurement_id}/stop
POST   /api/measurements/{measurement_id}/recording
POST   /api/measurements/{measurement_id}/recording/stop

GET    /api/recordings?device_id=&date_from=&date_to=&user_id=&project_id=&status=&limit=100&offset=0
GET    /api/recordings/{recording_id}
DELETE /api/recordings/{recording_id}
GET    /api/recordings/{recording_id}/samples?limit=1000&offset=0
GET    /api/recordings/{recording_id}/export?format=json|csv
POST   /api/recordings/{recording_id}/quality-analysis
GET    /api/recordings/{recording_id}/quality-analysis

GET    /api/assistant/status
POST   /api/assistant/chat

POST   /api/users
GET    /api/users?project_id=&limit=100&offset=0
GET    /api/users/{user_id}
PATCH  /api/users/{user_id}
DELETE /api/users/{user_id}

POST   /api/projects
GET    /api/projects?limit=100&offset=0
GET    /api/projects/{project_id}
PATCH  /api/projects/{project_id}
DELETE /api/projects/{project_id}
POST   /api/projects/{project_id}/users
DELETE /api/projects/{project_id}/users/{user_id}
```

Measurement start body:

```json
{
  "user_id": "usr_1",
  "project_id": "prj_1"
}
```

Recording start body:

```json
{
  "duration_s": 15
}
```

Project user assignment body:

```json
{
  "user_id": "usr_1"
}
```

## WebSocket Protocol

Device channel:

```text
ws://localhost:8080/ws/devices/{device_id}
```

Client live stream:

```text
ws://localhost:8080/ws/measurements/{measurement_id}/stream
```

Device hello:

```json
{
  "type": "hello",
  "device_id": "dev_sim-device-001",
  "is_simulated": true
}
```

Device samples:

```json
{
  "type": "samples",
  "device_id": "dev_sim-device-001",
  "measurement_id": "mes_6883cd24-dc32-404a-9081-60d49e376a0b",
  "recording_id": "rec_23424545665345",
  "sample_rate_hz": 25.0,
  "sensor_temp_c": 30.9,
  "samples": [
    {
      "index": 100,
      "ir": 84231,
      "red": 53211
    }
  ]
}
```

Client stream updates:

```json
{
  "type": "measurement_update",
  "measurement_id": "mes_6883cd24-dc32-404a-9081-60d49e376a0b",
  "device_id": "dev_sim-device-001",
  "active_recording_id": "rec_23424545665345",
  "timestamp": "2026-04-28T11:51:20.778832Z",
  "sample_rate_hz": 25.0,
  "sensor_temp_c": 30.9,
  "samples": [
    {
      "index": 100,
      "t_ms": 4000.0,
      "ir": 84231,
      "red": 53211,
      "ir_filtered": 0.031,
      "red_filtered": 0.024
    }
  ],
  "metrics": {
    "bpm": 88.2,
    "spo2": 95.1,
    "ratio": 0.5977,
    "signal_readiness": {
      "level": "high",
      "score": 0.91,
      "is_recording_ready": true,
      "reason": null
    }
  }
}
```

## Quality Analysis

`POST /api/recordings/{recording_id}/quality-analysis` requires `QUALITY_MODEL_PATH`.
Supported model files:

- `.pkl` via `pickle`
- `.joblib` via `joblib`
- `.json` threshold model for local deterministic testing

The model receives features in this order:

```text
peak_count, valid_peak_count, ibi_cv, perfusion_index, template_corr_mean,
relative_power_hr_band, baseline_drift, flatline_ratio, saturation_ratio, outlier_ratio
```

Model metadata defaults to:

```text
QUALITY_MODEL_TYPE=random_forest
QUALITY_MODEL_NAME=ppg_quality_rf
QUALITY_MODEL_VERSION=1.0.0
```

## Assistant Chat

`POST /api/assistant/chat` asks an LLM-backed data assistant a question about a
completed recording. The assistant can only use the recording row and the
stored ML `quality-analysis`; it does not receive raw samples.

Request:

```json
{
  "recording_id": "rec_23424545665345",
  "message": "Why is this recording quality low?"
}
```

Response:

```json
{
  "data": {
    "assistant": {
      "recording_id": "rec_23424545665345",
      "quality_analysis_id": "qlt_6883cd24-dc32-404a-9081-60d49e376a0b",
      "timestamp": "2026-05-03T12:00:00Z",
      "provider": {
        "type": "ollama",
        "model": "llama3.1:8b"
      },
      "message": "The ML quality analysis indicates..."
    }
  }
}
```

Required preconditions:

- the database is configured;
- the recording exists and has `status=completed`;
- `POST /api/recordings/{recording_id}/quality-analysis` has already completed.

Assistant configuration:

```text
LLM_ENABLED=true
LLM_PROVIDER=openai_compatible
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=...
LLM_MODEL=gpt-4.1-mini
LLM_TIMEOUT_SECONDS=30
LLM_TEMPERATURE=0.2
LLM_MAX_TOKENS=500
```

For local Ollama:

```text
LLM_ENABLED=true
LLM_PROVIDER=ollama
LLM_BASE_URL=http://localhost:11434
LLM_MODEL=llama3.1:8b
```

### Train BUT-PPG Random Forest

The repository can train a reproducible Random Forest quality model from the
local Brno University of Technology Smartphone PPG database:

```bash
python3 scripts/train_but_ppg_quality_rf.py \
  --dataset brno-university-of-technology-smartphone-ppg-database-but-ppg-2.0.0 \
  --output models/ppg_quality_rf.joblib
```

The script maps BUT-PPG `Quality=0` to `low` and `Quality=1` to `high`, extracts
the same features used by the API, evaluates with grouped subject splits, and
writes a metrics report next to the model:

```text
models/ppg_quality_rf.metrics.json
```

Use the generated model at runtime with:

```bash
QUALITY_MODEL_PATH=models/ppg_quality_rf.joblib
```

BUT-PPG is RGB smartphone PPG, while this project records MAX30105 IR/Red PPG.
The generated model is a useful first-pass quality estimator, but production
accuracy should be calibrated with recordings from the target hardware.

## Test

```bash
python3 -m unittest discover tests
```
