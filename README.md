# Pulso Python Backend

Backend for receiving PPG samples over WebSocket, filtering the red/IR signal, and exposing calculated BPM and SpO2 values over HTTP.

## Run

```bash
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8080
```

The backend loads `.env` automatically. `DATABASE_URL` or `POSTGRES_DSN` enables PostgreSQL
persistence. If a full DSN is not set, the app can build one from `POSTGRES_HOST`,
`POSTGRES_PORT`, `POSTGRES_PASSWORD`, `POSTGRES_USER`, and `POSTGRES_DB`; the default database
user is `vitkarino` and the default database name is `pulso`.

On startup the app creates the `users`, `projects`, `project_users`, `recordings`, and
`recordings_samples` tables when they do not exist.

WebSocket input:

```text
ws://localhost:8080/ws/devices/{device_id}
```

Live signal broadcast for UI charts:

```text
ws://localhost:8080/ws/measurements/{measurement_id}/stream
```

During an active measurement, `/ws/measurements/{measurement_id}/stream` broadcasts each processed device batch.
The `samples` array contains the filtered IR/red signal for the latest device batch:

```json
{
  "type": "measurement_sample_batch",
  "measurement_id": "measurement uuid",
  "device_id": "F4:65:0B:55:2E:80",
  "fs": 25,
  "temperature": 25.6,
  "sample_count": 25,
  "samples": [
    {
      "ir": 124.37,
      "r": 81.42
    }
  ],
  "metrics": {
    "device_id": "F4:65:0B:55:2E:80",
    "time": {
      "measured_at": "2026-04-28T11:51:20.778832Z"
    },
    "sample_rate": 25.0,
    "sensor_temp": 25.6,
    "bpm": 71.4,
    "spo2": 98.0,
    "ratio": 0.55,
    "sensor_confidence": 0.95,
    "signal_quality": {
      "level": "high",
      "reason": null,
      "peak_count": 15,
      "window_seconds": 10.0,
      "perfusion_index": 2.8,
      "samples_in_window": 250
    }
  }
}
```

Latest metrics:

```text
GET http://localhost:8080/api/devices
GET http://localhost:8080/api/devices/{device_id}/metrics
DELETE http://localhost:8080/api/devices/{device_id}
```

`GET /api/devices` returns v3.1 device objects:

```json
{
  "devices": [
    {
      "id": "sim-device-001",
      "is_simulated": true,
      "connection_status": "connected",
      "metrics": {
        "device_id": "sim-device-001",
        "time": {
          "measured_at": "2026-04-28T11:51:20.778832Z"
        },
        "sample_rate": 25.0,
        "sensor_temp": 30.9,
        "bpm": 88.2,
        "spo2": 95.1,
        "ratio": 0.5977,
        "sensor_confidence": 0.959,
        "signal_quality": {
          "level": "high",
          "reason": null,
          "peak_count": 15,
          "window_seconds": 10.0,
          "perfusion_index": 2.8995,
          "samples_in_window": 250
        }
      }
    }
  ],
  "limit": 100,
  "offset": 0
}
```

Start a fixed-time measurement:

```text
POST http://localhost:8080/api/devices/{device_id}/measurements
POST http://localhost:8080/api/measurements/{id}/stop
GET  http://localhost:8080/api/measurements/{id}
DELETE http://localhost:8080/api/measurements/{id}
GET  http://localhost:8080/api/measurements?limit=100&offset=0&date_from=2026-04-01&date_to=2026-04-26
GET  http://localhost:8080/api/measurements/{id}/samples?limit=1000&offset=0
GET  http://localhost:8080/api/measurements/{id}/export?format=json
GET  http://localhost:8080/api/measurements/{id}/export?format=csv
```

Measurement start request:

```json
{
  "duration_s": 15,
  "user_name": "tester",
  "user_id": "1",
  "project_name": "Demo",
  "project_id": "1"
}
```

All API date-time fields use UTC ISO-8601 strings with a `Z` suffix, for example
`2026-04-28T11:51:20.778832Z`. `date_from` and `date_to` filter by `started_at`;
date-only values use UTC day bounds.

Measurement responses use the v3.1 measurement object:

```json
{
  "id": "6883cd24-dc32-404a-9081-60d49e376a0b",
  "is_simulated": true,
  "user_id": "1",
  "project_id": "1",
  "device_id": "sim-device-001",
  "time": {
    "started_at": "2026-04-28T11:51:11.734138Z",
    "finished_at": "2026-04-28T11:51:20.778911Z",
    "duration_ms": 9045
  },
  "status": "completed",
  "channels": ["ir", "red"],
  "sample_rate": 25.0,
  "sensor_temp": 30.9,
  "bpm": 88.2,
  "spo2": 95.1,
  "ratio": 0.5977,
  "sensor_confidence": 0.959,
  "signal_quality": {
    "level": "high",
    "reason": null,
    "peak_count": 15,
    "window_seconds": 10.0,
    "perfusion_index": 2.8995,
    "samples_in_window": 250
  }
}
```

Projects API:

```text
GET http://localhost:8080/api/projects?limit=100&offset=0
PATCH http://localhost:8080/api/projects/{project_id}
DELETE http://localhost:8080/api/projects/{project_id}
```

Project patch request:

```json
{
  "title": "Demo",
  "description": "Updated description"
}
```

Users API:

```text
GET http://localhost:8080/api/users?project_id=1&limit=100&offset=0
PATCH http://localhost:8080/api/users/{user_id}
DELETE http://localhost:8080/api/users/{user_id}
```

User patch request:

```json
{
  "name": "Alice",
  "age": 31,
  "sex": "female"
}
```

Healthcheck:

```text
GET http://localhost:8080/api/health
```

All v3.1 HTTP JSON responses use this envelope:

```json
{
  "data": {},
  "$meta": {
    "status": "success",
    "time": {
      "iso": "2026-02-24T22:03:55Z"
    }
  }
}
```

## Input Format

```json
{
  "device": {
    "id": "A0:B7:65:12:34:56",
    "temp": 31.75,
    "fs": 25,
    "samples": [
      {
        "ir": 84231,
        "r": 53211
      }
    ]
  }
}
```

## Signal Processing Notes

The implementation uses a robust sliding-window PPG pipeline:

- Mean-centering before filtering to remove the DC offset.
- Butterworth band-pass filtering for the physiological heart-rate band.
- Peak detection on the filtered IR channel, with median inter-beat interval for BPM.
- Ratio-of-ratios SpO2 estimation from raw DC and RMS AC of the filtered red/IR channels.
- SpO2 uses its own stable RMS window threshold (`STABLE_SPO2_WINDOW_SECONDS`) and ignores the first
  `SPO2_WARMUP_CUT_SECONDS` seconds of that RMS window to avoid startup jumps.

SpO2 is an estimate and must be calibrated per optical sensor, LED current, placement, and enclosure before any real clinical use.

`sensor_confidence` is emitted in every metrics JSON response as a `0.0` to `1.0` score. It increases when the backend has enough window data, stable pulse intervals, a usable perfusion index, enough detected peaks, and a plausible red/IR ratio.

SpO2 is only emitted when the perfusion index is high enough for oxygen estimation. `MIN_PERFUSION_INDEX` controls contact detection, while `MIN_SPO2_PERFUSION_INDEX` controls whether oxygen is reliable enough to report. `SPO2_OFFSET` can be used after calibration against a reference pulse oximeter.

Measurements are session-based by default. Start a session, keep sending samples to `ws://localhost:8080/ws/devices/{device_id}`, and the backend will collect `MEASUREMENT_DURATION_SECONDS` seconds of data, then expose one final result using the v3.1 measurement JSON shape. The default duration is 15 seconds.

When no finger/contact is detected, vital values are not emitted as valid readings:

```json
{
  "bpm": null,
  "spo2": null,
  "ratio": null,
  "sensor_confidence": 0.0,
  "signal_quality": {
    "level": "no_contact",
    "peak_count": 0,
    "window_seconds": 10.0,
    "perfusion_index": null,
    "samples_in_window": 250,
    "reason": "finger not detected: optical signal is unstable or exposed"
  }
}
```
