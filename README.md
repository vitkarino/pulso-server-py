# Pulso Python Backend

Backend for receiving PPG samples over WebSocket, filtering the red/IR signal, and exposing calculated BPM and SpO2 values over HTTP.

## Run

```bash
export DATABASE_URL="postgresql+psycopg://postgres:postgres@localhost:5432/pulso"
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8080
```

`DATABASE_URL` enables PostgreSQL persistence for recordings. On startup the app creates the
`recordings` and `recordings_samples` tables when they do not exist.

WebSocket input:

```text
ws://localhost:8080/ws
```

Latest metrics:

```text
GET http://localhost:8080/metrics
GET http://localhost:8080/metrics/{device_id}
```

Start a fixed-time measurement:

```text
POST http://localhost:8080/measurements/{device_id}/start
POST http://localhost:8080/measurements/{device_id}/start?duration_seconds=15
GET http://localhost:8080/measurements/{device_id}
GET http://localhost:8080/measurements
```

Recordings API:

```text
POST http://localhost:8080/api/recordings/start
POST http://localhost:8080/api/recordings/start?duration=15
POST http://localhost:8080/api/recordings/{id}/stop
POST http://localhost:8080/api/recordings/stop-all
GET  http://localhost:8080/api/recordings/{id}
GET  http://localhost:8080/api/recordings?limit=100&offset=0&date_from=2026-04-01&date_to=2026-04-26
GET  http://localhost:8080/api/recordings/{id}/samples?limit=1000&offset=0
GET  http://localhost:8080/api/recordings/{id}/extract?format=json
GET  http://localhost:8080/api/recordings/{id}/extract?format=csv
GET  http://localhost:8080/api/recordings/extract?format=json
GET  http://localhost:8080/api/recordings/extract?format=csv
```

`date_from` and `date_to` filter by `started_at`. Date-only values use UTC day bounds.

Healthcheck:

```text
GET http://localhost:8080/health
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

- Hampel outlier suppression for single-sample spikes.
- Linear detrending to remove baseline drift.
- Butterworth band-pass filtering for the physiological heart-rate band.
- Peak detection on the filtered IR channel, with median inter-beat interval for BPM.
- Ratio-of-ratios SpO2 estimation from red and IR AC/DC components.

SpO2 is an estimate and must be calibrated per optical sensor, LED current, placement, and enclosure before any real clinical use.

`sensor_confidence` is emitted in every metrics JSON response as a `0.0` to `1.0` score. It increases when the backend has enough window data, stable pulse intervals, a usable perfusion index, enough detected peaks, and a plausible red/IR ratio.

SpO2 is only emitted when the perfusion index is high enough for oxygen estimation. `MIN_PERFUSION_INDEX` controls contact detection, while `MIN_SPO2_PERFUSION_INDEX` controls whether oxygen is reliable enough to report. `SPO2_OFFSET` can be used after calibration against a reference pulse oximeter.

Measurements are session-based by default. Start a session, keep sending samples to `ws://localhost:8080/ws`, and the backend will collect `MEASUREMENT_DURATION_SECONDS` seconds of data, then emit one final result using the same `VitalSigns` JSON shape as `/metrics`. The default duration is 15 seconds.

When no finger/contact is detected, vital values are not emitted as valid readings:

```json
{
  "bpm": null,
  "spo2": null,
  "ratio": null,
  "sensor_confidence": 0.0,
  "signal_quality": {
    "level": "no_contact",
    "reason": "finger not detected: optical signal is unstable or exposed"
  }
}
```
