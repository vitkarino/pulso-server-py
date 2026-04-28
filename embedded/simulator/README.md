# Pulso Embedded Simulator

Python simulator for the ESP32 device WebSocket protocol.

Run the backend first:

```bash
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8080
```

Then run the simulator:

```bash
python3 embedded/simulator/simulator.py --port 8080 --device-id sim-device-001
```

Start a measurement from Postman:

```text
POST http://127.0.0.1:8080/api/devices/sim-device-001/measurements
```

Body:

```json
{
  "duration_s": 15,
  "user_id": "1",
  "project_id": "1"
}
```

The simulator will receive the `start` command, send synthetic PPG batches with `measurement_id`,
and then send `finished`.

Useful options:

```bash
python3 embedded/simulator/simulator.py --port 8080 --device-id sim-device-001 --bpm 80 --fs 25
```

For a quick stream without calling the HTTP start endpoint:

```bash
python3 embedded/simulator/simulator.py --port 8080 --auto-start --auto-duration-s 15
```
