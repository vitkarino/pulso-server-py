#include <Arduino.h>
#include <ArduinoJson.h>
#include <MAX30105.h>
#include <WebSocketsClient.h>
#include <WiFi.h>
#include <Wire.h>

const char *ssid = "ssid";
const char *password = "pass";
const char *server_ip = "192.168.1.103";
const uint16_t server_port = 8080;
const char *deviceId = "dev_F4:65:0B:55:2E:80";
char wsPath[96];

const byte ledBrightness = 0x30;
const byte sampleAverage = 1;
const byte softwareSampleAverage = 3;
const byte ledMode = 2;
const int hardwareSampleRate = 400;
const float reportedSampleRate = (float)hardwareSampleRate / (float)softwareSampleAverage;
const int pulseWidth = 411;
const int adcRange = 4096;

const size_t batchSize = 64;
const unsigned long wifiReconnectIntervalMs = 5000;
const unsigned long temperatureReadIntervalMs = 5000;

struct PPGSample {
  uint32_t index;
  uint32_t ir;
  uint32_t red;
};

MAX30105 particleSensor;
WebSocketsClient webSocket;

PPGSample sampleBatch[batchSize];
size_t batchCount = 0;

StaticJsonDocument<512> inboundDoc;
StaticJsonDocument<512> controlDoc;
StaticJsonDocument<8192> batchDoc;
char jsonBuffer[8192];

bool wsConnected = false;
bool canStream = false;
char activeMeasurementId[48] = "";
unsigned long streamStartedAtMs = 0;
unsigned long streamDurationMs = 0;
unsigned long lastWiFiReconnectMs = 0;
unsigned long lastTemperatureReadMs = 0;
float lastTemperatureC = NAN;
size_t streamSamplesCollected = 0;
size_t targetStreamSamples = 0;
uint64_t averagedIrSum = 0;
uint64_t averagedRedSum = 0;
byte averagedSampleCount = 0;

void sendHello();
void sendLog(const char *message);
void sendStartAck();
void sendStopAck();
void sendFinished();
bool sendBatch();
void stopStreaming(bool clearMeasurement = true);
void resetSampleAveraging();
void flushAveragedSample();

void connectWiFi() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);

  Serial.print("Connecting WiFi");
  unsigned long startedAt = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - startedAt < 20000) {
    delay(250);
    Serial.print(".");
  }
  Serial.println();

  if (WiFi.status() == WL_CONNECTED) {
    Serial.print("WiFi connected: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("WiFi connect timeout");
  }
}

void reconnectWiFiIfNeeded() {
  if (WiFi.status() == WL_CONNECTED) {
    return;
  }

  stopStreaming(false);

  unsigned long now = millis();
  if (now - lastWiFiReconnectMs < wifiReconnectIntervalMs) {
    return;
  }

  lastWiFiReconnectMs = now;
  Serial.println("WiFi lost. Reconnecting...");
  WiFi.disconnect();
  WiFi.begin(ssid, password);
}

void setupSensor() {
  Wire.begin();

  if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("MAX30102 not found");
    while (true) {
      delay(1000);
    }
  }

  particleSensor.setup(
    ledBrightness,
    sampleAverage,
    ledMode,
    hardwareSampleRate,
    pulseWidth,
    adcRange
  );
  particleSensor.clearFIFO();

  Serial.println("MAX30102 ready");
}

void setupWebSocket() {
  snprintf(wsPath, sizeof(wsPath), "/ws/devices/%s", deviceId);
  webSocket.begin(server_ip, server_port, wsPath);
  webSocket.onEvent(webSocketEvent);
  webSocket.setReconnectInterval(5000);
  webSocket.enableHeartbeat(15000, 3000, 2);
}

void sendControl(const char *type) {
  if (!wsConnected) {
    return;
  }

  controlDoc.clear();
  controlDoc["type"] = type;
  controlDoc["device_id"] = deviceId;
  if (strcmp(type, "hello") == 0) {
    controlDoc["is_simulated"] = false;
  }
  if (activeMeasurementId[0] != '\0') {
    controlDoc["measurement_id"] = activeMeasurementId;
  }

  size_t bytesWritten = serializeJson(controlDoc, jsonBuffer, sizeof(jsonBuffer));
  if (bytesWritten == 0 || bytesWritten >= sizeof(jsonBuffer)) {
    Serial.println("Control JSON serialization failed");
    return;
  }

  webSocket.sendTXT(jsonBuffer, bytesWritten);
}

void sendHello() {
  sendControl("hello");
}

void sendLog(const char *message) {
  if (!wsConnected) {
    return;
  }

  controlDoc.clear();
  controlDoc["type"] = "log";
  controlDoc["device_id"] = deviceId;
  controlDoc["message"] = message;

  size_t bytesWritten = serializeJson(controlDoc, jsonBuffer, sizeof(jsonBuffer));
  if (bytesWritten > 0 && bytesWritten < sizeof(jsonBuffer)) {
    webSocket.sendTXT(jsonBuffer, bytesWritten);
  }
}

void sendStartAck() {
  sendControl("start_ack");
}

void sendStopAck() {
  sendControl("stop_ack");
}

void sendFinished() {
  if (!wsConnected) {
    return;
  }

  controlDoc.clear();
  controlDoc["type"] = "finished";
  controlDoc["device_id"] = deviceId;
  controlDoc["measurement_id"] = activeMeasurementId;
  controlDoc["reason"] = "duration_reached";

  size_t bytesWritten = serializeJson(controlDoc, jsonBuffer, sizeof(jsonBuffer));
  if (bytesWritten > 0 && bytesWritten < sizeof(jsonBuffer)) {
    webSocket.sendTXT(jsonBuffer, bytesWritten);
  }
}

void startStreaming(const char *measurementId, unsigned long durationMs) {
  if (measurementId == nullptr || measurementId[0] == '\0') {
    sendLog("start command without measurement_id");
    return;
  }

  snprintf(activeMeasurementId, sizeof(activeMeasurementId), "%s", measurementId);
  streamDurationMs = durationMs;
  streamStartedAtMs = millis();
  batchCount = 0;
  streamSamplesCollected = 0;
  targetStreamSamples = 0;
  resetSampleAveraging();
  if (durationMs > 0) {
    targetStreamSamples = max((size_t)1, (size_t)ceil((double)durationMs * (double)reportedSampleRate / 1000.0 - 1e-6));
  }
  canStream = true;

  particleSensor.clearFIFO();

  sendStartAck();
  Serial.print("Streaming started: ");
  Serial.println(activeMeasurementId);
}

void stopStreaming(bool clearMeasurement) {
  canStream = false;
  streamDurationMs = 0;
  streamStartedAtMs = 0;
  batchCount = 0;
  streamSamplesCollected = 0;
  targetStreamSamples = 0;
  resetSampleAveraging();

  if (clearMeasurement) {
    activeMeasurementId[0] = '\0';
  }
}

bool sendBatch() {
  if (batchCount == 0) {
    return true;
  }

  if (!wsConnected || !canStream || activeMeasurementId[0] == '\0') {
    batchCount = 0;
    return false;
  }

  batchDoc.clear();
  batchDoc["type"] = "samples";
  batchDoc["device_id"] = deviceId;
  batchDoc["measurement_id"] = activeMeasurementId;
  batchDoc["sample_rate_hz"] = reportedSampleRate;

  unsigned long now = millis();
  if (isnan(lastTemperatureC) || now - lastTemperatureReadMs >= temperatureReadIntervalMs) {
    lastTemperatureC = particleSensor.readTemperature();
    lastTemperatureReadMs = now;
  }
  if (!isnan(lastTemperatureC)) {
    batchDoc["sensor_temp_c"] = lastTemperatureC;
  }

  JsonArray samples = batchDoc["samples"].to<JsonArray>();
  for (size_t i = 0; i < batchCount; i++) {
    JsonObject sample = samples.add<JsonObject>();
    if (sample.isNull()) {
      sendLog("batch JSON capacity exceeded");
      batchCount = 0;
      return false;
    }
    sample["index"] = sampleBatch[i].index;
    sample["ir"] = sampleBatch[i].ir;
    sample["red"] = sampleBatch[i].red;
  }

  size_t needed = measureJson(batchDoc) + 1;
  if (needed > sizeof(jsonBuffer)) {
    sendLog("jsonBuffer too small for batch");
    batchCount = 0;
    return false;
  }

  size_t bytesWritten = serializeJson(batchDoc, jsonBuffer, sizeof(jsonBuffer));
  if (bytesWritten == 0 || bytesWritten >= sizeof(jsonBuffer)) {
    sendLog("batch JSON serialization failed");
    batchCount = 0;
    return false;
  }

  bool sent = webSocket.sendTXT(jsonBuffer, bytesWritten);
  if (!sent) {
    Serial.println("WebSocket send failed");
    stopStreaming(false);
    return false;
  }

  batchCount = 0;
  return true;
}

void addSampleToBatch(uint32_t irValue, uint32_t redValue) {
  if (!canStream) {
    return;
  }

  if (targetStreamSamples > 0 && streamSamplesCollected >= targetStreamSamples) {
    return;
  }

  if (batchCount >= batchSize && !sendBatch()) {
    return;
  }

  if (batchCount >= batchSize) {
    sendLog("batch overflow prevented");
    batchCount = 0;
    return;
  }

  sampleBatch[batchCount].index = streamSamplesCollected;
  sampleBatch[batchCount].ir = irValue;
  sampleBatch[batchCount].red = redValue;
  batchCount++;
  streamSamplesCollected++;

  if (batchCount >= batchSize) {
    sendBatch();
  }
}

void resetSampleAveraging() {
  averagedIrSum = 0;
  averagedRedSum = 0;
  averagedSampleCount = 0;
}

void flushAveragedSample() {
  if (averagedSampleCount == 0) {
    return;
  }

  uint32_t irValue = (uint32_t)((averagedIrSum + averagedSampleCount / 2) / averagedSampleCount);
  uint32_t redValue = (uint32_t)((averagedRedSum + averagedSampleCount / 2) / averagedSampleCount);
  resetSampleAveraging();
  addSampleToBatch(irValue, redValue);
}

void addRawSampleToAverager(uint32_t irValue, uint32_t redValue) {
  averagedIrSum += irValue;
  averagedRedSum += redValue;
  averagedSampleCount++;

  if (averagedSampleCount >= softwareSampleAverage) {
    flushAveragedSample();
  }
}

void collectSamplesFromFIFO() {
  particleSensor.check();

  while (particleSensor.available()) {
    uint32_t irValue = particleSensor.getFIFOIR();
    uint32_t redValue = particleSensor.getFIFORed();

    particleSensor.nextSample();
    addRawSampleToAverager(irValue, redValue);
  }
}

bool streamDurationFinished() {
  if (targetStreamSamples > 0 && streamSamplesCollected >= targetStreamSamples) {
    return true;
  }

  if (streamDurationMs == 0) {
    return false;
  }

  return (unsigned long)(millis() - streamStartedAtMs) >= streamDurationMs + 2000UL;
}

void handleStartCommand(JsonObject payload) {
  const char *measurementId = payload["measurement_id"] | "";
  float durationSeconds = payload["duration"] | 0.0;
  unsigned long durationMs = 0;

  if (durationSeconds > 0.0) {
    durationMs = (unsigned long)(durationSeconds * 1000.0);
  }

  startStreaming(measurementId, durationMs);
}

void handleStopCommand() {
  flushAveragedSample();
  sendBatch();
  sendStopAck();
  stopStreaming(true);
  Serial.println("Streaming stopped by backend");
}

void handleTextMessage(uint8_t *payload, size_t length) {
  inboundDoc.clear();
  DeserializationError error = deserializeJson(inboundDoc, payload, length);
  if (error) {
    Serial.print("Bad WS JSON: ");
    Serial.println(error.c_str());
    return;
  }

  const char *type = inboundDoc["type"] | "";
  if (strcmp(type, "hello_ack") == 0) {
    Serial.println("Backend hello acknowledged");
    return;
  }
  if (strcmp(type, "start") == 0) {
    handleStartCommand(inboundDoc.as<JsonObject>());
    return;
  }
  if (strcmp(type, "stop") == 0) {
    handleStopCommand();
    return;
  }
}

void webSocketEvent(WStype_t type, uint8_t *payload, size_t length) {
  switch (type) {
    case WStype_CONNECTED:
      wsConnected = true;
      Serial.println("WebSocket connected");
      sendHello();
      break;

    case WStype_DISCONNECTED:
      wsConnected = false;
      stopStreaming(false);
      Serial.println("WebSocket disconnected");
      break;

    case WStype_TEXT:
      handleTextMessage(payload, length);
      break;

    default:
      break;
  }
}

void setup() {
  Serial.begin(115200);
  delay(100);

  setupSensor();
  connectWiFi();
  setupWebSocket();
}

void loop() {
  reconnectWiFiIfNeeded();

  if (WiFi.status() != WL_CONNECTED) {
    delay(10);
    return;
  }

  webSocket.loop();

  if (!wsConnected || !canStream) {
    delay(5);
    return;
  }

  if (streamDurationFinished()) {
    if (targetStreamSamples == 0 || streamSamplesCollected < targetStreamSamples) {
      flushAveragedSample();
    }
    sendBatch();
    sendFinished();
    stopStreaming(true);
    Serial.println("Local duration finished");
    return;
  }

  collectSamplesFromFIFO();
}
