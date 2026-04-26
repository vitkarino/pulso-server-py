#include <Arduino.h>
#include <ArduinoJson.h>
#include <MAX30105.h>
#include <WebSocketsClient.h>
#include <WiFi.h>
#include <Wire.h>

const char *ssid = "YOUR_WIFI_SSID";
const char *password = "YOUR_WIFI_PASSWORD";
const char *server_ip = "192.168.1.100";
const uint16_t server_port = 8080;
const char *ws_path = "/ws/esp32";
const char *deviceId = "F4:65:0B:55:2E:80";

const byte ledBrightness = 0x1F;
const byte sampleAverage = 4;
const byte ledMode = 2;
const int hardwareSampleRate = 100;
const int reportedSampleRate = 25;
const int pulseWidth = 411;
const int adcRange = 4096;

const size_t batchSize = 25;
const unsigned long wifiReconnectIntervalMs = 5000;
const unsigned long temperatureReadIntervalMs = 5000;

struct PPGSample {
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
char activeRecordingId[40] = "";
unsigned long streamStartedAtMs = 0;
unsigned long streamDurationMs = 0;
unsigned long lastWiFiReconnectMs = 0;
unsigned long lastTemperatureReadMs = 0;
float lastTemperatureC = NAN;
size_t streamSamplesCollected = 0;
size_t targetStreamSamples = 0;

void sendHello();
void sendLog(const char *message);
void sendStartAck();
void sendStopAck();
void sendFinished();
bool sendBatch();
void stopStreaming(bool clearRecording = true);

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
  webSocket.begin(server_ip, server_port, ws_path);
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
  if (activeRecordingId[0] != '\0') {
    controlDoc["recording_id"] = activeRecordingId;
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
  sendControl("finished");
}

void startStreaming(const char *recordingId, unsigned long durationMs) {
  if (recordingId == nullptr || recordingId[0] == '\0') {
    sendLog("start command without recording_id");
    return;
  }

  snprintf(activeRecordingId, sizeof(activeRecordingId), "%s", recordingId);
  streamDurationMs = durationMs;
  streamStartedAtMs = millis();
  batchCount = 0;
  streamSamplesCollected = 0;
  targetStreamSamples = 0;
  if (durationMs > 0) {
    targetStreamSamples = max((size_t)1, (size_t)((durationMs * (unsigned long)reportedSampleRate + 999UL) / 1000UL));
  }
  canStream = true;

  particleSensor.clearFIFO();

  sendStartAck();
  Serial.print("Streaming started: ");
  Serial.println(activeRecordingId);
}

void stopStreaming(bool clearRecording) {
  canStream = false;
  streamDurationMs = 0;
  streamStartedAtMs = 0;
  batchCount = 0;
  streamSamplesCollected = 0;
  targetStreamSamples = 0;

  if (clearRecording) {
    activeRecordingId[0] = '\0';
  }
}

bool sendBatch() {
  if (batchCount == 0) {
    return true;
  }

  if (!wsConnected || !canStream || activeRecordingId[0] == '\0') {
    batchCount = 0;
    return false;
  }

  batchDoc.clear();
  JsonObject device = batchDoc["device"].to<JsonObject>();
  device["id"] = deviceId;
  device["recording_id"] = activeRecordingId;
  device["fs"] = reportedSampleRate;

  unsigned long now = millis();
  if (isnan(lastTemperatureC) || now - lastTemperatureReadMs >= temperatureReadIntervalMs) {
    lastTemperatureC = particleSensor.readTemperature();
    lastTemperatureReadMs = now;
  }
  if (!isnan(lastTemperatureC)) {
    device["temp"] = lastTemperatureC;
  }

  JsonArray samples = device["samples"].to<JsonArray>();
  for (size_t i = 0; i < batchCount; i++) {
    JsonObject sample = samples.add<JsonObject>();
    if (sample.isNull()) {
      sendLog("batch JSON capacity exceeded");
      batchCount = 0;
      return false;
    }
    sample["ir"] = sampleBatch[i].ir;
    sample["r"] = sampleBatch[i].red;
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

  if (batchCount >= batchSize && !sendBatch()) {
    return;
  }

  if (batchCount >= batchSize) {
    sendLog("batch overflow prevented");
    batchCount = 0;
    return;
  }

  sampleBatch[batchCount].ir = irValue;
  sampleBatch[batchCount].red = redValue;
  batchCount++;
  streamSamplesCollected++;

  if (batchCount >= batchSize) {
    sendBatch();
  }
}

void collectSamplesFromFIFO() {
  particleSensor.check();

  while (particleSensor.available()) {
    uint32_t irValue = particleSensor.getFIFOIR();
    uint32_t redValue = particleSensor.getFIFORed();

    particleSensor.nextSample();
    addSampleToBatch(irValue, redValue);
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
  const char *recordingId = payload["recording_id"] | "";
  float durationSeconds = payload["duration"] | 0.0;
  unsigned long durationMs = 0;

  if (durationSeconds > 0.0) {
    durationMs = (unsigned long)(durationSeconds * 1000.0);
  }

  startStreaming(recordingId, durationMs);
}

void handleStopCommand() {
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
    sendBatch();
    sendFinished();
    stopStreaming(true);
    Serial.println("Local duration finished");
    return;
  }

  collectSamplesFromFIFO();
}
