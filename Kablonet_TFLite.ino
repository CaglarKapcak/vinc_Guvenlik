#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>

#include "model.h"  // TensorFlow Lite modelimiz

// Sensör kütüphaneleri
#include <ACS712.h>
#include <Wire.h>
#include <EEPROM.h>
#include <SPI.h>
#include <SD.h>

// ESP32 için WiFi kütüphaneleri
#ifdef ESP32
  #include <WiFi.h>
  #include <HTTPClient.h>
#endif

// Pin tanımlamaları
#define NUM_SENSORS 3
#define SENSOR_PIN_1 A0
#define SENSOR_PIN_2 A1
#define SENSOR_PIN_3 A2
#define TEMP_SENSOR_PIN A3
#define ALARM_PIN 7
#define LED_PIN 6
#define SD_CS_PIN 10

// Ağ ayarları
const char* ssid = "KABLONET_AI";
const char* password = "guvenlik123";
const char* serverURL = "http://192.168.1.100:5000/api/anomaly";

// TensorFlow Lite globalleri
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
constexpr int kTensorArenaSize = 30 * 1024;
uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

// Sensör nesneleri
ACS712 sensors[] = {
  ACS712(ACS712_05B, SENSOR_PIN_1),
  ACS712(ACS712_05B, SENSOR_PIN_2),
  ACS712(ACS712_05B, SENSOR_PIN_3)
};

// Global değişkenler
float baselineValues[NUM_SENSORS];
float thresholds[NUM_SENSORS];
float temperatureHistory[24];  // Son 24 saatlik sıcaklık
int tempIndex = 0;
bool isCalibrated = false;
bool sdCardAvailable = false;
int anomalyCount = 0;

// Model için giriş verisi dizisi
float sensorReadings[NUM_SENSORS * 10];  // Son 10 örnek

void setup() {
  Serial.begin(115200);
  while (!Serial);
  
  pinMode(ALARM_PIN, OUTPUT);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);
  
  // SD kart başlatma
  if (SD.begin(SD_CS_PIN)) {
    sdCardAvailable = true;
    Serial.println("SD kart başlatıldı.");
  } else {
    Serial.println("SD kart hatası!");
  }
  
  // WiFi bağlantısı (ESP32 için)
  #ifdef ESP32
    setupWiFi();
  #endif
  
  // TensorFlow Lite başlatma
  setupTFLite();
  
  // Kalibrasyon verilerini yükle
  loadCalibration();
  
  // Kalibre edilmemişse kalibrasyon yap
  if (!isCalibrated) {
    calibrateSensors();
  }
  
  Serial.println("Sistem hazır. Manyetik alan izleniyor...");
}

void loop() {
  static unsigned long lastSampleTime = 0;
  static unsigned long lastReportTime = 0;
  static int sampleCount = 0;
  
  // 100ms'de bir örnek al
  if (millis() - lastSampleTime >= 100) {
    lastSampleTime = millis();
    
    // Sensör verilerini oku ve işle
    readAndProcessSensors();
    
    // TensorFlow Lite ile anomali tespiti
    bool anomalyDetected = detectAnomalyWithTFLite();
    
    if (anomalyDetected) {
      anomalyCount++;
      triggerAlarm();
      logAnomaly();
      
      // Sunucuya bildirim gönder (WiFi varsa)
      #ifdef ESP32
        sendAnomalyAlert();
      #endif
    }
    
    sampleCount++;
  }
  
  // Her 60 saniyede bir sistem durum raporu
  if (millis() - lastReportTime >= 60000) {
    lastReportTime = millis();
    generateSystemReport();
    
    // Sıcaklık kaydı
    float currentTemp = readTemperature();
    temperatureHistory[tempIndex] = currentTemp;
    tempIndex = (tempIndex + 1) % 24;
    
    // Sıcaklık değişimini kontrol et
    checkTemperatureDrift();
  }
}

// TensorFlow Lite başlatma fonksiyonu
void setupTFLite() {
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Modeli yükle
  model = tf2::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model version does not match Schema");
    return;
  }

  // Tüm operasyonları çözümle
  static tflite::AllOpsResolver resolver;

  // Interpreter oluştur
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Tensorları ayır
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }

  // Giriş ve çıkış tensorlarını al
  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("TensorFlow Lite başlatıldı.");
}

// WiFi bağlantısı (ESP32 için)
#ifdef ESP32
void setupWiFi() {
  Serial.print("WiFi bağlanıyor: ");
  Serial.println(ssid);
  
  WiFi.begin(ssid, password);
  int attempts = 0;
  
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    Serial.print(".");
    attempts++;
  }
  
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\nWiFi bağlandı!");
    Serial.print("IP adresi: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("\nWiFi bağlantı hatası!");
  }
}
#endif

// Sensör verilerini oku ve işle
void readAndProcessSensors() {
  static int readingIndex = 0;
  
  for (int i = 0; i < NUM_SENSORS; i++) {
    float reading = sensors[i].getCurrentAC();
    
    // Dairesel tamponda sakla (son 10 örnek)
    sensorReadings[i * 10 + readingIndex] = reading;
  }
  
  readingIndex = (readingIndex + 1) % 10;
}

// TensorFlow Lite ile anomali tespiti
bool detectAnomalyWithTFLite() {
  // Giriş verisini hazırla
  for (int i = 0; i < NUM_SENSORS * 10; i++) {
    input->data.f[i] = sensorReadings[i];
  }
  
  // Çıkarım yap
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed");
    return false;
  }
  
  // Çıktıyı değerlendir (0-1 arası anomali skoru)
  float anomaly_score = output->data.f[0];
  
  // Eşik değerini aşıyorsa anomali var
  if (anomaly_score > 0.8) {
    Serial.print("Anomali tespit edildi! Skor: ");
    Serial.println(anomaly_score);
    return true;
  }
  
  return false;
}

// Alarm tetikleme
void triggerAlarm() {
  digitalWrite(ALARM_PIN, HIGH);
  digitalWrite(LED_PIN, HIGH);
  
  // Alarmı 3 saniye sonra kapat
  delay(3000);
  digitalWrite(ALARM_PIN, LOW);
  digitalWrite(LED_PIN, LOW);
}

// Anomali kaydı
void logAnomaly() {
  String logEntry = "Anomali," + String(millis()) + ",";
  for (int i = 0; i < NUM_SENSORS; i++) {
    logEntry += String(sensors[i].getCurrentAC()) + ",";
  }
  logEntry += String(readTemperature());
  
  Serial.println(logEntry);
  
  // SD karta yaz (varsa)
  if (sdCardAvailable) {
    File logFile = SD.open("anomali.log", FILE_WRITE);
    if (logFile) {
      logFile.println(logEntry);
      logFile.close();
    }
  }
}

// Sistem raporu oluştur
void generateSystemReport() {
  String report = "Rapor," + String(millis()) + 
                  ",Anomali:" + String(anomalyCount) +
                  ",Sıcaklık:" + String(readTemperature());
  
  Serial.println(report);
  
  if (sdCardAvailable) {
    File reportFile = SD.open("system.log", FILE_WRITE);
    if (reportFile) {
      reportFile.println(report);
      reportFile.close();
    }
  }
}

// Kalibrasyon fonksiyonları
void calibrateSensors() {
  Serial.println("Kalibrasyon başlıyor... Lütfen kablo normal durumda olsun.");
  delay(3000);
  
  for (int i = 0; i < NUM_SENSORS; i++) {
    baselineValues[i] = getAverageReading(i, 100);  // 100 örnek ortalaması
    thresholds[i] = calculateThreshold(i);
  }
  
  saveCalibration();
  isCalibrated = true;
  Serial.println("Kalibrasyon tamamlandı.");
}

float getAverageReading(int sensorIndex, int samples) {
  float sum = 0;
  for (int i = 0; i < samples; i++) {
    sum += sensors[sensorIndex].getCurrentAC();
    delay(10);
  }
  return sum / samples;
}

float calculateThreshold(int sensorIndex) {
  // Standart sapma hesaplama
  float mean = baselineValues[sensorIndex];
  float sumSq = 0;
  
  for (int i = 0; i < 50; i++) {
    float reading = sensors[sensorIndex].getCurrentAC();
    sumSq += (reading - mean) * (reading - mean);
    delay(10);
  }
  
  float stdDev = sqrt(sumSq / 50);
  return stdDev * 3.0;  // 3 sigma kuralı
}

// EEPROM işlemleri
void saveCalibration() {
  int addr = 0;
  EEPROM.put(addr, isCalibrated);
  addr += sizeof(isCalibrated);
  
  for (int i = 0; i < NUM_SENSORS; i++) {
    EEPROM.put(addr, baselineValues[i]);
    addr += sizeof(baselineValues[i]);
    EEPROM.put(addr, thresholds[i]);
    addr += sizeof(thresholds[i]);
  }
  
  #ifdef ESP32
    EEPROM.commit();
  #endif
}

void loadCalibration() {
  int addr = 0;
  EEPROM.get(addr, isCalibrated);
  addr += sizeof(isCalibrated);
  
  if (isCalibrated) {
    for (int i = 0; i < NUM_SENSORS; i++) {
      EEPROM.get(addr, baselineValues[i]);
      addr += sizeof(baselineValues[i]);
      EEPROM.get(addr, thresholds[i]);
      addr += sizeof(thresholds[i]);
    }
    Serial.println("Kalibrasyon verileri yüklendi.");
  }
}

// Sıcaklık okuma ve kontrol
float readTemperature() {
  int reading = analogRead(TEMP_SENSOR_PIN);
  float voltage = reading * (5.0 / 1023.0);
  return (voltage - 0.5) * 100;  // LM35 için
}

void checkTemperatureDrift() {
  float currentTemp = readTemperature();
  float avgTemp = 0;
  
  for (int i = 0; i < 24; i++) {
    avgTemp += temperatureHistory[i];
  }
  avgTemp /= 24;
  
  // Sıcaklıkta önemli değişim varsa yeniden kalibre et
  if (abs(currentTemp - avgTemp) > 5.0) {
    Serial.println("Sıcaklık değişimi tespit edildi. Yeniden kalibrasyon öneriliyor.");
    // Otomatik kalibrasyon yapılabilir veya kullanıcı uyarılabilir
  }
}

// Sunucuya anomali bildirimi (ESP32 için)
#ifdef ESP32
void sendAnomalyAlert() {
  if (WiFi.status() != WL_CONNECTED) {
    return;
  }
  
  HTTPClient http;
  http.begin(serverURL);
  http.addHeader("Content-Type", "application/json");
  
  String jsonPayload = "{\"anomaly_count\": " + String(anomalyCount) + 
                       ", \"timestamp\": " + String(millis()) + "}";
  
  int httpResponseCode = http.POST(jsonPayload);
  
  if (httpResponseCode > 0) {
    Serial.print("Sunucuya bildirim gönderildi. Durum kodu: ");
    Serial.println(httpResponseCode);
  } else {
    Serial.print("Sunucu hatası: ");
    Serial.println(httpResponseCode);
  }
  
  http.end();
}
#endif
