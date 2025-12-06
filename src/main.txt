/**
 * RECEIVER FIRMWARE (Strict Sync Mode)
 * - ไม่มีการใช้ค่า Default (0 หรือ -100)
 * - รอจนกว่าจะได้ค่าจริงจาก Tx ครบทั้ง 3 ตัว จึงจะแสดงผล
 * - เคลียร์ค่าทิ้งทุกครั้งเมื่อแสดงผลเสร็จ เพื่อเริ่มรอบใหม่
 */
#include <Arduino.h>
#include <esp_now.h>
#include <WiFi.h>
#include <esp_wifi.h>
#include "common.h"

// --- Config RGB LED ---
#ifdef RGB_BUILTIN
  #undef RGB_BUILTIN
#endif
#define RGB_BUILTIN 48

// MAC Address ของ Tx ทั้ง 3 ตัว
const uint8_t MacTx1[] = {0x9C, 0x13, 0x9E, 0x92, 0x6C, 0x98};
const uint8_t MacTx2[] = {0xD0, 0xCF, 0x13, 0x15, 0x8E, 0x30};
const uint8_t MacTx3[] = {0x9C, 0x13, 0x9E, 0x90, 0xC7, 0x08};

// --- ตัวแปร Global สำหรับ Sniffer ---
// ใช้เก็บค่าสดๆ จาก Hardware แต่ยังไม่ถือว่า "ได้รับ" จนกว่าจะผ่าน Process
volatile int8_t raw_rssi_from_sniffer[4] = {0}; 

// Queue สำหรับส่งสัญญาณบอก Main Loop
typedef struct {
    uint8_t sender_id;
    int8_t rssi;
} QueueData;

QueueHandle_t rssiQueue;

// ฟังก์ชันเปรียบเทียบ MAC
bool isMacEqual(const uint8_t *mac1, const uint8_t *mac2) {
    return memcmp(mac1, mac2, 6) == 0;
}

// --- SNIFFER FUNCTION (ด่านหน้า รับข้อมูลดิบ) ---
void promiscuous_rx_cb(void* buf, wifi_promiscuous_pkt_type_t type) {
    if (type != WIFI_PKT_MGMT && type != WIFI_PKT_DATA) return;
    wifi_promiscuous_pkt_t *p = (wifi_promiscuous_pkt_t*)buf;
    uint8_t *sourceMac = p->payload + 10;
    
    // เมื่อเจอ MAC ที่รู้จัก ให้บันทึกค่า RSSI ของจริง ณ เวลานั้นทันที
    if (isMacEqual(sourceMac, MacTx1)) {
        raw_rssi_from_sniffer[1] = p->rx_ctrl.rssi;
    } else if (isMacEqual(sourceMac, MacTx2)) {
        raw_rssi_from_sniffer[2] = p->rx_ctrl.rssi;
    } else if (isMacEqual(sourceMac, MacTx3)) {
        raw_rssi_from_sniffer[3] = p->rx_ctrl.rssi;
    }
}

// --- ESP-NOW CALLBACK (ตัวกระตุ้นจังหวะ) ---
void OnDataRecv(const uint8_t * mac, const uint8_t *incomingData, int len) {
    uint8_t senderId = 0;
    if (isMacEqual(mac, MacTx1)) senderId = 1;
    else if (isMacEqual(mac, MacTx2)) senderId = 2;
    else if (isMacEqual(mac, MacTx3)) senderId = 3;

    if (senderId != 0) {
        // ดึงค่า RSSI ของจริง ที่ Sniffer เพิ่งจับได้เมื่อเสี้ยววินาทีที่แล้ว
        int8_t real_rssi = raw_rssi_from_sniffer[senderId];

        // ถ้าค่าเป็น 0 แสดงว่า Sniffer อาจจะยังจับไม่ได้ในจังหวะนี้ -> ไม่เอา! ไม่ส่ง!
        if (real_rssi != 0) {
            QueueData qData;
            qData.sender_id = senderId;
            qData.rssi = real_rssi;
            xQueueSendFromISR(rssiQueue, &qData, NULL);
        }
    }
}

void setup() {
    Serial.begin(115200);
    pinMode(RGB_BUILTIN, OUTPUT);
    neopixelWrite(RGB_BUILTIN, 0, 0, 0);

    rssiQueue = xQueueCreate(20, sizeof(QueueData));

    WiFi.mode(WIFI_STA);
    esp_wifi_set_channel(1, WIFI_SECOND_CHAN_NONE);

    // Setup Sniffer
    wifi_promiscuous_filter_t filt = { 
        .filter_mask = WIFI_PROMIS_FILTER_MASK_MGMT | WIFI_PROMIS_FILTER_MASK_DATA 
    };
    esp_wifi_set_promiscuous_filter(&filt);
    esp_wifi_set_promiscuous(true);
    esp_wifi_set_promiscuous_rx_cb(&promiscuous_rx_cb);

    if (esp_now_init() != ESP_OK) return;
    esp_now_register_recv_cb(OnDataRecv);
    
    Serial.println("Rx Ready: Strict Real-Data Mode.");
}

// --- ตัวแปรสำหรับ Main Loop (Buffer พักข้อมูล) ---
int8_t buffer_rssi[4];       // ที่พักข้อมูล
bool flag_received[4] = {false, false, false, false}; // ตัวเช็คว่าได้ของหรือยัง

void loop() {
    QueueData qData;
    
    // รอข้อมูลจาก Queue
    if (xQueueReceive(rssiQueue, &qData, portMAX_DELAY) == pdTRUE) {
        
        // 1. รับข้อมูลจริงมาใส่ Buffer
        // ต่อให้รับ Tx1 มาแล้ว แต่ถ้ามี Tx1 มาใหม่ที่สดกว่า ก็จะทับลงไปเลย (Update Latest)
        buffer_rssi[qData.sender_id] = qData.rssi;
        
        // 2. ยกธงว่า "ฉันมีข้อมูลของ ID นี้แล้วนะ"
        flag_received[qData.sender_id] = true;

        // 3. เช็คเงื่อนไขศักดิ์สิทธิ์: ครบ 3 ตัวหรือยัง?
        if (flag_received[1] == true && 
            flag_received[2] == true && 
            flag_received[3] == true) {
            
            // --- เงื่อนไขเป็นจริง: มีของครบ 3 อย่าง และเป็นของจริงทั้งหมด ---
            
            Serial.printf("{\"rssi_1\":%d, \"rssi_2\":%d, \"rssi_3\":%d}\n", 
                          buffer_rssi[1], buffer_rssi[2], buffer_rssi[3]);

            // ไฟเขียว: ภารกิจรอบนี้เสร็จสิ้น
            neopixelWrite(RGB_BUILTIN, 10, 0, 0); 
            //delay(5);
            neopixelWrite(RGB_BUILTIN, 0, 0, 0);

            // 4. RESET ระบบ! (สำคัญมาก)
            // เอาธงลงทั้งหมด เพื่อบังคับให้รอบหน้าต้องรอรับใหม่ให้ครบ 3 ตัวอีกครั้ง
            flag_received[1] = false;
            flag_received[2] = false;
            flag_received[3] = false;
            
            // (ค่าใน buffer_rssi ไม่ต้องลบก็ได้ เพราะเราใช้ flag_received เป็นตัวกั้น
            // เราจะไม่มีวันอ่านค่าเก่าออกมาได้จนกว่า flag จะเป็น true ซึ่ง flag จะ true ก็ต่อเมื่อมีค่าใหม่มาทับ)
        }
    }
}