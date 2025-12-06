#ifndef COMMON_H
#define COMMON_H

#include <Arduino.h>

// โครงสร้างข้อมูลที่ส่งผ่าน ESP-NOW
typedef struct struct_message {
    uint8_t id;         // 1=Tx1, 2=Tx2, 3=Tx3
    uint32_t msgId;     // Packet Counter (เพื่อดูว่ามี Packet loss ไหม)
} struct_message;

// สีสำหรับ RGB LED
#define COLOR_OFF    0
#define COLOR_RED    0x001000  // G R B format (ขึ้นอยู่กับบอร์ด)
#define COLOR_GREEN  0x100000
#define COLOR_BLUE   0x000010
#define COLOR_WHITE  0x101010

#endif