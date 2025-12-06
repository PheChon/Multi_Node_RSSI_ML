import serial
import json
import time
import csv
import os
import threading
from collections import deque

# --- CONFIGURATION (ตั้งค่าระบบ) ---
SERIAL_PORT = 'COM10'   # <--- เช็ค Port ของ Rx ให้ถูกต้อง
BAUD_RATE = 115200
FILENAME = 'fingerprint_dataset_400.csv'  # ชื่อไฟล์ที่จะบันทึก
SAMPLES_PER_BATCH = 400  # เก็บข้อมูล 400 ค่า ต่อการกด 1 ครั้ง
COUNTDOWN_SEC = 15        # ให้เวลาเดินออกมา 5 วินาที (เผื่อเวลาเดิน)

# พิกัดอ้างอิง P1-P9 (สำหรับแสดงผลให้ดูง่ายๆ)
REF_POINTS = {
    "P1": "(0.35, 0.35)", "P2": "(0.71, 0.35)", "P3": "(1.06, 0.35)",
    "P4": "(0.35, 0.71)", "P5": "(0.71, 0.71)", "P6": "(1.06, 0.71)",
    "P7": "(0.35, 1.06)", "P8": "(0.71, 1.06)", "P9": "(1.06, 1.06)"
}

# --- GLOBAL VARIABLES ---
serial_q = deque()
is_running = True

# --- SERIAL READER THREAD ---
# แยก Thread อ่านข้อมูล เพื่อให้ไม่พลาดข้อมูลแม้แต่วินาทีเดียว
def serial_reader():
    global is_running
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print(f"[System] Connected to {SERIAL_PORT}")
        while is_running:
            try:
                if ser.in_waiting > 0:
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                    # กรองเฉพาะ JSON ที่สมบูรณ์
                    if line.startswith('{') and '}' in line:
                        serial_q.append(line)
            except Exception:
                pass
    except Exception as e:
        print(f"[Error] Serial connection failed: {e}")
        is_running = False

# --- MAIN PROGRAM ---
def main():
    global is_running
    
    # 1. สร้าง/เปิดไฟล์ CSV
    file_exists = os.path.exists(FILENAME)
    try:
        f = open(FILENAME, 'a', newline='')
        writer = csv.writer(f)
        if not file_exists:
            # เขียนหัวตาราง (Header)
            writer.writerow(['Tx1', 'Tx2', 'Tx3', 'Position_Label', 'Coordinate_Hint'])
            print(f"[System] Created new file: {FILENAME}")
        else:
            print(f"[System] Appending to existing file: {FILENAME}")
    except Exception as e:
        print(f"[Error] Cannot open file: {e}")
        return

    # เริ่มอ่าน Serial
    t = threading.Thread(target=serial_reader)
    t.daemon = True
    t.start()
    time.sleep(2) # รอเชื่อมต่อ

    if not is_running:
        return

    print("\n" + "="*50)
    print(f"   DATA COLLECTION (Batch: {SAMPLES_PER_BATCH} samples)")
    print("="*50)
    print("Reference Coordinates:")
    for k, v in REF_POINTS.items():
        print(f"  {k}: {v}", end="  ")
        if k in ['P3', 'P6', 'P9']: print() # ขึ้นบรรทัดใหม่สวยๆ
    print("-" * 50)

    try:
        while True:
            # 2. ระบุจุดที่ยืน (Label)
            print("\n>>> Enter Position Label (e.g., P1) or 'q' to quit:")
            label = input("Label: ").strip().upper()
            
            if label == 'Q': 
                break
            
            # หาพิกัดเพื่อบันทึกเป็น Hint (ถ้ามีในรายการ)
            coord_hint = REF_POINTS.get(label, "Unknown")
            print(f"--> Selected: {label} at {coord_hint}")

            batch_count = 0
            while True:
                print(f"\n--- [Position: {label}] | Angle/Batch No: {batch_count+1} ---")
                print("   [ Action Required ]")
                print("   1. Walk to Rx -> Rotate Angle")
                print("   2. Walk AWAY")
                print("   3. Press ENTER to Record")
                print("   (Or type 'n' to finish this point and pick new P-point)")
                
                cmd = input(">> Command: ").strip().lower()
                if cmd == 'n':
                    print(f"Finished collecting for {label}.")
                    break 
                
                # 3. นับถอยหลัง (Safety Time)
                print(f"Get Ready! Walking away in {COUNTDOWN_SEC} seconds...")
                for i in range(COUNTDOWN_SEC, 0, -1):
                    print(f" {i}...", end='', flush=True)
                    time.sleep(1)
                print("\n[STARTED] Recording data... Please stand still.")

                # 4. ล้างข้อมูลเก่าทิ้ง (Flush Buffer) เพื่อเอาข้อมูลสดๆ เท่านั้น
                serial_q.clear() 
                
                # 5. ลูปเก็บข้อมูลจนครบ 400 ค่า
                collected_data = []
                while len(collected_data) < SAMPLES_PER_BATCH:
                    if serial_q:
                        raw_line = serial_q.popleft()
                        try:
                            # Parse JSON: {"rssi_1":-50, "rssi_2":-60, ...}
                            data = json.loads(raw_line)
                            
                            # ดึงค่า (ถ้าไม่มีให้ใส่ -100)
                            r1 = data.get('rssi_1', -100)
                            r2 = data.get('rssi_2', -100)
                            r3 = data.get('rssi_3', -100)
                            
                            # เก็บลง List ชั่วคราว
                            collected_data.append([r1, r2, r3, label, coord_hint])
                            
                            # แสดงผล Progress
                            if len(collected_data) % 50 == 0:
                                print(f"\r   Progress: {len(collected_data)}/{SAMPLES_PER_BATCH} | Latest: {r1}, {r2}, {r3}", end='')
                        except:
                            continue
                    
                    # เช็คว่า Serial หลุดไหม
                    if not is_running:
                        print("\n[Error] Serial disconnected!")
                        break
                
                # 6. บันทึกลงไฟล์
                if collected_data:
                    writer.writerows(collected_data)
                    f.flush() # บังคับเขียนลง Disk ทันที
                    print(f"\n[DONE] Saved {len(collected_data)} samples for {label}.")
                    batch_count += 1
                else:
                    print("\n[Warning] No data collected. Check connection.")

    except KeyboardInterrupt:
        print("\n[System] Stopping by User...")
    finally:
        is_running = False
        f.close()
        print("[System] File closed. Exiting.")

if __name__ == "__main__":
    main()