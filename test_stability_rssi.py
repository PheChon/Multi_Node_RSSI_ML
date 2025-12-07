import serial
import json
import matplotlib.pyplot as plt
import numpy as np
import time
from collections import deque

# --- CONFIGURATION (ตั้งค่าระบบ) ---
SERIAL_PORT = 'COM10'   # <--- เช็ค Port
BAUD_RATE = 115200

# พิกัด Anchors (3 ตัว) - สำหรับห้องขนาด Root 2 x Root 2
ANCHORS = {
    1: np.array([0.0, 0.0]),                # Tx1: มุมซ้ายล่าง
    2: np.array([np.sqrt(2), 0.0]),         # Tx2: มุมขวาล่าง (~1.414, 0)
    3: np.array([np.sqrt(2)/2, np.sqrt(2)]) # Tx3: กึ่งกลางด้านบน (~0.707, 1.414)
}

# ขนาดห้อง (Diagonal) เพื่อใช้คำนวณ % Error
# เส้นทแยงมุม = sqrt(root2^2 + root2^2) = sqrt(2+2) = 2.0 เมตร
ROOM_DIAGONAL_CM = 2.0 * 100 

# ค่าคงที่ (RSSI Math)
RSSI_A = -48.1      
PATH_LOSS_N = 2.2   

# การกรองข้อมูล
FILTER_SIZE = 5
rssi_buffers = {1: deque(maxlen=FILTER_SIZE), 2: deque(maxlen=FILTER_SIZE), 3: deque(maxlen=FILTER_SIZE)}

# --- CALCULATION FUNCTIONS ---
def rssi_to_distance(rssi):
    if rssi >= 0: rssi = -1.0 
    return 10 ** ((RSSI_A - rssi) / (10 * PATH_LOSS_N))

def trilaterate_linear(anchors, distances):
    try:
        xa, ya = anchors[1]; xb, yb = anchors[2]; xc, yc = anchors[3]
        da = distances[1]; db = distances[2]; dc = distances[3]

        A1 = 2 * (xa - xb); B1 = 2 * (ya - yb)
        C1 = (db**2 - da**2) - (xb**2 - xa**2) - (yb**2 - ya**2)
        A2 = 2 * (xc - xb); B2 = 2 * (yc - yb)
        C2 = (db**2 - dc**2) - (xb**2 - xc**2) - (yb**2 - yc**2)

        Matrix_A = np.array([[A1, B1], [A2, B2]])
        Vector_B = np.array([C1, C2])
        return np.linalg.solve(Matrix_A, Vector_B)
    except Exception:
        return None

# --- MAIN TEST FUNCTION ---
def run_stability_test():
    print("\n" + "="*40)
    print("   RSSI STABILITY TESTER (Full Info Graph)")
    print("="*40)
    try:
        real_x = float(input(">> ใส่พิกัดจริง X (เมตร): "))
        real_y = float(input(">> ใส่พิกัดจริง Y (เมตร): "))
        duration = int(input(">> ใส่เวลาที่ต้องการทดสอบ (วินาที): "))
    except ValueError:
        print("Input ผิดพลาด! กรุณาใส่ตัวเลข")
        return

    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print(f"\n[System] Connected to {SERIAL_PORT}. Starting test in 3 seconds...")
        time.sleep(1); print("3..."); time.sleep(1); print("2..."); time.sleep(1); print("1... GO!")
    except Exception as e:
        print(f"[Error] Serial connection failed: {e}")
        return

    start_time = time.time()
    timestamps = []
    errors_cm = []
    
    print("-" * 50)
    print(f"{'Time (s)':<10} | {'Est (X, Y)':<20} | {'Error (cm)':<10}")
    print("-" * 50)

    while (time.time() - start_time) < duration:
        try:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if not line: continue
            
            data = json.loads(line)
            current_distances = {}
            valid_read = True
            
            for nid in [1, 2, 3]:
                key = f"rssi_{nid}"
                if key in data:
                    rssi_buffers[nid].append(data[key])
                    if len(rssi_buffers[nid]) > 0:
                        avg_rssi = sum(rssi_buffers[nid]) / len(rssi_buffers[nid])
                        current_distances[nid] = rssi_to_distance(avg_rssi)
                    else:
                        valid_read = False
                else:
                    valid_read = False

            if valid_read and len(current_distances) == 3:
                pos = trilaterate_linear(ANCHORS, current_distances)
                
                if pos is not None:
                    err_m = np.sqrt((pos[0] - real_x)**2 + (pos[1] - real_y)**2)
                    err_cm = err_m * 100
                    
                    current_t = time.time() - start_time
                    timestamps.append(current_t)
                    errors_cm.append(err_cm)
                    
                    print(f"{current_t:<10.1f} | ({pos[0]:.2f}, {pos[1]:.2f})      | {err_cm:.2f} cm")

        except (json.JSONDecodeError, ValueError):
            pass
        except KeyboardInterrupt:
            print("\n[Stop] หยุดก่อนกำหนด")
            break

    ser.close()

    if len(errors_cm) == 0:
        print("\n[Error] ไม่ได้รับข้อมูลเลย กรุณาเช็ค Tx/Rx")
        return

    # คำนวณสถิติ
    avg_error_cm = np.mean(errors_cm)
    max_error_cm = np.max(errors_cm)
    min_error_cm = np.min(errors_cm)
    avg_error_percent = (avg_error_cm / ROOM_DIAGONAL_CM) * 100

    print("\n" + "="*40)
    print("   TEST RESULTS SUMMARY")
    print("="*40)
    print(f"1. ค่าเฉลี่ย Error: {avg_error_cm:.2f} cm ({avg_error_percent:.2f}%)")
    print(f"2. Min Error:      {min_error_cm:.2f} cm")
    print(f"3. Max Error:      {max_error_cm:.2f} cm")
    print("="*40)

    # --- PLOTTING ---
    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, errors_cm, label='Error (cm)', color='blue', linewidth=1.5)
    plt.axhline(y=avg_error_cm, color='red', linestyle='--', label='Average Error')
    
    # [ส่วนที่เพิ่มใหม่] สร้าง Title แบบรวมข้อมูลครบถ้วน
    title_text = (
        f"Stability Test at Real Pos ({real_x}, {real_y})\n"
        f"Avg Error: {avg_error_cm:.2f} cm ({avg_error_percent:.2f}%)\n"
        f"[Min: {min_error_cm:.2f} cm | Max: {max_error_cm:.2f} cm]"
    )
    plt.title(title_text, fontsize=12, fontweight='bold')
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Error Distance (cm)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    
    # ปรับ Layout ไม่ให้ข้อความตกขอบ
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_stability_test()