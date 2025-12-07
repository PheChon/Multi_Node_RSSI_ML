import serial
import json
import matplotlib.pyplot as plt
import numpy as np
import time
from collections import deque

# --- CONFIGURATION ---
SERIAL_PORT = 'COM10'
BAUD_RATE = 115200

# พิกัด Anchors (ห้อง Root 2)
ANCHORS = {
    1: np.array([0.0, 0.0]),
    2: np.array([np.sqrt(2), 0.0]),
    3: np.array([np.sqrt(2)/2, np.sqrt(2)])
}

PATH_LOSS_N = 2.2   # ค่า n

# --- CALCULATION FUNCTIONS ---
def get_distance(rssi, A, n):
    if rssi >= 0: rssi = -1.0
    # สูตร Path Loss: RSSI = A - 10*n*log10(d)
    # ดังนั้น: d = 10 ^ ((A - RSSI) / (10 * n))
    try:
        val = (A - rssi) / (10 * n)
        dist = 10 ** val
        return dist
    except:
        return 0.0

def trilaterate(anchors, d1, d2, d3):
    try:
        xa, ya = anchors[1]; xb, yb = anchors[2]; xc, yc = anchors[3]
        
        # ใช้สมการ Linearization แบบ Tx1 เป็นจุดอ้างอิง (0,0)
        # 2x(xi) + 2y(yi) = di^2 - d1^2 + xi^2 + yi^2
        # Tx2 (เทียบ Tx1)
        A1 = 2 * xb; B1 = 2 * yb
        C1 = d1**2 - d2**2 + xb**2 + yb**2
        
        # Tx3 (เทียบ Tx1)
        A2 = 2 * xc; B2 = 2 * yc
        C2 = d1**2 - d3**2 + xc**2 + yc**2

        Matrix_A = np.array([[A1, B1], [A2, B2]])
        Vector_B = np.array([C1, C2])
        
        result = np.linalg.solve(Matrix_A, Vector_B)
        return result
    except Exception:
        return None

# --- MAIN TUNING SCRIPT ---
def run_tuning():
    print("\n" + "="*50)
    print("   AUTO-TUNING RSSI PARAMETER (Fixed Version)")
    print("="*50)

    try:
        real_x = float(input(">> พิกัดจริง X (m): "))
        real_y = float(input(">> พิกัดจริง Y (m): "))
        capture_time = int(input(">> เวลาเก็บข้อมูล (วินาที) [แนะนำ 30]: "))
        
        print("\n--- ตั้งค่าช่วงการจูน A (แนะนำ: -70 ถึง -30) ---")
        start_a = float(input("   A เริ่มต้น (เช่น -70): "))
        end_a = float(input("   A สิ้นสุด (เช่น -30): "))
        step_a = float(input("   เพิ่มทีละ (เช่น 1.0): "))
        
    except ValueError:
        print("Input ผิดพลาด")
        return

    # 2. เก็บข้อมูลสด
    raw_data = [] 
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print(f"\n[System] เริ่มเก็บข้อมูลดิบ {capture_time} วินาที...")
        
        start_t = time.time()
        while time.time() - start_t < capture_time:
            if ser.in_waiting:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                try:
                    data = json.loads(line)
                    r1 = data.get('rssi_1', -100)
                    r2 = data.get('rssi_2', -100)
                    r3 = data.get('rssi_3', -100)
                    
                    if r1 != -100 and r2 != -100 and r3 != -100 and r1 != 0:
                        raw_data.append([r1, r2, r3])
                        print(f"\rCollected: {len(raw_data)} samples | Last: {r1},{r2},{r3}", end='')
                except: pass
            
        print("\n[System] เก็บข้อมูลเสร็จสิ้น!")
        ser.close()
        
    except Exception as e:
        print(f"Error: {e}")
        return

    if not raw_data:
        print("ไม่ได้รับข้อมูลเลย จบการทำงาน")
        return

    # 3. เริ่มคำนวณ
    print(f"\n[System] กำลังจำลองการคำนวณ...")
    
    # สร้างช่วง A ให้ถูกต้อง (รองรับกรณีพิมพ์สลับมากน้อย)
    if start_a > end_a: start_a, end_a = end_a, start_a
    a_values = np.arange(start_a, end_a + step_a, step_a)
    
    avg_errors_cm = []
    valid_a = []
    
    best_a = 0
    min_avg_error = float('inf')

    for test_a in a_values:
        errors = []
        valid_count = 0
        
        for sample in raw_data:
            d1 = get_distance(sample[0], test_a, PATH_LOSS_N)
            d2 = get_distance(sample[1], test_a, PATH_LOSS_N)
            d3 = get_distance(sample[2], test_a, PATH_LOSS_N)
            
            pos = trilaterate(ANCHORS, d1, d2, d3)
            
            if pos is not None:
                err = np.sqrt((pos[0] - real_x)**2 + (pos[1] - real_y)**2)
                # กรองค่า Error ที่เป็นไปไม่ได้ (เช่น > 10 เมตร) ออกไปก่อน
                if err < 10.0: 
                    errors.append(err)
                    valid_count += 1
        
        if len(errors) > 0:
            mean_err_m = np.mean(errors)
            mean_err_cm = mean_err_m * 100
            
            avg_errors_cm.append(mean_err_cm)
            valid_a.append(test_a)
            
            if mean_err_cm < min_avg_error:
                min_avg_error = mean_err_cm
                best_a = test_a
            
            # Debug: แสดงผลบางค่าเพื่อดูว่าโค้ดวิ่งไหม
            # print(f"A={test_a:.1f} -> Err={mean_err_cm:.2f} cm (Valid: {valid_count})")

    # 4. แสดงผลลัพธ์
    if min_avg_error == float('inf'):
        print("\n[Error] คำนวณไม่ได้เลย ลองปรับช่วง A ใหม่ หรือเช็คพิกัดจริง")
        return

    print("\n" + "="*50)
    print(f"   TUNING RESULTS")
    print("="*50)
    print(f"🏆 Best A value:  {best_a}")
    print(f"📉 Minimum Error: {min_avg_error:.2f} cm")
    print("="*50)

    # 5. วาดกราฟ
    plt.figure(figsize=(10, 6))
    plt.plot(valid_a, avg_errors_cm, marker='o', linestyle='-', color='b', markersize=4)
    plt.plot(best_a, min_avg_error, marker='*', color='r', markersize=15, label=f'Best A={best_a}')
    
    plt.title(f"Parameter Tuning: Error vs RSSI_A\n(Real Pos: {real_x},{real_y})")
    plt.xlabel("Parameter A (Reference RSSI)")
    plt.ylabel("Average Position Error (cm)")
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    run_tuning()