import serial
import json
import matplotlib.pyplot as plt
import numpy as np
import time
import joblib
from sklearn.neighbors import KNeighborsRegressor

# --- CONFIGURATION (ตั้งค่าระบบ) ---
SERIAL_PORT = 'COM10'   # <--- เช็ค Port
BAUD_RATE = 115200

# *** ชื่อไฟล์โมเดล *** (ต้องมีไฟล์นี้อยู่ในโฟลเดอร์เดียวกัน)
MODEL_FILE = 'best_knn_model_k7.pkl' 

# ขนาดห้อง (Diagonal) เพื่อใช้คำนวณ % Error
# ห้องกว้าง sqrt(2) สูง sqrt(2) -> เส้นทแยงมุม = 2.0 เมตร
ROOM_DIAGONAL_CM = 2.0 * 100 

# --- MAIN TEST FUNCTION ---
def run_stability_test():
    print("\n" + "="*50)
    print(f"   k-NN STABILITY TESTER (Model: {MODEL_FILE})")
    print("="*50)

    # 1. โหลดโมเดล AI
    print(f"[System] Loading model...")
    try:
        model = joblib.load(MODEL_FILE)
        print("[System] Model loaded successfully!")
    except Exception as e:
        print(f"[Error] Load model failed: {e}")
        print("กรุณาเช็คว่ามีไฟล์ .pkl อยู่ในโฟลเดอร์เดียวกับโค้ดนี้หรือไม่")
        return

    # 2. รับ Input พิกัดจริง
    try:
        real_x = float(input(">> ใส่พิกัดจริง X (เมตร): "))
        real_y = float(input(">> ใส่พิกัดจริง Y (เมตร): "))
        duration = int(input(">> ใส่เวลาที่ต้องการทดสอบ (วินาที): "))
    except ValueError:
        print("Input ผิดพลาด! กรุณาใส่ตัวเลข")
        return

    # 3. เชื่อมต่อ Serial
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
    
    print("-" * 60)
    print(f"{'Time (s)':<10} | {'Input RSSI':<20} | {'Pred (X, Y)':<20} | {'Error'}")
    print("-" * 60)

    # 4. Loop เก็บข้อมูล
    while (time.time() - start_time) < duration:
        try:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if line.startswith('{') and '}' in line:
                data = json.loads(line)
                
                # รับค่า RSSI
                r1 = data.get('rssi_1', -100)
                r2 = data.get('rssi_2', -100)
                r3 = data.get('rssi_3', -100)

                # กรองค่าขยะ (-100 หรือ 0)
                if r1 == -100 or r2 == -100 or r3 == -100 or r1 == 0:
                    continue

                # --- AI PREDICTION ---
                # ให้ AI ทำนายพิกัด (แบบดิบๆ ไม่ผ่าน Smoothing เพื่อวัด Performance จริง)
                input_data = [[r1, r2, r3]]
                prediction = model.predict(input_data)
                pred_x, pred_y = prediction[0]

                # --- ERROR CALCULATION ---
                err_m = np.sqrt((pred_x - real_x)**2 + (pred_y - real_y)**2)
                err_cm = err_m * 100
                
                # บันทึกผล
                current_t = time.time() - start_time
                timestamps.append(current_t)
                errors_cm.append(err_cm)
                
                print(f"{current_t:<10.1f} | {r1},{r2},{r3:<12} | ({pred_x:.2f}, {pred_y:.2f})      | {err_cm:.2f} cm")

        except (json.JSONDecodeError, ValueError):
            pass
        except KeyboardInterrupt:
            print("\n[Stop] หยุดก่อนกำหนด")
            break

    ser.close()

    # 5. สรุปผล
    if len(errors_cm) == 0:
        print("\n[Error] ไม่ได้รับข้อมูลเลย กรุณาเช็ค Tx/Rx")
        return

    avg_error_cm = np.mean(errors_cm)
    max_error_cm = np.max(errors_cm)
    min_error_cm = np.min(errors_cm)
    avg_error_percent = (avg_error_cm / ROOM_DIAGONAL_CM) * 100

    print("\n" + "="*40)
    print("   TEST RESULTS SUMMARY (k-NN)")
    print("="*40)
    print(f"ระยะเวลาทดสอบ:      {timestamps[-1]:.1f} วินาที")
    print(f"จำนวนข้อมูลที่ได้:     {len(errors_cm)} samples")
    print("-" * 40)
    print(f"1. ค่าเฉลี่ย Error (cm): {avg_error_cm:.2f} cm")
    print(f"2. ค่าเฉลี่ย Error (%):  {avg_error_percent:.2f} %")
    print(f"   (Min: {min_error_cm:.1f} cm | Max: {max_error_cm:.1f} cm)")
    print("="*40)

    # 6. วาดกราฟ
    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, errors_cm, label='Error (cm)', color='purple', linewidth=1.5)
    plt.axhline(y=avg_error_cm, color='red', linestyle='--', label=f'Average: {avg_error_cm:.2f} cm')
    
    # Title แบบละเอียด
    title_text = (
        f"k-NN Stability Test at Real Pos ({real_x}, {real_y})\n"
        f"Avg Error: {avg_error_cm:.2f} cm ({avg_error_percent:.2f}%)\n"
        f"[Min: {min_error_cm:.2f} cm | Max: {max_error_cm:.2f} cm]"
    )
    plt.title(title_text, fontsize=12, fontweight='bold')
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Error Distance (cm)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_stability_test()