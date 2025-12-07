import serial
import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import threading
import joblib  # สำหรับโหลดโมเดล
import time

# --- CONFIGURATION (ตั้งค่าระบบ) ---
SERIAL_PORT = 'COM10'   # <--- เช็ค Port ให้ชัวร์
BAUD_RATE = 115200

# *** ใส่ชื่อไฟล์โมเดลที่ดีที่สุดที่คุณเลือก ***
MODEL_FILE = 'best_knn_model_k7.pkl'  # <--- แก้ตรงนี้ครับ! (เช่น k3 หรือ k5)

# พิกัดอ้างอิง P1-P9 (สำหรับวาดบนกราฟ)
REF_POINTS = {
    "P1": [0.35, 0.35], "P2": [0.71, 0.35], "P3": [1.06, 0.35],
    "P4": [0.35, 0.71], "P5": [0.71, 0.71], "P6": [1.06, 0.71],
    "P7": [0.35, 1.06], "P8": [0.71, 1.06], "P9": [1.06, 1.06]
}

# ค่าความหน่วง (Smoothing Factor)
# 0.1 = นิ่งมากแต่ช้า (Delay)
# 0.9 = ไวมากแต่สั่น (Jitter)
# แนะนำ: 0.2 - 0.4
ALPHA = 0.1 

# --- VARIABLES ---
current_pos = np.array([0.71, 0.71])  # เริ่มต้นที่กลางห้อง
target_pos = np.array([0.71, 0.71])   # เป้าหมายที่ AI บอก
is_running = True
model = None

# --- AI & SERIAL THREAD ---
def process_data():
    global target_pos, is_running, model
    
    # 1. โหลดโมเดล
    print(f"[System] Loading model: {MODEL_FILE}...")
    try:
        model = joblib.load(MODEL_FILE)
        print("[System] Model loaded successfully!")
    except Exception as e:
        print(f"[Error] Load model failed: {e}")
        is_running = False
        return

    # 2. เชื่อมต่อ Serial
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print(f"[System] Connected to {SERIAL_PORT}")
    except Exception as e:
        print(f"[Error] Serial failed: {e}")
        is_running = False
        return

    while is_running:
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

                # 3. ให้ AI ทำนาย (Predict)
                # input ต้องเป็น 2D array: [[r1, r2, r3]]
                input_data = [[r1, r2, r3]]
                prediction = model.predict(input_data)
                
                # อัปเดตเป้าหมายใหม่
                target_pos = prediction[0]
                
                # Debug ข้อมูลดิบ
                # print(f"RSSI: {r1},{r2},{r3} -> Pred: {target_pos}")

        except Exception:
            pass

# --- VISUALIZATION ---
def update_plot(frame):
    global current_pos
    
    plt.cla()
    
    # 1. คำนวณ Smoothing (EMA Filter)
    # ขยับจุดปัจจุบันเข้าหาจุดเป้าหมายทีละนิด (ALPHA)
    current_pos = (current_pos * (1 - ALPHA)) + (target_pos * ALPHA)

    # 2. วาดกรอบห้อง
    ROOM_SIZE = 1.414
    plt.plot([0, ROOM_SIZE, ROOM_SIZE, 0, 0], [0, 0, ROOM_SIZE, ROOM_SIZE, 0], 'k-', linewidth=2)
    
    # 3. วาดจุดอ้างอิง P1-P9
    for label, coord in REF_POINTS.items():
        plt.scatter(coord[0], coord[1], c='gray', alpha=0.3, s=50)
        plt.text(coord[0], coord[1], label, fontsize=8, color='gray', ha='center', va='bottom')

    # 4. วาดตำแหน่งผู้ใช้ (จุดสีน้ำเงิน)
    plt.scatter(current_pos[0], current_pos[1], c='blue', s=300, edgecolors='white', linewidth=2, label='User', zorder=10)
    
    # แสดงพิกัดตัวเลข
    plt.text(current_pos[0], current_pos[1]-0.15, 
             f"X: {current_pos[0]:.2f} m\nY: {current_pos[1]:.2f} m", 
             ha='center', color='blue', fontweight='bold')

    plt.xlim(-0.2, ROOM_SIZE + 0.2)
    plt.ylim(-0.2, ROOM_SIZE + 0.2)
    plt.title(f"Real-time Tracking (Model: {MODEL_FILE})")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.gca().set_aspect('equal', adjustable='box')

# --- MAIN ---
if __name__ == "__main__":
    # เริ่ม Thread คำนวณ
    t = threading.Thread(target=process_data)
    t.daemon = True
    t.start()

    # เริ่มกราฟ
    fig = plt.figure(figsize=(8, 8))
    ani = animation.FuncAnimation(fig, update_plot, interval=50) # 50ms = 20 FPS
    plt.show()
    
    is_running = False