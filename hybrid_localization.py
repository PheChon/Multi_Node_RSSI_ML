import serial
import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import threading
import joblib
import time
from collections import deque

# --- CONFIGURATION ---
SERIAL_PORT = 'COM10'
BAUD_RATE = 115200

# ไฟล์โมเดล AI
MODEL_FILE = 'best_knn_model_k7.pkl' 

# พิกัด Anchors (Math)
ANCHORS = {
    1: np.array([0.0, 0.0]),
    2: np.array([np.sqrt(2), 0.0]),
    3: np.array([np.sqrt(2)/2, np.sqrt(2)])
}

# ค่าคงที่ RSSI (Math)
RSSI_A = -48.1      
PATH_LOSS_N = 2.2   

# น้ำหนัก (Weight) - ปรับเล่นได้ตรงนี้
WEIGHT_KNN = 0.7  # เชื่อ AI 70%
WEIGHT_MATH = 0.3 # เชื่อ Math 30%

# ขนาดห้อง
ROOM_SIZE = 1.414

# --- GLOBAL VARIABLES ---
current_pos_math = np.array([0.71, 0.71])
current_pos_knn = np.array([0.71, 0.71])
current_pos_hybrid = np.array([0.71, 0.71])
real_pos = np.array([0.71, 0.71])

rssi_buffers = {1: deque(maxlen=5), 2: deque(maxlen=5), 3: deque(maxlen=5)}
is_running = True
model = None

# --- CALCULATION FUNCTIONS ---
def rssi_to_dist(rssi):
    if rssi >= 0: rssi = -1.0
    return 10 ** ((RSSI_A - rssi) / (10 * PATH_LOSS_N))

def solve_trilateration(d1, d2, d3):
    try:
        # ใช้สูตร Linearization เดิม
        xa, ya = ANCHORS[1]; xb, yb = ANCHORS[2]; xc, yc = ANCHORS[3]
        A = np.array([[2*(xa-xb), 2*(ya-yb)], [2*(xa-xc), 2*(ya-yc)]])
        B = np.array([
            d2**2 - d1**2 - xb**2 + xa**2 - yb**2 + ya**2,
            d3**2 - d1**2 - xc**2 + xa**2 - yc**2 + ya**2
        ])
        # ใช้ Least Squares แทน solve เพื่อความทนทานกว่า
        res = np.linalg.lstsq(A, B, rcond=None)
        return res[0] # [x, y]
    except:
        return None

# --- PROCESS THREAD ---
def process_data():
    global current_pos_math, current_pos_knn, current_pos_hybrid, is_running, model
    
    # 1. โหลดโมเดล
    try:
        model = joblib.load(MODEL_FILE)
        print(f"[System] Loaded {MODEL_FILE}")
    except:
        print("[Error] Model not found")
        is_running = False
        return

    # 2. เชื่อมต่อ Serial
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print(f"[System] Connected to {SERIAL_PORT}")
    except:
        print("[Error] Serial Failed")
        is_running = False
        return

    while is_running:
        try:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if not line.startswith('{'): continue
            
            data = json.loads(line)
            r1 = data.get('rssi_1', -100)
            r2 = data.get('rssi_2', -100)
            r3 = data.get('rssi_3', -100)

            # กรองค่า
            if r1 == -100 or r2 == -100 or r3 == -100 or r1 == 0: continue

            # --- 1. Math Calculation ---
            d1 = rssi_to_dist(r1)
            d2 = rssi_to_dist(r2)
            d3 = rssi_to_dist(r3)
            math_res = solve_trilateration(d1, d2, d3)
            
            if math_res is not None:
                current_pos_math = math_res

            # --- 2. AI Prediction ---
            knn_res = model.predict([[r1, r2, r3]])[0]
            current_pos_knn = knn_res

            # --- 3. Hybrid Fusion ---
            # สูตรผสม: (KNN * 0.7) + (Math * 0.3)
            hybrid_x = (current_pos_knn[0] * WEIGHT_KNN) + (current_pos_math[0] * WEIGHT_MATH)
            hybrid_y = (current_pos_knn[1] * WEIGHT_KNN) + (current_pos_math[1] * WEIGHT_MATH)
            current_pos_hybrid = np.array([hybrid_x, hybrid_y])

            # Debug ดูผลเทียบกัน
            # print(f"Math:{current_pos_math} | AI:{current_pos_knn} -> Hyb:{current_pos_hybrid}")

        except:
            pass

# --- PLOTTING ---
def update_plot(frame):
    plt.cla()
    
    # วาดกรอบห้อง
    plt.plot([0, ROOM_SIZE, ROOM_SIZE, 0, 0], [0, 0, ROOM_SIZE, ROOM_SIZE, 0], 'k-', linewidth=2)
    
    # 1. วาดจุดจริง (Ground Truth)
    plt.scatter(real_pos[0], real_pos[1], c='green', s=300, marker='*', label='Real Position')
    plt.text(real_pos[0], real_pos[1]+0.1, "REAL", color='green', ha='center', fontweight='bold')

    # 2. วาดจุด Math (สีแดงจางๆ)
    plt.scatter(current_pos_math[0], current_pos_math[1], c='red', s=100, alpha=0.3, label='Math Only')
    
    # 3. วาดจุด AI (สีน้ำเงินจางๆ)
    plt.scatter(current_pos_knn[0], current_pos_knn[1], c='blue', s=100, alpha=0.3, label='AI Only')

    # 4. วาดจุด Hybrid (สีม่วง - พระเอกของเรา)
    plt.scatter(current_pos_hybrid[0], current_pos_hybrid[1], c='purple', s=200, label='Hybrid (Fused)')
    
    # คำนวณ Error สดๆ ของ Hybrid
    err = np.sqrt(np.sum((current_pos_hybrid - real_pos)**2))
    
    plt.title(f"Hybrid Localization (AI {WEIGHT_KNN*100}% + Math {WEIGHT_MATH*100}%)\nCurrent Error: {err*100:.1f} cm")
    plt.legend(loc='upper right')
    plt.xlim(-0.5, 2.0)
    plt.ylim(-0.5, 2.0)
    plt.grid(True, linestyle=':')
    plt.gca().set_aspect('equal')

# --- MAIN ---
if __name__ == "__main__":
    # รับ Input จุดจริงเพื่อเปรียบเทียบ
    try:
        rx = float(input("Enter Real X: "))
        ry = float(input("Enter Real Y: "))
        real_pos = np.array([rx, ry])
    except:
        print("Using default center (0.71, 0.71)")

    t = threading.Thread(target=process_data)
    t.daemon = True
    t.start()

    fig = plt.figure(figsize=(8, 8))
    ani = animation.FuncAnimation(fig, update_plot, interval=50)
    plt.show()