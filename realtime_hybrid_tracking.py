import serial
import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import threading
import joblib
import time
from collections import deque

# --- CONFIGURATION (ตั้งค่าระบบ) ---
SERIAL_PORT = 'COM10'   # <--- เช็ค Port
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
RSSI_A = -48.5      
PATH_LOSS_N = 2.2   

# น้ำหนัก (Weight) - ปรับความเชื่อถือตรงนี้
WEIGHT_KNN = 0.7  # เชื่อ AI 70%
WEIGHT_MATH = 0.3 # เชื่อ Math 30%

# ขนาดห้อง (สำหรับวาดกราฟ)
ROOM_SIZE = 1.414

# พิกัดอ้างอิง P1-P9 (สำหรับวาดจุดอ้างอิงจางๆ)
REF_POINTS = {
    "P1": [0.35, 0.35], "P2": [0.71, 0.35], "P3": [1.06, 0.35],
    "P4": [0.35, 0.71], "P5": [0.71, 0.71], "P6": [1.06, 0.71],
    "P7": [0.35, 1.06], "P8": [0.71, 1.06], "P9": [1.06, 1.06]
}

# --- GLOBAL VARIABLES ---
current_pos_math = np.array([0.71, 0.71])
current_pos_knn = np.array([0.71, 0.71])
current_pos_hybrid = np.array([0.71, 0.71]) # นี่คือค่าที่เราจะเอาไปโชว์

rssi_buffers = {1: deque(maxlen=5), 2: deque(maxlen=5), 3: deque(maxlen=5)}
is_running = True
model = None

# --- CALCULATION FUNCTIONS ---
def rssi_to_dist(rssi):
    if rssi >= 0: rssi = -1.0
    return 10 ** ((RSSI_A - rssi) / (10 * PATH_LOSS_N))

def solve_trilateration(d1, d2, d3):
    try:
        xa, ya = ANCHORS[1]; xb, yb = ANCHORS[2]; xc, yc = ANCHORS[3]
        A = np.array([[2*(xa-xb), 2*(ya-yb)], [2*(xa-xc), 2*(ya-yc)]])
        B = np.array([
            d2**2 - d1**2 - xb**2 + xa**2 - yb**2 + ya**2,
            d3**2 - d1**2 - xc**2 + xa**2 - yc**2 + ya**2
        ])
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
        print("[System] Real-time tracking started... (Press Ctrl+C to stop)")
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

            # กรองค่าขยะ
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
            # สูตรผสม: (KNN * W_KNN) + (Math * W_MATH)
            hybrid_x = (current_pos_knn[0] * WEIGHT_KNN) + (current_pos_math[0] * WEIGHT_MATH)
            hybrid_y = (current_pos_knn[1] * WEIGHT_KNN) + (current_pos_math[1] * WEIGHT_MATH)
            
            # อัปเดตตำแหน่ง (ไม่มี Smoothing เพิ่มเติม เพราะการผสมกันคือ Smoothing ชั้นดีแล้ว)
            current_pos_hybrid = np.array([hybrid_x, hybrid_y])

            # Debug (ถ้าอยากดูค่าดิบให้เอาคอมเมนท์ออก)
            # print(f"Hyb: ({hybrid_x:.2f}, {hybrid_y:.2f})")

        except:
            pass

# --- PLOTTING ---
def update_plot(frame):
    plt.cla()
    
    # วาดกรอบห้อง
    plt.plot([0, ROOM_SIZE, ROOM_SIZE, 0, 0], [0, 0, ROOM_SIZE, ROOM_SIZE, 0], 'k-', linewidth=2)
    
    # วาดจุดอ้างอิง P1-P9 (สีเทาจางๆ) เพื่อให้รู้ว่าเดินถึงไหนแล้ว
    for label, coord in REF_POINTS.items():
        plt.scatter(coord[0], coord[1], c='gray', alpha=0.2, s=30)
        plt.text(coord[0], coord[1], label, fontsize=8, color='gray', alpha=0.5, ha='center', va='bottom')

    # วาดตำแหน่ง Hybrid (User) - จุดเดียวเน้นๆ
    plt.scatter(current_pos_hybrid[0], current_pos_hybrid[1], 
                c='blue', s=300, edgecolors='white', linewidth=2, label='User Position', zorder=10)
    
    # แสดงพิกัดตัวเลขข้างๆ จุด
    plt.text(current_pos_hybrid[0], current_pos_hybrid[1]-0.15, 
             f"X: {current_pos_hybrid[0]:.2f}\nY: {current_pos_hybrid[1]:.2f}", 
             ha='center', color='blue', fontweight='bold')

    plt.title(f"Real-time Tracking (Hybrid AI + Math)")
    plt.xlim(-0.2, ROOM_SIZE + 0.2)
    plt.ylim(-0.2, ROOM_SIZE + 0.2)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.gca().set_aspect('equal', adjustable='box')

# --- MAIN ---
if __name__ == "__main__":
    t = threading.Thread(target=process_data)
    t.daemon = True
    t.start()

    fig = plt.figure(figsize=(8, 8))
    ani = animation.FuncAnimation(fig, update_plot, interval=50) # Refresh 20 FPS
    plt.show()
    
    is_running = False