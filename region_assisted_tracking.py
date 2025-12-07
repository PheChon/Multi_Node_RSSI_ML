import serial
import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import threading
import pandas as pd
from collections import deque
from sklearn.neighbors import KNeighborsRegressor

# --- CONFIGURATION ---
SERIAL_PORT = 'COM10'
BAUD_RATE = 115200
DATASET_FILE = 'cleaned_data.csv' # ใช้ไฟล์ข้อมูลดิบ (ไม่ใช่ไฟล์ Model .pkl)

# รัศมีวงค้นหา (หน่วยเมตร)
# ถ้า Math บอกว่าอยู่ตรงนี้ AI จะค้นหาแค่ข้อมูลในระยะนี้เท่านั้น
SEARCH_RADIUS = 1.0  

# พิกัด Anchors (Math)
ANCHORS = {
    1: np.array([0.0, 0.0]),
    2: np.array([np.sqrt(2), 0.0]),
    3: np.array([np.sqrt(2)/2, np.sqrt(2)])
}

# ค่าคงที่ RSSI (Math)
RSSI_A = -48.5      
PATH_LOSS_N = 2.2   

# พิกัดจริงของจุด P1-P9 (Reference)
REF_POINTS = {
    "P1": [0.35, 0.35], "P2": [0.71, 0.35], "P3": [1.06, 0.35],
    "P4": [0.35, 0.71], "P5": [0.71, 0.71], "P6": [1.06, 0.71],
    "P7": [0.35, 1.06], "P8": [0.71, 1.06], "P9": [1.06, 1.06]
}

ROOM_SIZE = 1.414

# --- GLOBAL VARIABLES ---
current_pos_math = np.array([0.71, 0.71])
current_pos_final = np.array([0.71, 0.71])
active_labels = [] # เก็บรายชื่อจุดที่อยู่ในวงค้นหา (เช่น ['P1', 'P2'])

df_data = None # เก็บข้อมูลทั้งหมด
is_running = True

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
        return res[0]
    except:
        return None

# --- PROCESS THREAD ---
def process_data():
    global current_pos_math, current_pos_final, active_labels, is_running, df_data
    
    # 1. โหลดข้อมูลดิบ
    try:
        df_data = pd.read_csv(DATASET_FILE)
        print(f"[System] Loaded dataset: {len(df_data)} rows")
    except:
        print(f"[Error] File {DATASET_FILE} not found!")
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

            if r1 == -100 or r2 == -100 or r3 == -100 or r1 == 0: continue

            # --- STEP 1: Math Calculation (ไฟฉายส่องทาง) ---
            d1 = rssi_to_dist(r1)
            d2 = rssi_to_dist(r2)
            d3 = rssi_to_dist(r3)
            math_pos = solve_trilateration(d1, d2, d3)
            
            if math_pos is None: continue # ถ้า Math พัง ให้ข้ามไปก่อน
            
            # Clamp พิกัด Math ให้อยู่ในห้อง (กันค่าหลุดโลก)
            math_pos[0] = max(0, min(ROOM_SIZE, math_pos[0]))
            math_pos[1] = max(0, min(ROOM_SIZE, math_pos[1]))
            current_pos_math = math_pos

            # --- STEP 2: Filter Data (จำกัดวงค้นหา) ---
            # หาว่าจุด P ไหนบ้างที่อยู่ใกล้ Math ในระยะ SEARCH_RADIUS
            valid_labels = []
            for label, coord in REF_POINTS.items():
                dist = np.sqrt(np.sum((math_pos - coord)**2))
                if dist <= SEARCH_RADIUS:
                    valid_labels.append(label)
            
            active_labels = valid_labels # ส่งไปโชว์ในกราฟ

            # ถ้าไม่เจอจุดไหนเลย (Math หลุดไปไกล) -> ใช้ข้อมูลทั้งหมด (Fallback)
            if not valid_labels:
                filtered_df = df_data
            else:
                # เลือกเฉพาะแถวที่เป็นจุดใกล้เคียง
                filtered_df = df_data[df_data['Position_Label'].isin(valid_labels)]

            # --- STEP 3: AI Prediction (หาของละเอียด) ---
            if len(filtered_df) > 0:
                # เทรน AI สดๆ ด้วยข้อมูลที่คัดมาแล้ว (k-NN มันเร็ว ทำได้สบายมาก)
                X_train = filtered_df[['Tx1', 'Tx2', 'Tx3']].values
                # แปลง Label เป็นพิกัด
                Y_train = np.array([REF_POINTS[l] for l in filtered_df['Position_Label']])
                
                # ใช้ K น้อยลงหน่อยเพราะข้อมูลน้อยลง (เช่น K=3)
                knn = KNeighborsRegressor(n_neighbors=min(5, len(filtered_df)), weights='distance')
                knn.fit(X_train, Y_train)
                
                prediction = knn.predict([[r1, r2, r3]])[0]
                
                # Smoothing (Alpha Filter)
                ALPHA = 0.3
                current_pos_final = (current_pos_final * (1 - ALPHA)) + (prediction * ALPHA)
            else:
                # ถ้าไม่มีข้อมูลเลย ใช้ค่า Math ไปก่อน
                current_pos_final = math_pos

        except Exception as e:
            pass

# --- PLOTTING ---
def update_plot(frame):
    plt.cla()
    
    # วาดกรอบห้อง
    plt.plot([0, ROOM_SIZE, ROOM_SIZE, 0, 0], [0, 0, ROOM_SIZE, ROOM_SIZE, 0], 'k-', linewidth=2)
    
    # 1. วาดจุดอ้างอิง P1-P9
    for label, coord in REF_POINTS.items():
        # ถ้าจุดไหนอยู่ในวงค้นหา ให้เป็นสีเขียวเข้ม
        color = 'green' if label in active_labels else 'gray'
        alpha = 0.8 if label in active_labels else 0.2
        size = 80 if label in active_labels else 30
        
        plt.scatter(coord[0], coord[1], c=color, alpha=alpha, s=size)
        plt.text(coord[0], coord[1], label, fontsize=8, color=color, ha='center', va='bottom')

    # 2. วาดจุด Math (ไฟฉาย) + วงกลมค้นหา
    plt.scatter(current_pos_math[0], current_pos_math[1], c='red', marker='x', s=100, label='Math Guiding')
    search_circle = plt.Circle(current_pos_math, SEARCH_RADIUS, color='red', fill=False, linestyle='--', alpha=0.3)
    plt.gca().add_patch(search_circle)

    # 3. วาดจุด Final (User)
    plt.scatter(current_pos_final[0], current_pos_final[1], c='blue', s=250, edgecolors='white', linewidth=2, label='User (AI)', zorder=10)
    plt.text(current_pos_final[0], current_pos_final[1]-0.15, 
             f"({current_pos_final[0]:.2f}, {current_pos_final[1]:.2f})", 
             ha='center', color='blue', fontweight='bold')

    plt.title(f"Region-Assisted k-NN (Radius: {SEARCH_RADIUS}m)\nActive Points: {', '.join(active_labels)}")
    plt.legend(loc='upper right', fontsize='small')
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
    ani = animation.FuncAnimation(fig, update_plot, interval=50)
    plt.show()
    
    is_running = False