import serial
import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import threading
from collections import deque

# --- CONFIGURATION (ตั้งค่าระบบ) ---
SERIAL_PORT = 'COM10'  # <--- Port ตามที่คุณระบุ
BAUD_RATE = 115200

# พิกัด Anchors (อัปเดตตามค่า Square Root ที่ขอ)
# ใช้ np.sqrt(3) เพื่อความแม่นยำทางคณิตศาสตร์
ANCHORS = {
    1: np.array([0.0, 0.0]),               # Tx1: (0, 0)
    2: np.array([np.sqrt(3), 0.0]),        # Tx2: (√3, 0)
    3: np.array([np.sqrt(3)/2, np.sqrt(3)]) # Tx3: (√3/2, √3)
}

# ค่าคงที่จากเอกสาร
RSSI_A = -45.0      # ค่า A (RSSI ที่ 1 เมตร) ตามที่ระบุใหม่
PATH_LOSS_N = 2.2   # ค่า n

# การกรองข้อมูล
FILTER_SIZE = 5

# --- GLOBAL VARIABLES ---
rssi_buffers = {1: deque(maxlen=FILTER_SIZE), 2: deque(maxlen=FILTER_SIZE), 3: deque(maxlen=FILTER_SIZE)}
current_distances = {1: 0.0, 2: 0.0, 3: 0.0}
current_pos = None

# --- CALCULATION FUNCTIONS ---

def rssi_to_distance(rssi):
    """ สูตรคำนวณระยะทางจาก RSSI """
    # ป้องกันค่าบวก (Error ทางฟิสิกส์) ให้ปัดเป็นระยะใกล้สุดๆ แทน
    if rssi >= 0: rssi = -1.0 
    
    exponent = (RSSI_A - rssi) / (10 * PATH_LOSS_N)
    dist = 10 ** exponent
    return dist

def trilaterate_linear(anchors, distances):
    """ คำนวณหาตำแหน่ง (x, y) ด้วย Linear Algebra """
    try:
        xa, ya = anchors[1]
        xb, yb = anchors[2]
        xc, yc = anchors[3]
        
        da = distances[1]
        db = distances[2]
        dc = distances[3]

        # สมการที่ 1 (Sphere B - Sphere A)
        A1 = 2 * (xa - xb)
        B1 = 2 * (ya - yb)
        C1 = (db**2 - da**2) - (xb**2 - xa**2) - (yb**2 - ya**2)

        # สมการที่ 2 (Sphere B - Sphere C)
        A2 = 2 * (xc - xb)
        B2 = 2 * (yc - yb)
        C2 = (db**2 - dc**2) - (xb**2 - xc**2) - (yb**2 - yc**2)

        # สร้าง Matrix แก้สมการ
        Matrix_A = np.array([[A1, B1], [A2, B2]])
        Vector_B = np.array([C1, C2])

        result = np.linalg.solve(Matrix_A, Vector_B)
        return result

    except np.linalg.LinAlgError:
        return None

# --- SERIAL THREAD ---
def read_serial():
    global current_pos
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print(f"Connected to {SERIAL_PORT}")
        print(f"Anchors Configured:\nTx1: {ANCHORS[1]}\nTx2: {ANCHORS[2]}\nTx3: {ANCHORS[3]}")
        
        while True:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if not line: continue
            
            try:
                data = json.loads(line)
                
                # Update Buffer & Calculate
                for nid in [1, 2, 3]:
                    key = f"rssi_{nid}"
                    if key in data:
                        rssi_buffers[nid].append(data[key])
                        avg_rssi = sum(rssi_buffers[nid]) / len(rssi_buffers[nid])
                        current_distances[nid] = rssi_to_distance(avg_rssi)

                # คำนวณตำแหน่งเมื่อข้อมูลพร้อม
                if all(len(b) > 0 for b in rssi_buffers.values()):
                    pos = trilaterate_linear(ANCHORS, current_distances)
                    if pos is not None:
                        current_pos = pos
                        print(f"Dist: {current_distances[1]:.2f}, {current_distances[2]:.2f}, {current_distances[3]:.2f}")
                        print(f"Pos : X={pos[0]:.2f}, Y={pos[1]:.2f}")
                        print("-" * 15)

            except json.JSONDecodeError:
                pass
            except Exception:
                pass
                
    except Exception as e:
        print(f"Serial Error: {e}")

# --- PLOTTING ---
def update_plot(frame):
    plt.cla()
    
    # วาด Anchor Points
    for nid, pos in ANCHORS.items():
        plt.scatter(pos[0], pos[1], c='red', s=150, marker='s')
        # แสดงพิกัดทศนิยม 2 ตำแหน่ง
        plt.text(pos[0], pos[1]-0.2, f"Tx{nid}\n({pos[0]:.2f},{pos[1]:.2f})", ha='center', fontsize=9)
        
        # วาดวงกลมรัศมี
        circle = plt.Circle(pos, current_distances[nid], color='r', fill=False, linestyle=':', alpha=0.3)
        plt.gca().add_patch(circle)

    # วาดตำแหน่ง Rx
    if current_pos is not None:
        x, y = current_pos
        plt.scatter(x, y, c='blue', s=200, marker='*', label='Rx')
        plt.text(x, y+0.15, f"({x:.2f}, {y:.2f})", ha='center', color='blue', fontweight='bold')

    # ตั้งค่ากราฟ (ขยายขอบเขตเล็กน้อยเพื่อให้เห็น Tx3 ชัดเจน)
    plt.xlim(-0.5, 2.5) 
    plt.ylim(-0.5, 2.5)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.title(f"Trilateration (Tx Coordinates Updated)")
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.gca().set_aspect('equal', adjustable='box')

# --- MAIN ---
if __name__ == "__main__":
    t = threading.Thread(target=read_serial)
    t.daemon = True
    t.start()

    fig = plt.figure(figsize=(8, 8))
    ani = animation.FuncAnimation(fig, update_plot, interval=200)
    plt.show()