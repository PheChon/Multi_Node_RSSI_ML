import serial
import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import threading
from collections import deque

# --- CONFIGURATION (ตั้งค่าระบบ) ---
SERIAL_PORT = 'COM10'  # <--- อย่าลืมเช็ค Port
BAUD_RATE = 115200

# พิกัด Anchors (ตามที่คุณตั้งค่าไว้)
ANCHORS = {
    1: np.array([0.0, 0.0]),                # Tx1: (0, 0)
    2: np.array([np.sqrt(3), 0.0]),         # Tx2: (√3, 0)
    3: np.array([np.sqrt(3)/2, np.sqrt(3)]) # Tx3: (√3/2, √3)
}

# ค่าคงที่จากเอกสาร
RSSI_A = -48.5      
PATH_LOSS_N = 2.2   

# การกรองข้อมูล
FILTER_SIZE = 5

# --- GLOBAL VARIABLES ---
rssi_buffers = {1: deque(maxlen=FILTER_SIZE), 2: deque(maxlen=FILTER_SIZE), 3: deque(maxlen=FILTER_SIZE)}
current_distances = {1: 0.0, 2: 0.0, 3: 0.0}
current_pos = None

# ตัวแปรเก็บตำแหน่งจริง (เฉลย)
real_x = 0.0
real_y = 0.0

# --- CALCULATION FUNCTIONS ---

def rssi_to_distance(rssi):
    if rssi >= 0: rssi = -1.0 
    exponent = (RSSI_A - rssi) / (10 * PATH_LOSS_N)
    dist = 10 ** exponent
    return dist

def trilaterate_linear(anchors, distances):
    try:
        xa, ya = anchors[1]
        xb, yb = anchors[2]
        xc, yc = anchors[3]
        
        da = distances[1]
        db = distances[2]
        dc = distances[3]

        A1 = 2 * (xa - xb)
        B1 = 2 * (ya - yb)
        C1 = (db**2 - da**2) - (xb**2 - xa**2) - (yb**2 - ya**2)

        A2 = 2 * (xc - xb)
        B2 = 2 * (yc - yb)
        C2 = (db**2 - dc**2) - (xb**2 - xc**2) - (yb**2 - yc**2)

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
        print(f"\n[System] Connected to {SERIAL_PORT}")
        print(f"[Test] Target Real Position: ({real_x}, {real_y})")
        print("-" * 50)
        
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

                # คำนวณตำแหน่งและ Error
                if all(len(b) > 0 for b in rssi_buffers.values()):
                    pos = trilaterate_linear(ANCHORS, current_distances)
                    if pos is not None:
                        current_pos = pos
                        
                        # --- ส่วนคำนวณ Error ---
                        # Euclidean Distance: sqrt((x2-x1)^2 + (y2-y1)^2)
                        error_dist = np.sqrt((pos[0] - real_x)**2 + (pos[1] - real_y)**2)
                        
                        print(f"Est: ({pos[0]:.2f}, {pos[1]:.2f}) | Real: ({real_x}, {real_y}) | ERROR: {error_dist:.4f} m")

            except json.JSONDecodeError:
                pass
            except Exception:
                pass
                
    except Exception as e:
        print(f"Serial Error: {e}")

# --- PLOTTING ---
def update_plot(frame):
    plt.cla()
    
    # 1. วาด Anchor Points
    for nid, pos in ANCHORS.items():
        plt.scatter(pos[0], pos[1], c='red', s=150, marker='s')
        plt.text(pos[0], pos[1]-0.2, f"Tx{nid}", ha='center', fontsize=9)
        # วาดวงกลมรัศมี (Optional: เอาออกได้ถ้าลายตา)
        # circle = plt.Circle(pos, current_distances[nid], color='r', fill=False, linestyle=':', alpha=0.1)
        # plt.gca().add_patch(circle)

    # 2. วาดจุดจริง (Ground Truth) - สีเขียว
    plt.scatter(real_x, real_y, c='green', s=200, marker='X', label='REAL Pos')
    plt.text(real_x, real_y+0.1, f"Real\n({real_x},{real_y})", ha='center', color='green', fontweight='bold')

    # 3. วาดตำแหน่งที่คำนวณได้ (Estimated) - สีน้ำเงิน
    if current_pos is not None:
        x, y = current_pos
        plt.scatter(x, y, c='blue', s=200, marker='*', label='ESTIMATED')
        plt.text(x, y-0.2, f"Est\n({x:.2f}, {y:.2f})", ha='center', color='blue')
        
        # ลากเส้นเชื่อมแสดง Error
        plt.plot([real_x, x], [real_y, y], 'r--', alpha=0.5, label='Error Line')
        
        # คำนวณ Error สดๆ เพื่อโชว์บนหัวกราฟ
        err = np.sqrt((x - real_x)**2 + (y - real_y)**2)
        plt.title(f"Real-time Error Test (Error: {err*100:.1f} cm)")
    else:
        plt.title("Waiting for data...")

    # ตั้งค่ากราฟ
    plt.xlim(-0.5, 2.5) 
    plt.ylim(-0.5, 2.5)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.legend(loc='upper right')
    plt.gca().set_aspect('equal', adjustable='box')

# --- MAIN ---
if __name__ == "__main__":
    # 1. รับค่าพิกัดจริงจากผู้ใช้ก่อนเริ่ม
    print("=== RSSI ERROR TESTING PROGRAM ===")
    try:
        real_x = float(input("Enter REAL X position (meters): "))
        real_y = float(input("Enter REAL Y position (meters): "))
    except ValueError:
        print("Invalid input! Defaulting to (0.866, 0.5) [Center]")
        real_x, real_y = 0.866, 0.5

    # 2. เริ่ม Thread
    t = threading.Thread(target=read_serial)
    t.daemon = True
    t.start()

    # 3. เปิดกราฟ
    fig = plt.figure(figsize=(8, 8))
    ani = animation.FuncAnimation(fig, update_plot, interval=200)
    plt.show()