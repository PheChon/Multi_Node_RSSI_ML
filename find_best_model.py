import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

# --- CONFIGURATION ---
DATASET_FILE = 'cleaned_data.csv'
MAX_K = 20  # จะลองไล่หาค่า K ถึงเท่าไหร่ (ปกติ 1-20 ก็พอ)

# พิกัดจริง (Reference)
COORDINATES = {
    "P1": [0.35, 0.35], "P2": [0.71, 0.35], "P3": [1.06, 0.35],
    "P4": [0.35, 0.71], "P5": [0.71, 0.71], "P6": [1.06, 0.71],
    "P7": [0.35, 1.06], "P8": [0.71, 1.06], "P9": [1.06, 1.06]
}

def find_best_k():
    print(f"กำลังโหลดข้อมูลจาก {DATASET_FILE}...")
    try:
        df = pd.read_csv(DATASET_FILE)
    except FileNotFoundError:
        print("[Error] ไม่พบไฟล์ข้อมูล!")
        return

    # 1. เตรียมข้อมูล
    X = df[['Tx1', 'Tx2', 'Tx3']].values
    Y = np.array([COORDINATES.get(label, [0,0]) for label in df['Position_Label']])

    # 2. แบ่งข้อมูล (Train 80% / Test 20%)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    print(f"-> เริ่มการค้นหาค่า K ที่ดีที่สุด (1 - {MAX_K})...")
    print("-" * 50)
    print(f"{'K':<5} | {'Mean Error (m)':<15} | {'Max Error (m)':<15}")
    print("-" * 50)

    results = []
    best_k = 0
    min_error = float('inf')
    best_model = None

    # 3. ลูปทดสอบค่า K ต่างๆ (เฉพาะเลขคี่ เพื่อลดโอกาสเสมอในการโหวต)
    for k in range(1, MAX_K + 1):
        # สร้างและเทรนโมเดล
        model = KNeighborsRegressor(n_neighbors=k, weights='distance')
        model.fit(X_train, Y_train)

        # ทดสอบ
        Y_pred = model.predict(X_test)
        
        # คำนวณ Error (Euclidean Distance)
        errors = np.sqrt(np.sum((Y_test - Y_pred)**2, axis=1))
        mean_error = np.mean(errors)
        max_error = np.max(errors)

        results.append(mean_error)
        
        print(f"{k:<5} | {mean_error:.4f} m        | {max_error:.4f} m")

        # เช็คว่าเป็นแชมป์ใหม่หรือไม่?
        if mean_error < min_error:
            min_error = mean_error
            best_k = k
            best_model = model

    print("-" * 50)
    print(f"\n🏆 สรุป: ค่า K ที่ดีที่สุดคือ K = {best_k}")
    print(f"   ด้วยความคลาดเคลื่อนเฉลี่ย: {min_error*100:.2f} cm")

    # 4. บันทึกโมเดลที่ดีที่สุด
    best_model_filename = f'best_knn_model_k{best_k}.pkl'
    joblib.dump(best_model, best_model_filename)
    print(f"[Success] บันทึกโมเดลที่ดีที่สุดไว้ที่: '{best_model_filename}'")
    print("คุณสามารถนำไฟล์นี้ไปใช้ในโค้ด Real-time ได้เลย!")

    # 5. วาดกราฟเปรียบเทียบ
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, MAX_K + 1), results, marker='o', linestyle='-', color='b')
    plt.title('Error Rate vs. K Value')
    plt.xlabel('K (Number of Neighbors)')
    plt.ylabel('Mean Error (Meters)')
    plt.xticks(range(1, MAX_K + 1))
    plt.grid(True)
    
    # วงกลมจุดที่ดีที่สุด
    plt.plot(best_k, min_error, marker='o', markersize=12, markerfacecolor='red', markeredgecolor='black')
    plt.text(best_k, min_error + 0.01, f'Best K={best_k}', ha='center', color='red', fontweight='bold')
    
    plt.show()

if __name__ == "__main__":
    find_best_k()