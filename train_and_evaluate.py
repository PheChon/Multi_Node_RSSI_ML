import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import joblib  # ไว้บันทึกโมเดล

# --- CONFIGURATION ---
DATASET_FILE = 'cleaned_data.csv'  # ใช้ไฟล์ที่ Clean แล้ว
K_NEIGHBORS = 9         # <--- ลองเปลี่ยนเลขนี้ดู (เช่น 3, 5, 7) ชื่อไฟล์จะเปลี่ยนตาม

# พิกัดจริงของจุด P1-P9 (Reference)
COORDINATES = {
    "P1": [0.35, 0.35], "P2": [0.71, 0.35], "P3": [1.06, 0.35],
    "P4": [0.35, 0.71], "P5": [0.71, 0.71], "P6": [1.06, 0.71],
    "P7": [0.35, 1.06], "P8": [0.71, 1.06], "P9": [1.06, 1.06]
}

def evaluate_model():
    print(f"กำลังโหลดข้อมูลจาก {DATASET_FILE}...")
    try:
        df = pd.read_csv(DATASET_FILE)
    except FileNotFoundError:
        print("[Error] ไม่พบไฟล์ข้อมูล! กรุณาเช็คชื่อไฟล์")
        return

    # 1. เตรียมข้อมูล Input (X) และ Output (Y)
    X = df[['Tx1', 'Tx2', 'Tx3']].values
    
    # แปลง Label (P1..) เป็นพิกัดจริง (X, Y)
    Y = np.array([COORDINATES.get(label, [0,0]) for label in df['Position_Label']])

    # 2. แบ่งข้อมูล 80% Train, 20% Test
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    print(f"-> จำนวนข้อมูลทั้งหมด: {len(df)}")
    print(f"-> ใช้สอน (Train 80%): {len(X_train)}")
    print(f"-> ใช้สอบ (Test 20%):  {len(X_test)}")
    print("-" * 30)

    # 3. สร้างและสอนโมเดล (k-NN)
    print(f"[AI] กำลังเทรนโมเดลด้วยค่า K={K_NEIGHBORS}...")
    model = KNeighborsRegressor(n_neighbors=K_NEIGHBORS, weights='distance')
    model.fit(X_train, Y_train)
    print("[AI] โมเดลเรียนรู้เสร็จแล้ว!")

    # 4. ทดสอบโมเดล (ทำนายพิกัดจาก X_test)
    Y_pred = model.predict(X_test)

    # 5. คำนวณความคลาดเคลื่อน (Error Metrics)
    errors = np.sqrt(np.sum((Y_test - Y_pred)**2, axis=1))
    
    mean_error = np.mean(errors)
    max_error = np.max(errors)
    threshold = 0.5 # 50 cm
    accuracy_pass = np.sum(errors < threshold) / len(errors) * 100

    print("\n" + "="*30)
    print(f"   ผลการสอบ (K={K_NEIGHBORS})")
    print("="*30)
    print(f"ความคลาดเคลื่อนเฉลี่ย (Mean Error): {mean_error:.4f} เมตร ({mean_error*100:.2f} ซม.)")
    print(f"ผิดพลาดสูงสุด (Max Error):       {max_error:.4f} เมตร")
    print(f"ความแม่นยำ (ผิดไม่เกิน {threshold*100} ซม.): {accuracy_pass:.2f}%")
    print("="*30)

    # 6. บันทึกโมเดลเก็บไว้ใช้จริง (ตั้งชื่อไฟล์ตามค่า K)
    model_filename = f'knn{K_NEIGHBORS}_model.pkl'  # <--- แก้ตรงนี้ครับ
    joblib.dump(model, model_filename)
    print(f"[Success] บันทึกโมเดลไว้ที่ '{model_filename}' เรียบร้อย")

    # 7. วาดกราฟแสดงผล
    plot_results(Y_test, Y_pred, errors)

def plot_results(y_true, y_pred, errors):
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true[:, 0], y_true[:, 1], c='green', alpha=0.5, label='Actual Position')
    plt.scatter(y_pred[:, 0], y_pred[:, 1], c='red', alpha=0.5, marker='x', label='AI Prediction')
    
    for i in range(min(50, len(y_true))):
        plt.plot([y_true[i, 0], y_pred[i, 0]], [y_true[i, 1], y_pred[i, 1]], 'gray', alpha=0.3)

    plt.title(f"AI Evaluation K={K_NEIGHBORS} (Mean Error: {np.mean(errors)*100:.1f} cm)")
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.legend()
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

if __name__ == "__main__":
    evaluate_model()