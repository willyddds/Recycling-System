import tensorflow as tf
import numpy as np
import os
import cv2
from sklearn.metrics import classification_report, confusion_matrix

# === 1. 載入 int8 TFLite 模型 ===
interpreter = tf.lite.Interpreter(model_path="model_int8.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape'][1:3]

# 取得量化參數 (scale, zero_point)
input_scale, input_zero_point = input_details[0]['quantization']

# === 2. 定義影像預處理函式 ===
def preprocess_img(img_path, input_shape, scale, zero_point):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, input_shape)
    img = img.astype(np.float32) / 255.0  # 先歸一化到 [0,1]

    # 將資料量化為 int8
    img = img / scale + zero_point
    img = np.clip(img, -128, 127).astype(np.int8)
    img = np.expand_dims(img, axis=0)
    return img

# === 3. 載入測試資料 ===
def load_test_data(img_dir):
    X = []
    y = []
    class_names = sorted(os.listdir(img_dir))  # 每個子資料夾為一個類別
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    for class_name in class_names:
        class_folder = os.path.join(img_dir, class_name)
        if not os.path.isdir(class_folder):
            continue
        for fname in os.listdir(class_folder):
            if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                img_path = os.path.join(class_folder, fname)
                img = preprocess_img(img_path, input_shape, input_scale, input_zero_point)
                if img is not None:
                    X.append(img)
                    y.append(class_to_idx[class_name])
    return X, y, class_names

# === 4. 推論模型 ===
def inference_tflite(X):
    y_pred = []
    for img in X:
        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        pred = np.argmax(output, axis=-1)[0]
        y_pred.append(pred)
    return y_pred

# === 5. 主程式 (執行測試) ===
test_folder = "TrashBox"  # <-- 請修改為你的測試資料夾路徑
X_test, y_true, class_names = load_test_data(test_folder)
y_pred = inference_tflite(X_test)

# === 6. 輸出分類報告 (包含Accuracy, Precision, Recall, F1-score) ===
print("📊 Classification Report：\n")
print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

print("🔍 Confusion Matrix：\n")
print(confusion_matrix(y_true, y_pred))
