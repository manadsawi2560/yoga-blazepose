import base64
from io import BytesIO
from pathlib import Path
import sys

from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
from joblib import load
import mediapipe as mp

# ให้ import src/features.py ได้
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "src"))
from features import landmarks_to_feature  # noqa: E402

app = Flask(__name__)

# โหลดโมเดลครั้งเดียวตอนสตาร์ต
MODEL_DIR = ROOT / "models"
model = load(MODEL_DIR / "yoga_cls.joblib")
scaler = load(MODEL_DIR / "scaler.joblib")
label_encoder = load(MODEL_DIR / "label_encoder.joblib")

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

ALLOWED = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def _read_image_from_request(file_storage):
    data = file_storage.read()
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

def _annotate_pose(frame_bgr, pose_landmarks):
    mp_drawing.draw_landmarks(
        frame_bgr,
        pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(thickness=2, circle_radius=2),
        mp_drawing.DrawingSpec(thickness=2),
    )
    return frame_bgr

def _bgr_to_base64_jpeg(img_bgr):
    ok, buf = cv2.imencode(".jpg", img_bgr)
    if not ok:
        return None
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

@app.get("/")
def index():
    return render_template("index.html")

@app.post("/predict")
def predict():
    if "file" not in request.files:
        return jsonify({"ok": False, "error": "no_file"}), 400
    f = request.files["file"]
    if f.filename == "":
        return jsonify({"ok": False, "error": "empty_filename"}), 400

    ext = Path(f.filename).suffix.lower()
    if ext not in ALLOWED:
        return jsonify({"ok": False, "error": f"unsupported_extension {ext}"}), 400

    img_bgr = _read_image_from_request(f)
    if img_bgr is None:
        return jsonify({"ok": False, "error": "cannot_read_image"}), 400

    # ใช้โหมด static image เพื่อความแม่นยำ
    with mp_pose.Pose(static_image_mode=True, model_complexity=1, enable_segmentation=False) as pose:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

    if not res.pose_landmarks:
        return jsonify({"ok": False, "error": "pose_not_detected"}), 200

    lm = [[p.x, p.y, p.z, p.visibility] for p in res.pose_landmarks.landmark]
    feat = landmarks_to_feature(lm).reshape(1, -1)
    X = scaler.transform(feat)

    y_idx = model.predict(X)[0]
    y_name = label_encoder.inverse_transform([y_idx])[0]

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        topk_idx = np.argsort(probs)[::-1][:5]
        topk = [
            {"label": label_encoder.inverse_transform([i])[0], "prob": float(probs[i])}
            for i in topk_idx
        ]
    else:
        topk = [{"label": y_name, "prob": 1.0}]

    annotated = img_bgr.copy()
    annotated = _annotate_pose(annotated, res.pose_landmarks)
    annotated_b64 = _bgr_to_base64_jpeg(annotated)

    return jsonify({"ok": True, "predicted": y_name, "topk": topk, "annotated": annotated_b64})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
