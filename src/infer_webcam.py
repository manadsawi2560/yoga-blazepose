import os
import cv2
import mediapipe as mp
import numpy as np
from joblib import load
from features import landmarks_to_feature

mp_pose = mp.solutions.pose

def main():
    model = load("models/yoga_cls.joblib")
    scaler = load("models/scaler.joblib")
    le = load("models/label_encoder.joblib") if os.path.exists("models/label_encoder.joblib") else None

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera"); return

    with mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False) as pose:
        while True:
            ok, frame = cap.read()
            if not ok: break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            label = "No pose"
            if res.pose_landmarks:
                lm = [[p.x, p.y, p.z, p.visibility] for p in res.pose_landmarks.landmark]
                feat = landmarks_to_feature(lm).reshape(1, -1)
                X = scaler.transform(feat)
                pred = model.predict(X)[0]
                label = le.inverse_transform([pred])[0] if le is not None else str(pred)

                mp.solutions.drawing_utils.draw_landmarks(
                    frame, res.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS,
                    mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=2),
                    mp.solutions.drawing_utils.DrawingSpec(thickness=1)
                )

            cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
            cv2.imshow("Yoga-82 BlazePose (XGB)", frame)
            if cv2.waitKey(1) & 0xFF == 27: break  # ESC เพื่อออก

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
