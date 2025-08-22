import argparse
from pathlib import Path
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
from tqdm import tqdm
from features import landmarks_to_feature

mp_pose = mp.solutions.pose
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def iter_images(split_dir: Path):
    classes = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])
    for cls in classes:
        for p in (split_dir / cls).rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                yield p, cls

def process_image(pose, img_path: Path):
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = pose.process(rgb)
    if not res.pose_landmarks:
        return None
    lm = [[q.x, q.y, q.z, q.visibility] for q in res.pose_landmarks.landmark]
    return landmarks_to_feature(lm)

def run_split(split_name: str, split_dir: Path, out_csv: Path):
    rows, labels = [], []
    with mp_pose.Pose(static_image_mode=True, model_complexity=1, enable_segmentation=False) as pose:
        paths = list(iter_images(split_dir))
        for p, cls in tqdm(paths, desc=f"[{split_name}]"):
            feat = process_image(pose, p)
            if feat is not None:
                rows.append(feat)
                labels.append(cls)
    if not rows:
        raise RuntimeError(f"No features extracted for split {split_name}")

    X = np.vstack(rows)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df["label"] = labels
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Saved {split_name}: {out_csv}  shape={df.shape}")

def main(data_root: Path, out_dir: Path):
    mapping = {
        "train": (data_root / "train", out_dir / "yoga82_train.csv"),
        "val":   (data_root / "val",   out_dir / "yoga82_val.csv"),
        "test":  (data_root / "test",  out_dir / "yoga82_test.csv"),
    }
    for name, (indir, outcsv) in mapping.items():
        if not indir.exists():
            print(f"Skip {name}: {indir} not found"); continue
        run_split(name, indir, outcsv)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="data")
    ap.add_argument("--out_dir", type=str, default="data")
    args = ap.parse_args()
    main(Path(args.data_root), Path(args.out_dir))
