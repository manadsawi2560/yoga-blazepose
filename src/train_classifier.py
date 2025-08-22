import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight
from joblib import dump
from xgboost import XGBClassifier

# สำหรับเซฟรูปในสภาพแวดล้อมที่ไม่มีจอ
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_xy(csv_path: Path):
    df = pd.read_csv(csv_path)
    y = df["label"].values
    X = df.drop(columns=["label"]).values.astype(np.float32)
    return X, y


def main(csv_train: Path, csv_val: Path, csv_test: Path, out_dir: Path,
         n_estimators=1000, learning_rate=0.05, max_depth=8,
         subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, reg_alpha=0.0,
         min_child_weight=1.0, use_gpu=False, seed=42):

    # ---- load data ----
    X_tr, y_tr_raw = load_xy(csv_train)
    X_va, y_va_raw = load_xy(csv_val)
    X_te, y_te_raw = load_xy(csv_test)

    # ---- encode labels (fit ที่ train เท่านั้น) ----
    le = LabelEncoder()
    y_tr = le.fit_transform(y_tr_raw)
    y_va = le.transform(y_va_raw)
    y_te = le.transform(y_te_raw)
    n_classes = len(le.classes_)

    # ---- scale (tree ไม่จำเป็น แต่ให้คงรูปกับ inference) ----
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_va_sc = scaler.transform(X_va)
    X_te_sc = scaler.transform(X_te)

    # ---- class weights -> sample_weight สำหรับ train ----
    cls_w = compute_class_weight(class_weight="balanced",
                                 classes=np.arange(n_classes), y=y_tr)
    w_map = {i: w for i, w in enumerate(cls_w)}
    w_tr = np.array([w_map[i] for i in y_tr], dtype=np.float32)

    # ---- XGBoost params (API ≥ 2.0) ----
    device_cfg = dict(tree_method="hist", device="cuda") if use_gpu else dict(tree_method="hist", device="cpu")

    clf = XGBClassifier(
        objective="multi:softprob", num_class=n_classes,
        n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
        subsample=subsample, colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda, reg_alpha=reg_alpha, min_child_weight=min_child_weight,
        random_state=seed, n_jobs=-1, verbosity=1,
        eval_metric="mlogloss",          # ✅ ใส่ใน constructor
        early_stopping_rounds=50,        # ✅ ใส่ใน constructor
        **device_cfg
    )

    # ---- train + early stopping บน val ----
    clf.fit(
        X_tr_sc, y_tr,
        sample_weight=w_tr,
        eval_set=[(X_va_sc, y_va)],
        verbose=True
    )

    # ---- evaluate on test ----
    y_pred = clf.predict(X_te_sc)

    # 1) classification report -> CSV
    report_dict = classification_report(
        le.inverse_transform(y_te),
        le.inverse_transform(y_pred),
        digits=4,
        output_dict=True
    )
    report_df = pd.DataFrame(report_dict).transpose()

    # 2) confusion matrices (raw + normalized)
    cm = confusion_matrix(y_te, y_pred, labels=np.arange(n_classes))
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = cm.astype("float") / np.clip(cm.sum(axis=1, keepdims=True), 1.0, None)

    # ---- save artifacts ----
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # โมเดล/สเกลเลอร์/เอนโค้ดเดอร์
    dump(clf, out_dir / "yoga_cls.joblib")
    dump(scaler, out_dir / "scaler.joblib")
    dump(le, out_dir / "label_encoder.joblib")
    (out_dir / "classes.txt").write_text("\n".join(le.classes_), encoding="utf-8")

    # รายงาน
    report_df.to_csv(metrics_dir / "classification_report.csv", encoding="utf-8")
    np.save(metrics_dir / "confusion_matrix_raw.npy", cm)
    np.save(metrics_dir / "confusion_matrix_norm.npy", cm_norm)

    # รูป Confusion Matrix (Normalized, no labels – ป้องกันล้นรูปเมื่อคลาสเยอะ)
    fig, ax = plt.subplots(figsize=(22, 20), dpi=150)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm)
    disp.plot(include_values=False, cmap="viridis", ax=ax, colorbar=True)
    ax.set_title("Normalized Confusion Matrix (Test set)")
    # ซ่อน tick labels เพราะมีคลาสจำนวนมาก
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    fig.savefig(metrics_dir / "confusion_matrix_norm.png")
    plt.close(fig)

    # อีกไฟล์: top-30 classes by support (อ่านง่ายขึ้น)
    counts = np.bincount(y_te, minlength=n_classes)
    topk = min(30, n_classes)
    top_idx = np.argsort(counts)[::-1][:topk]
    cm_top = cm_norm[np.ix_(top_idx, top_idx)]
    labels_top = le.inverse_transform(top_idx)

    fig2, ax2 = plt.subplots(figsize=(16, 14), dpi=150)
    disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_top, display_labels=labels_top)
    disp2.plot(include_values=False, cmap="viridis", ax=ax2, colorbar=True)
    plt.setp(ax2.get_xticklabels(), rotation=90, ha="center", fontsize=6)
    plt.setp(ax2.get_yticklabels(), fontsize=6)
    ax2.set_title(f"Top-{topk} Classes (by support) – Normalized CM")
    plt.tight_layout()
    fig2.savefig(metrics_dir / "confusion_matrix_top30.png")
    plt.close(fig2)

    # สรุปสั้น ๆ ใน stdout
    print("\nSaved artifacts to:", out_dir)
    print(" - metrics/classification_report.csv")
    print(" - metrics/confusion_matrix_norm.png")
    print(" - metrics/confusion_matrix_top30.png")
    print(" - metrics/confusion_matrix_raw.npy, confusion_matrix_norm.npy")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_train", type=str, default="data/yoga82_train.csv")
    ap.add_argument("--csv_val", type=str, default="data/yoga82_val.csv")
    ap.add_argument("--csv_test", type=str, default="data/yoga82_test.csv")
    ap.add_argument("--out_dir", type=str, default="models")
    ap.add_argument("--n_estimators", type=int, default=1000)
    ap.add_argument("--learning_rate", type=float, default=0.05)
    ap.add_argument("--max_depth", type=int, default=8)
    ap.add_argument("--subsample", type=float, default=0.8)
    ap.add_argument("--colsample_bytree", type=float, default=0.8)
    ap.add_argument("--reg_lambda", type=float, default=1.0)
    ap.add_argument("--reg_alpha", type=float, default=0.0)
    ap.add_argument("--min_child_weight", type=float, default=1.0)
    ap.add_argument("--use_gpu", action="store_true")
    args = ap.parse_args()

    main(
        Path(args.csv_train), Path(args.csv_val), Path(args.csv_test), Path(args.out_dir),
        n_estimators=args.n_estimators, learning_rate=args.learning_rate, max_depth=args.max_depth,
        subsample=args.subsample, colsample_bytree=args.colsample_bytree,
        reg_lambda=args.reg_lambda, reg_alpha=args.reg_alpha,
        min_child_weight=args.min_child_weight, use_gpu=args.use_gpu
    )
