"""build_notebook.py - sinh code/lab_day2.ipynb (notebook Colab/Kaggle chạy end-to-end).

Chạy:  python code/build_notebook.py
Ghi:   code/lab_day2.ipynb
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

CELLS: list[tuple[str, str]] = []


def md(s):
    CELLS.append(("markdown", s))


def code(s):
    CELLS.append(("code", s))


md("""# Lab Day 2 — Backbone, công thức huấn luyện và suy luận trên DeepWeeds

Notebook **chạy end-to-end trên Google Colab** (GPU T4+): tự động clone repo, tải dataset,
train, đánh giá bằng `eval.py`, sinh `results.xlsx` / `report.md` / `curves/` / `predictions/`.

**Quy tắc quan trọng nhất:** chọn mọi thứ trên **val**; **test chỉ chạy một lần mỗi seed** ở Bước 4.

Chạy lần lượt từ trên xuống. Đặt `QUICK=True` để chạy bản rút gọn (đủ ngưỡng điểm) hoặc `False` để chạy đầy đủ.
""")

md("## 0. Cài đặt môi trường")
code('''# Cài thư viện. Colab đã có torch/torchvision; cần timm, openpyxl, scikit-learn.
!pip -q install timm openpyxl scikit-learn

import sys, os, platform, subprocess, shutil, json, time
import numpy as np, pandas as pd, torch, timm
print("python", platform.python_version(), "| torch", torch.__version__,
      "| torchvision", __import__("torchvision").__version__, "| timm", timm.__version__)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "KHÔNG CÓ GPU — hãy bật GPU trong Runtime!")
assert torch.cuda.is_available(), "Cần GPU để train. Runtime > Change runtime type > GPU (T4)."
''')

md("## 1. Mount Google Drive (lưu bền kết quả) + cấu hình")
code('''from google.colab import drive
drive.mount("/content/drive")

REPO_URL = "https://github.com/meth04/K4-DAY02-NguyenVanThan-02859.git"
DRIVE = "/content/drive/MyDrive"
REPO_DIR = f"{DRIVE}/K4-DAY02-NguyenVanThan-02859"     # repo nằm trên Drive -> sống sót khi ngắt phiên
DATA_ROOT = f"{DRIVE}/deepweeds_data"                   # dataset cũng đặt trên Drive
QUICK = True                                            # True: bản rút gọn đủ điểm; False: đầy đủ
EPOCHS = 8 if QUICK else 12                             # epoch cho Bước 1 (backbone)
ABL_EPOCHS = 6 if QUICK else 8                          # epoch cho Bước 2 (ablation)
FINAL_EPOCHS = 10 if QUICK else 15                      # epoch cho chung kết
print("REPO_DIR =", REPO_DIR, "| QUICK =", QUICK)
''')

md("## 2. Clone repo (tự động)")
code('''if not os.path.exists(REPO_DIR):
    subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)
else:
    subprocess.run(["git", "-C", REPO_DIR, "pull", "--ff-only"], check=False)

os.chdir(REPO_DIR)
sys.path.insert(0, REPO_DIR)            # để import eval.py (gốc repo)
sys.path.insert(0, f"{REPO_DIR}/code")  # các module train/model/...

import train, dataset, model, losses, inference, benchmark, study
from train import Config, run
print("CWD:", os.getcwd())
print("repo:", os.listdir(REPO_DIR)[:12])
''')

md("""## 3. Tải dataset DeepWeeds (~490 MB, kiểm tra MD5)

Ảnh từ Zenodo, nhãn + fold chia sẵn từ GitHub tác giả. Tải vào Drive để lần sau không tải lại.
""")
code('''import hashlib
os.makedirs(f"{DATA_ROOT}/labels", exist_ok=True)
IMAGES_DIR = f"{DATA_ROOT}/images"
LABELS_DIR = f"{DATA_ROOT}/labels"
zip_path = f"{DATA_ROOT}/images.zip"

have_images = os.path.isdir(IMAGES_DIR) and len(os.listdir(IMAGES_DIR)) >= 17509
if not have_images:
    if not os.path.exists(zip_path):
        !wget -q --show-progress -O {zip_path} "https://zenodo.org/records/7939060/files/images.zip?download=1"
    h = hashlib.md5()
    with open(zip_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    assert h.hexdigest() == "b7b30f96d466fba86016aa5a26606e0f", f"MD5 sai: {h.hexdigest()}"
    os.makedirs(IMAGES_DIR, exist_ok=True)
    !unzip -q -n {zip_path} -d {IMAGES_DIR}
print("Số ảnh:", len(os.listdir(IMAGES_DIR)))

BASE = "https://raw.githubusercontent.com/AlexOlsen/DeepWeeds/master/labels"
for name in ["labels", "train_subset0", "val_subset0", "test_subset0"]:
    p = f"{LABELS_DIR}/{name}.csv"
    if not os.path.exists(p):
        !wget -q -O {p} {BASE}/{name}.csv
print("Labels:", os.listdir(LABELS_DIR))
''')

md("## 4. Bước 0 — EDA và kiểm tra chia dữ liệu (bắt buộc)")
code('''train_df, val_df, test_df = dataset.load_split(LABELS_DIR, fold=0)
info = dataset.check_split(train_df, val_df, test_df, IMAGES_DIR)

import matplotlib.pyplot as plt
counts = pd.DataFrame({s: pd.Series(info["per_class"][s]).sort_index() for s in ["train", "val", "test"]}).fillna(0)
counts.index = dataset.CLASS_NAMES
counts.plot(kind="bar", figsize=(12, 4), title="Phân bố lớp theo tập (fold 0)")
plt.ylabel("số ảnh"); plt.tight_layout(); plt.show()
print(counts.astype(int))
print("Tỉ lệ lớp lớn nhất / nhỏ nhất (train):",
      round(counts["train"].max() / counts["train"].min(), 1))
''')

md("Xem vài ảnh mỗi lớp (kiểm tra nhãn khớp ảnh, `Negative` trông thế nào):")
code('''from PIL import Image
fig, axes = plt.subplots(3, 3, figsize=(9, 9))
for c, ax in enumerate(axes.ravel()):
    sub = train_df[train_df["Label"] == c]
    if len(sub):
        img = Image.open(f"{IMAGES_DIR}/{sub.iloc[0]['Filename']}").convert("RGB")
        ax.imshow(img); ax.set_title(dataset.CLASS_NAMES[c], fontsize=9)
    ax.axis("off")
plt.tight_layout(); plt.show()
''')

md("""### 4.1. Kiểm tra pipeline trước khi chạy thật (GUIDE.md mục 1.3)

Loss ban đầu ≈ ln(9) = 2.197; overfit 1 batch nhỏ; xem ảnh sau augmentation.
""")
code('''import self_test as st
ok = [st.test_focal_gamma0_equals_ce(), st.test_ls_eps0_equals_ce(), st.test_cutmix_area(),
      st.test_fuse_conv_bn(), st.test_temperature(), st.test_initial_loss(),
      st.test_overfit_one_batch()]
print(f"{sum(ok)}/{len(ok)} kiểm tra pipeline đạt.")
''')

md("## 5. Metadata backbone (#params, GMAC, tag trọng số)")
code('''!python code/backbone_meta.py --out runs/backbone_meta.json \\
    --backbones resnet50 resnext50_32x4d convnext_tiny deit_small_patch16_224 efficientnet_b0 mobilenetv3_large_100
''')

md("""## 6. Bước 1 — So sánh ≥ 5 backbone (cùng công thức nền, 1 seed)

Đủ ràng buộc: ResNet, ResNeXt/ConvNeXt, transformer (DeiT), mạng nhẹ (EfficientNet, MobileNetV3).
""")
code('''def run_exp(**kw):
    base = dict(images_dir=IMAGES_DIR, labels_dir=LABELS_DIR, out_dir="runs",
                pred_dir="predictions", curves_dir="curves", num_workers=2, amp=True)
    base.update(kw)
    return run(Config(**base))

BACKBONES = [
    ("B01", "resnet50"),                  # ResNet (mốc)
    ("B02", "resnext50_32x4d"),           # ResNeXt
    ("B03", "convnext_tiny"),             # ConvNeXt
    ("B04", "deit_small_patch16_224"),    # transformer
    ("B05", "efficientnet_b0"),           # nhẹ
    ("B06", "mobilenetv3_large_100"),     # nhẹ hơn nữa
]
for eid, bb in BACKBONES:
    run_exp(exp_id=eid, backbone=bb, seed=0, epochs=EPOCHS, init="finetune", aug="basic",
            loss="ce", img_size=224, batch_size=64)
''')

md("Chọn 1–2 backbone đi tiếp dựa trên macro-F1 val (tự động chọn tốt nhất):")
code('''import glob
rows = []
for p in sorted(glob.glob("runs/B*/seed0/summary.json")):
    d = json.load(open(p))
    rows.append({k: d[k] for k in ["exp_id", "backbone", "val_macro_f1", "val_top1",
                                    "params_m", "train_time_per_epoch_s"]})
bb_table = pd.DataFrame(rows).sort_values("val_macro_f1", ascending=False)
print(bb_table.to_string(index=False))
CHOSEN_BB = bb_table.iloc[0]["backbone"]
CHOSEN_EXP = bb_table.iloc[0]["exp_id"]
print("\\n=> Chọn backbone:", CHOSEN_BB, f"(exp {CHOSEN_EXP})")
print("Ghi chu: chon backbone theo macro-F1 val; co the chon theo can bang F1/do tre "
      "neu can trien khai thoi gian thuc (xem bang Latency).")
''')

md("""## 7. Bước 2 — Công thức huấn luyện (≥ 3 trục, mỗi lần khác nền 1 yếu tố)

Trục: A khởi tạo, B augmentation, C loss, D sampler, E optimizer/LR, F EMA, G độ phân giải.
""")
code('''# T00 = công thức nền trên backbone đã chọn
run_exp(exp_id="T00", backbone=CHOSEN_BB, seed=0, epochs=ABL_EPOCHS,
        init="finetune", aug="basic", loss="ce", img_size=224)

# Mỗi thí nghiệm chỉ khác T00 đúng MỘT yếu tố
run_exp(exp_id="T01", backbone=CHOSEN_BB, seed=0, epochs=ABL_EPOCHS, init="frozen", aug="basic", loss="ce")       # A
run_exp(exp_id="T02", backbone=CHOSEN_BB, seed=0, epochs=ABL_EPOCHS, init="scratch", aug="basic", loss="ce")      # A
run_exp(exp_id="T03", backbone=CHOSEN_BB, seed=0, epochs=ABL_EPOCHS, init="finetune", aug="color", loss="ce")     # B
run_exp(exp_id="T04", backbone=CHOSEN_BB, seed=0, epochs=ABL_EPOCHS, init="finetune", aug="trivial", loss="ce")   # B
run_exp(exp_id="T05", backbone=CHOSEN_BB, seed=0, epochs=ABL_EPOCHS, init="finetune", aug="basic", loss="ce", mix="mixup")   # B
run_exp(exp_id="T06", backbone=CHOSEN_BB, seed=0, epochs=ABL_EPOCHS, init="finetune", aug="basic", loss="ce", mix="cutmix")  # B
run_exp(exp_id="T07", backbone=CHOSEN_BB, seed=0, epochs=ABL_EPOCHS, init="finetune", aug="basic", loss="ls", label_smoothing=0.1)  # C
run_exp(exp_id="T08", backbone=CHOSEN_BB, seed=0, epochs=ABL_EPOCHS, init="finetune", aug="basic", loss="focal", focal_gamma=2.0)   # C
run_exp(exp_id="T09", backbone=CHOSEN_BB, seed=0, epochs=ABL_EPOCHS, init="finetune", aug="basic", loss="ce_weighted", class_weight_beta=0.0)  # C
run_exp(exp_id="T10", backbone=CHOSEN_BB, seed=0, epochs=ABL_EPOCHS, init="finetune", aug="basic", loss="ce", sampler="balanced")   # D
run_exp(exp_id="T11", backbone=CHOSEN_BB, seed=0, epochs=ABL_EPOCHS, init="finetune", aug="basic", loss="ce", ema_decay=0.999)       # F
''')

md("Kết hợp các yếu tố tốt nhất (kiểm tra cộng dồn hay triệt tiêu) — chỉnh theo kết quả phía trên:")
code('''# Ví dụ kết hợp: label smoothing + cutmix + EMA (đổi theo kết quả ablation của bạn)
run_exp(exp_id="T12", backbone=CHOSEN_BB, seed=0, epochs=ABL_EPOCHS,
        init="finetune", aug="basic", loss="ls", label_smoothing=0.1,
        mix="cutmix", ema_decay=0.999)

abl = pd.DataFrame([json.load(open(p)) for p in sorted(glob.glob("runs/T*/seed0/summary.json"))])
abl = abl[["exp_id", "val_macro_f1", "val_top1", "init", "aug", "loss", "mix"]]
base = abl.loc[abl.exp_id == "T00", "val_macro_f1"]
abl["Δ vs T00"] = (abl["val_macro_f1"] - float(base.iloc[0])).round(4)
print(abl.sort_values("val_macro_f1", ascending=False).to_string(index=False))
''')

md("## 8. Bước 3 — Phương pháp suy luận (≥ 4) + đo độ trễ")
code('''# So sánh I00..I08 trên val bằng checkpoint T00, đo độ trễ p50/p95/p99 (warmup + synchronize)
inf_rows = study.inference_study(exp_id="T00", seed=0, out_dir="runs", iters=100, warmup=15)
inf_df = pd.DataFrame(inf_rows)
print(inf_df.to_string(index=False))
''')

md("## 9. Bước 4 — Chung kết: ≥ 3 seed, test đúng MỘT lần mỗi seed")

md("""Chỉnh `BEST` theo kết quả Bước 2/3 (ví dụ dưới đây là lựa chọn hợp lý). Sau đó chạy
cấu hình chung kết **và mốc** `T00` với cùng số seed, bật `save_test_predictions=True`.
""")
code('''# ==== CHỈNH CẤU HÌNH CHUNG KẾT Ở ĐÂY (dựa trên val) ====
BEST = dict(
    backbone=CHOSEN_BB,
    init="finetune",
    aug="basic",          # đổi thành giá trị thắng ở trục B nếu có
    loss="ls",            # đổi thành loss thắng ở trục C nếu có
    label_smoothing=0.1,
    mix=None,             # ví dụ "cutmix" nếu thắng
    ema_decay=None,       # ví dụ 0.999 nếu thắng
    epochs=FINAL_EPOCHS,
)
SEEDS = [0, 1, 2]         # >= 3 seed
print("Chung kết:", BEST)

# 1) Cấu hình chung kết F01
for s in SEEDS:
    run_exp(exp_id="F01", seed=s, save_test_predictions=True, **BEST)

# 2) Mốc: T00 + I00 (cùng số seed) để tính mức cải thiện
for s in SEEDS:
    run_exp(exp_id="T00", seed=s, backbone=CHOSEN_BB, init="finetune", aug="basic",
            loss="ce", epochs=BEST["epochs"], save_test_predictions=True)
''')

md("Sinh dự đoán test/val cho chung kết (TTA lật + temperature scaling) và mốc (1-view):")
code('''# Chung kết: TTA lật ngang (K=2) + temperature scaling (T khớp trên VAL)
study.finalize_test("F01", seeds=SEEDS, method="hflip", temperature=True,
                    out_dir="runs", pred_dir="predictions")

# Mốc: 1 view, không TTA, không temperature -> đúng "T00 + I00"
study.finalize_test("T00", seeds=SEEDS, method="identity", temperature=False,
                    out_dir="runs", pred_dir="predictions")
print("Xong. Các file trong predictions/:", sorted(os.listdir("predictions"))[:12], "...")
''')

md("## 10. Sản phẩm: results.xlsx, report.md, curves/")
code('''!python code/make_results.py --labels-dir {LABELS_DIR} --out-dir runs --pred-dir predictions --out results.xlsx --final F01 --baseline T00
!python code/make_report.py --labels-dir {LABELS_DIR} --xlsx results.xlsx --final F01 --baseline T00 --out report.md
print("curves/:", sorted(os.listdir("curves"))[:8], "...")
print("results.xlsx:", os.path.exists("results.xlsx"), "| report.md:", os.path.exists("report.md"))
''')

md("## 11. Chấm bằng eval.py (đúng định nghĩa của lớp) + tự chấm phần I")
code('''!python eval.py score --pred "predictions/F01_seed*_test.csv" --test-csv {LABELS_DIR}/test_subset0.csv --labels {LABELS_DIR}/labels.csv --tag F01 --out eval_out
!python eval.py score --pred "predictions/T00_seed*_test.csv" --test-csv {LABELS_DIR}/test_subset0.csv --labels {LABELS_DIR}/labels.csv --tag T00 --out eval_out
''')
code('''# Tự chấm phần I (I5: lấy p95 batch-1 của cấu hình thời gian thực từ bảng Latency)
lat = pd.DataFrame(json.load(open("runs/latency.json")))
p95 = float(lat.iloc[0]["p95"])
!python eval.py grade --final "predictions/F01_seed*_test.csv" --baseline "predictions/T00_seed*_test.csv" \\
    --uncal "predictions/F01uncal_seed*_test.csv" --final-val "predictions/F01_seed*_val.csv" \\
    --latency-p95-ms {p95} --latency-method proper \\
    --test-csv {LABELS_DIR}/test_subset0.csv --val-csv {LABELS_DIR}/val_subset0.csv \\
    --labels {LABELS_DIR}/labels.csv --out eval_out
''')

md("""## 12. Ma trận nhầm lẫn + phân tích lỗi (đưa vào báo cáo)""")
code('''import numpy as np
from eval import load_group, confusion_matrix
g = load_group("predictions/F01_seed*_test.csv", f"{LABELS_DIR}/test_subset0.csv", ref_what="test")
cm = sum(m["confusion"] for m in g.metrics)
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(9)); ax.set_xticklabels(dataset.CLASS_NAMES, rotation=90, fontsize=8)
ax.set_yticks(range(9)); ax.set_yticklabels(dataset.CLASS_NAMES, fontsize=8)
ax.set_xlabel("dự đoán"); ax.set_ylabel("nhãn thật"); ax.set_title("Ma trận nhầm lẫn (test, tổng qua seed)")
for i in range(9):
    for j in range(9):
        if cm[i, j]:
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=7,
                    color="white" if cm[i, j] > cm.max()/2 else "black")
plt.colorbar(im); plt.tight_layout(); plt.savefig("curves/confusion_F01.png", dpi=130); plt.show()
''')

md("""## 13. Lưu & nộp bài

- Notebook này nằm trong repo (`code/lab_day2.ipynb`) và repo nằm trên Drive → đã tự lưu.
- Commit **code, results.xlsx, report.md, curves/, predictions/** (KHÔNG commit dataset/checkpoint):
  ```bash
  cd {REPO_DIR}
  git add code results.xlsx report.md curves predictions SUBMISSION_README.md
  git commit -m "Lab Day 2: ket qua chung ket"
  git push
  ```
- Kiểm tra lại bằng `python eval.py score` cho từng nhóm file trước khi nộp.
""")
code('''print("Hoàn tất. Các sản phẩm:")
for p in ["results.xlsx", "report.md", "curves", "predictions", "code"]:
    print(" -", p, "->", "OK" if os.path.exists(p) else "THIẾU")
''')


def build():
    nb = {
        "cells": [],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "colab": {"provenance": [], "gpuType": "T4"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    for kind, src in CELLS:
        cell = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True)}
        if kind == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        nb["cells"].append(cell)
    out = HERE / "lab_day2.ipynb"
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Đã ghi {out} với {len(nb['cells'])} cell")


if __name__ == "__main__":
    build()
