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

Notebook **chạy end-to-end trên Google Colab** (GPU T4+): tự clone repo, tải dataset, train,
đánh giá bằng `eval.py`, sinh `results.xlsx` / `report.md` / `curves/` / `predictions/`.

**Quy tắc quan trọng nhất:** chọn mọi thứ trên **val**; **test chỉ chạy một lần mỗi seed** ở Bước 4.

Chạy lần lượt từ trên xuống. `QUICK=True` = bản rút gọn (đủ ngưỡng điểm), `False` = đầy đủ.
""")

md("## 0. Cài đặt môi trường")
code('''# Colab đã có torch/torchvision; cần timm, openpyxl, scikit-learn.
!pip -q install timm openpyxl scikit-learn

import sys, os, platform, subprocess, shutil, json, time, glob
import numpy as np, pandas as pd, torch, torchvision, timm
print("python", platform.python_version(), "| torch", torch.__version__,
      "| torchvision", torchvision.__version__, "| timm", timm.__version__)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "KHÔNG CÓ GPU")
assert torch.cuda.is_available(), "Cần GPU: Runtime > Change runtime type > GPU (T4)."
''')

md("## 1. Mount Google Drive + cấu hình đường dẫn")
code('''from google.colab import drive
drive.mount("/content/drive")

REPO_URL = "https://github.com/meth04/K4-DAY02-NguyenVanThan-02859.git"
DRIVE    = "/content/drive/MyDrive"
REPO_DIR = f"{DRIVE}/K4-DAY02-NguyenVanThan-02859"   # repo trên Drive -> sống sót khi ngắt phiên
DATA     = f"{DRIVE}/deepweeds_data"                 # zip + labels trên Drive (tải 1 lần)
WORK     = "/content/lab"                            # đọc ảnh & ghi checkpoint trên ĐĨA LOCAL (nhanh)
RUNS, PRED, CURVES = f"{WORK}/runs", f"{WORK}/predictions", f"{WORK}/curves"

QUICK       = True          # True: bản rút gọn đủ điểm; False: đầy đủ
EPOCHS      = 8  if QUICK else 12   # Bước 1 (backbone)
ABL_EPOCHS  = 6  if QUICK else 8    # Bước 2 (ablation)
FINAL_EPOCHS= 10 if QUICK else 15   # chung kết
for d in (RUNS, PRED, CURVES, WORK):
    os.makedirs(d, exist_ok=True)
print("REPO_DIR =", REPO_DIR, "| WORK =", WORK, "| QUICK =", QUICK)
''')

md("## 2. Clone repo (tự động)")
code('''if not os.path.exists(REPO_DIR):
    subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)
else:
    subprocess.run(["git", "-C", REPO_DIR, "pull", "--ff-only"], check=False)

os.chdir(REPO_DIR)
sys.path.insert(0, REPO_DIR)            # import eval.py (gốc repo)
sys.path.insert(0, f"{REPO_DIR}/code")  # module train/model/...

import dataset, model, losses, inference, benchmark, study, self_test
from train import Config, run
print("CWD:", os.getcwd())
print("repo:", sorted(os.listdir(REPO_DIR))[:12])
''')

md("## 3. Tải dataset DeepWeeds (~490 MB, kiểm MD5) và giải nén ra đĩa local")
code('''import hashlib
os.makedirs(f"{DATA}/labels", exist_ok=True)
LABELS_DIR = f"{DATA}/labels"
IMAGES_DIR = f"{WORK}/images"          # LOCAL cho nhanh
zip_path   = f"{DATA}/images.zip"

if not (os.path.isdir(IMAGES_DIR) and len(os.listdir(IMAGES_DIR)) >= 17509):
    if not os.path.exists(zip_path):
        subprocess.run(["wget", "-q", "--show-progress", "-O", zip_path,
                        "https://zenodo.org/records/7939060/files/images.zip?download=1"], check=True)
    h = hashlib.md5()
    with open(zip_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    assert h.hexdigest() == "b7b30f96d466fba86016aa5a26606e0f", f"MD5 sai: {h.hexdigest()}"
    os.makedirs(IMAGES_DIR, exist_ok=True)
    subprocess.run(["unzip", "-q", "-n", zip_path, "-d", IMAGES_DIR], check=True)
print("Số ảnh:", len(os.listdir(IMAGES_DIR)))

BASE = "https://raw.githubusercontent.com/AlexOlsen/DeepWeeds/master/labels"
for name in ["labels", "train_subset0", "val_subset0", "test_subset0"]:
    p = f"{LABELS_DIR}/{name}.csv"
    if not os.path.exists(p):
        subprocess.run(["wget", "-q", "-O", p, f"{BASE}/{name}.csv"], check=True)
print("Labels:", sorted(os.listdir(LABELS_DIR)))
''')

md("## 4. Bước 0 — EDA và kiểm tra chia dữ liệu (bắt buộc)")
code('''train_df, val_df, test_df = dataset.load_split(LABELS_DIR, fold=0)
info = dataset.check_split(train_df, val_df, test_df, IMAGES_DIR)

import matplotlib.pyplot as plt
counts = pd.DataFrame({s: pd.Series(info["per_class"][s]).sort_index()
                       for s in ["train", "val", "test"]}).fillna(0)
counts.index = dataset.CLASS_NAMES
counts.plot(kind="bar", figsize=(12, 4), title="Phân bố lớp theo tập (fold 0)")
plt.ylabel("số ảnh"); plt.tight_layout(); plt.show()
print(counts.astype(int))
print("Tỉ lệ lớp lớn nhất / nhỏ nhất (train):",
      round(counts["train"].max() / counts["train"].min(), 1))
''')

code('''from PIL import Image
fig, axes = plt.subplots(3, 3, figsize=(9, 9))
for c, ax in enumerate(axes.ravel()):
    sub = train_df[train_df["Label"] == c]
    if len(sub):
        ax.imshow(Image.open(f"{IMAGES_DIR}/{sub.iloc[0]['Filename']}").convert("RGB"))
        ax.set_title(dataset.CLASS_NAMES[c], fontsize=9)
    ax.axis("off")
plt.tight_layout(); plt.show()
''')

md("### 4.1. Kiểm tra pipeline (GUIDE.md mục 1.3): loss ban đầu ≈ ln 9, overfit 1 batch, focal γ=0, CutMix, gộp BN, temperature")
code('''ok = [self_test.test_focal_gamma0_equals_ce(), self_test.test_ls_eps0_equals_ce(),
      self_test.test_cutmix_area(), self_test.test_fuse_conv_bn(), self_test.test_temperature(),
      self_test.test_initial_loss(), self_test.test_overfit_one_batch()]
print(f"{sum(ok)}/{len(ok)} kiểm tra pipeline đạt.")
''')

md("## 5. Metadata backbone (#params, GMAC, tag trọng số)")
code('''subprocess.run([sys.executable, "code/backbone_meta.py",
    "--out", f"{RUNS}/backbone_meta.json", "--backbones",
    "resnet50", "resnext50_32x4d", "convnext_tiny",
    "deit_small_patch16_224", "efficientnet_b0", "mobilenetv3_large_100"], check=True)
''')

md("""## 6. Bước 1 — So sánh ≥ 5 backbone (cùng công thức nền, 1 seed)

Đủ ràng buộc: ResNet, ResNeXt, ConvNeXt, transformer (DeiT), mạng nhẹ (EfficientNet, MobileNetV3).
""")
code('''def run_exp(**kw):
    base = dict(images_dir=IMAGES_DIR, labels_dir=LABELS_DIR, out_dir=RUNS,
                pred_dir=PRED, curves_dir=CURVES, num_workers=2, amp=True)
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

code('''rows = []
for p in sorted(glob.glob(f"{RUNS}/B*/seed0/summary.json")):
    d = json.load(open(p))
    rows.append({k: d[k] for k in ["exp_id", "backbone", "val_macro_f1", "val_top1",
                                    "params_m", "train_time_per_epoch_s"]})
bb_table = pd.DataFrame(rows).sort_values("val_macro_f1", ascending=False)
print(bb_table.to_string(index=False))
CHOSEN_BB = bb_table.iloc[0]["backbone"]
CHOSEN_EXP = bb_table.iloc[0]["exp_id"]
print("\\n=> Chọn backbone:", CHOSEN_BB, f"(exp {CHOSEN_EXP})")
print("Ghi chu: chon theo macro-F1 val; co the chon theo can bang F1/do tre neu can thoi gian thuc.")
''')

md("Đo độ trễ sơ bộ batch-1 của từng backbone (warmup + synchronize):")
code('''lat_bb = {}
for eid, bb in BACKBONES:
    m = model.build_model(bb, pretrained=False, num_classes=9).to("cuda").eval()
    m.load_state_dict(torch.load(f"{RUNS}/{eid}/seed0/best.pt", map_location="cuda")["state_dict"])
    r = benchmark.latency_report(m, 1, 224, "fp32", "cuda", warmup=10, iters=100)
    lat_bb[eid] = r
    print(f"{eid} {bb}: p50={r['p50']:.1f}ms p95={r['p95']:.1f}ms")
    del m; torch.cuda.empty_cache()
json.dump(lat_bb, open(f"{RUNS}/backbone_latency.json", "w"))
''')

md("""## 7. Bước 2 — Công thức huấn luyện (≥ 3 trục, mỗi lần khác nền 1 yếu tố)

Trục: A khởi tạo · B augmentation · C loss · D sampler · E optimizer/LR · F EMA · G độ phân giải.
""")
code('''# T00 = công thức nền trên backbone đã chọn
run_exp(exp_id="T00", backbone=CHOSEN_BB, seed=0, epochs=ABL_EPOCHS, init="finetune", aug="basic", loss="ce")

# Mỗi thí nghiệm chỉ khác T00 đúng MỘT yếu tố (nguyên tắc N1)
run_exp(exp_id="T01", backbone=CHOSEN_BB, seed=0, epochs=ABL_EPOCHS, init="frozen",   aug="basic",   loss="ce")                     # A
run_exp(exp_id="T02", backbone=CHOSEN_BB, seed=0, epochs=ABL_EPOCHS, init="scratch",  aug="basic",   loss="ce")                     # A
run_exp(exp_id="T03", backbone=CHOSEN_BB, seed=0, epochs=ABL_EPOCHS, init="finetune", aug="color",   loss="ce")                     # B
run_exp(exp_id="T04", backbone=CHOSEN_BB, seed=0, epochs=ABL_EPOCHS, init="finetune", aug="trivial", loss="ce")                     # B
run_exp(exp_id="T05", backbone=CHOSEN_BB, seed=0, epochs=ABL_EPOCHS, init="finetune", aug="basic",   loss="ce", mix="mixup")       # B
run_exp(exp_id="T06", backbone=CHOSEN_BB, seed=0, epochs=ABL_EPOCHS, init="finetune", aug="basic",   loss="ce", mix="cutmix")      # B
run_exp(exp_id="T07", backbone=CHOSEN_BB, seed=0, epochs=ABL_EPOCHS, init="finetune", aug="basic",   loss="ls", label_smoothing=0.1) # C
run_exp(exp_id="T08", backbone=CHOSEN_BB, seed=0, epochs=ABL_EPOCHS, init="finetune", aug="basic",   loss="focal", focal_gamma=2.0)  # C
run_exp(exp_id="T09", backbone=CHOSEN_BB, seed=0, epochs=ABL_EPOCHS, init="finetune", aug="basic",   loss="ce_weighted", class_weight_beta=0.0)  # C
run_exp(exp_id="T10", backbone=CHOSEN_BB, seed=0, epochs=ABL_EPOCHS, init="finetune", aug="basic",   loss="ce", sampler="balanced") # D
run_exp(exp_id="T11", backbone=CHOSEN_BB, seed=0, epochs=ABL_EPOCHS, init="finetune", aug="basic",   loss="ce", ema_decay=0.999)    # F
''')

code('''# Kết hợp các yếu tố tốt nhất (kiểm tra cộng dồn hay triệt tiêu) — chỉnh theo kết quả trên
run_exp(exp_id="T12", backbone=CHOSEN_BB, seed=0, epochs=ABL_EPOCHS,
        init="finetune", aug="basic", loss="ls", label_smoothing=0.1,
        mix="cutmix", ema_decay=0.999)

abl = pd.DataFrame([json.load(open(p)) for p in sorted(glob.glob(f"{RUNS}/T*/seed0/summary.json"))])
abl = abl[["exp_id", "val_macro_f1", "val_top1", "init", "aug", "loss", "mix"]]
base = float(abl.loc[abl.exp_id == "T00", "val_macro_f1"].iloc[0])
abl["Δ vs T00"] = (abl["val_macro_f1"] - base).round(4)
print(abl.sort_values("val_macro_f1", ascending=False).to_string(index=False))
''')

md("## 8. Bước 3 — Phương pháp suy luận (≥ 4) + đo độ trễ p50/p95/p99")
code('''inf_rows = study.inference_study(exp_id="T00", seed=0, out_dir=RUNS, iters=100, warmup=15)
print(pd.DataFrame(inf_rows).to_string(index=False))
''')

md("## 9. Bước 4 — Chung kết: ≥ 3 seed, test đúng MỘT lần mỗi seed")
md("""Chỉnh `BEST` theo kết quả Bước 2/3 (ví dụ dưới là lựa chọn hợp lý), rồi chạy cấu hình chung kết
**và mốc** `T00` với cùng số seed, bật `save_test_predictions=True`.
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

for s in SEEDS:
    run_exp(exp_id="F01", seed=s, save_test_predictions=True, **BEST)

# Mốc: T00 + I00, cùng số seed
for s in SEEDS:
    run_exp(exp_id="T00", seed=s, backbone=CHOSEN_BB, init="finetune", aug="basic",
            loss="ce", epochs=BEST["epochs"], save_test_predictions=True)
''')

code('''# Chung kết: TTA lật ngang (K=2) + temperature scaling (T khớp trên VAL)
study.finalize_test("F01", seeds=SEEDS, method="hflip", temperature=True, out_dir=RUNS, pred_dir=PRED)
# Mốc: 1 view, không TTA, không temperature -> đúng "T00 + I00"
study.finalize_test("T00", seeds=SEEDS, method="identity", temperature=False, out_dir=RUNS, pred_dir=PRED)
print("predictions/:", sorted(os.listdir(PRED))[:12], "...")
''')

md("## 10. Sản phẩm: results.xlsx, report.md, curves/ (ghi vào repo trên Drive)")
code('''subprocess.run([sys.executable, "code/make_results.py",
    "--labels-dir", LABELS_DIR, "--out-dir", RUNS, "--pred-dir", PRED,
    "--out", f"{REPO_DIR}/results.xlsx", "--final", "F01", "--baseline", "T00"], check=True)
subprocess.run([sys.executable, "code/make_report.py",
    "--labels-dir", LABELS_DIR, "--xlsx", f"{REPO_DIR}/results.xlsx",
    "--final", "F01", "--baseline", "T00", "--out", f"{REPO_DIR}/report.md"], check=True)
print("curves/:", sorted(os.listdir(CURVES))[:8], "...")
print("results.xlsx:", os.path.exists(f"{REPO_DIR}/results.xlsx"), "| report.md:", os.path.exists(f"{REPO_DIR}/report.md"))
''')

md("## 11. Chấm bằng eval.py + tự chấm phần I")
code('''subprocess.run([sys.executable, "eval.py", "score", "--pred", f"{PRED}/F01_seed*_test.csv",
    "--test-csv", f"{LABELS_DIR}/test_subset0.csv", "--labels", f"{LABELS_DIR}/labels.csv",
    "--tag", "F01", "--out", f"{REPO_DIR}/eval_out"], check=True)
subprocess.run([sys.executable, "eval.py", "score", "--pred", f"{PRED}/T00_seed*_test.csv",
    "--test-csv", f"{LABELS_DIR}/test_subset0.csv", "--labels", f"{LABELS_DIR}/labels.csv",
    "--tag", "T00", "--out", f"{REPO_DIR}/eval_out"], check=True)
''')
code('''# p95 batch-1 của cấu hình thời gian thực (I5)
lat = pd.DataFrame(json.load(open(f"{RUNS}/latency.json")))
p95 = float(lat.iloc[0]["p95"])
subprocess.run([sys.executable, "eval.py", "grade",
    "--final", f"{PRED}/F01_seed*_test.csv", "--baseline", f"{PRED}/T00_seed*_test.csv",
    "--uncal", f"{PRED}/F01uncal_seed*_test.csv", "--final-val", f"{PRED}/F01_seed*_val.csv",
    "--latency-p95-ms", str(p95), "--latency-method", "proper",
    "--test-csv", f"{LABELS_DIR}/test_subset0.csv", "--val-csv", f"{LABELS_DIR}/val_subset0.csv",
    "--labels", f"{LABELS_DIR}/labels.csv", "--out", f"{REPO_DIR}/eval_out"], check=True)
''')

md("## 12. Ma trận nhầm lẫn + phân tích lỗi")
code('''from eval import load_group
g = load_group(f"{PRED}/F01_seed*_test.csv", f"{LABELS_DIR}/test_subset0.csv", ref_what="test")
cm = sum(m["confusion"] for m in g.metrics)
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(9)); ax.set_xticklabels(dataset.CLASS_NAMES, rotation=90, fontsize=8)
ax.set_yticks(range(9)); ax.set_yticklabels(dataset.CLASS_NAMES, fontsize=8)
ax.set_xlabel("dự đoán"); ax.set_ylabel("nhãn thật")
ax.set_title("Ma trận nhầm lẫn (test, tổng qua seed)")
for i in range(9):
    for j in range(9):
        if cm[i, j]:
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=7,
                    color="white" if cm[i, j] > cm.max()/2 else "black")
plt.colorbar(im); plt.tight_layout()
os.makedirs(CURVES, exist_ok=True)
plt.savefig(f"{CURVES}/confusion_F01.png", dpi=130); plt.show()
''')

md("## 13. Lưu sản phẩm về repo (Drive) + commit")
code('''# Copy sản phẩm từ đĩa local về repo trên Drive để commit
for src, dst in [(PRED, f"{REPO_DIR}/predictions"), (CURVES, f"{REPO_DIR}/curves")]:
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

# log nhỏ để truy vết exp_id (config/summary/history), KHÔNG copy checkpoint
LOGS = f"{REPO_DIR}/logs"
if os.path.exists(LOGS):
    shutil.rmtree(LOGS)
os.makedirs(LOGS, exist_ok=True)
for p in glob.glob(f"{RUNS}/*/seed*/*"):
    if os.path.basename(p) in ("config.json", "summary.json", "history.csv"):
        rel = os.path.relpath(p, RUNS)
        d = os.path.join(LOGS, os.path.dirname(rel))
        os.makedirs(d, exist_ok=True)
        shutil.copy(p, os.path.join(LOGS, rel))

print("Đã copy predictions/, curves/, logs/ vào repo.")
print("Bước tiếp theo (chạy trong cell dưới):")
''')
code('''# Commit & push (chạy nếu bạn có quyền ghi). KHÔNG commit dataset/checkpoint.
os.chdir(REPO_DIR)
subprocess.run(["git", "add", "code", "results.xlsx", "report.md", "curves",
                "predictions", "logs", "SUBMISSION_README.md"], check=True)
subprocess.run(["git", "-c", "user.name=Colab", "-c", "user.email=colab@local",
                "commit", "-m", "Lab Day 2: ket qua chung ket (Colab)"], check=False)
subprocess.run(["git", "push"], check=False)
''')
code('''print("Hoàn tất. Sản phẩm trong repo:")
for p in ["results.xlsx", "report.md", "curves", "predictions", "logs", "code"]:
    print(" -", p, "->", "OK" if os.path.exists(f"{REPO_DIR}/{p}") else "THIẾU")
''')
code('''# (Tuỳ chọn) Điểm thưởng: chạy nhiều fold cho cấu hình cuối -> mean ± std qua fold.
# Chỉ bật khi còn ngân sách GPU. Đặt RUN_MULTIFOLD=True.
RUN_MULTIFOLD = False
if RUN_MULTIFOLD:
    for fold in (1, 2):
        for s in SEEDS:
            run_exp(exp_id=f"F01f{fold}", seed=s, fold=fold, save_test_predictions=True, **BEST)
    print("Đã train multi-fold; tính chỉ số bằng eval.py score với --test-csv test_subset{fold}.csv tương ứng.")
''')

md("""## 14. Ghi chú cho báo cáo (bạn bổ sung nhận xét)

- **Backbone**: bảng `Backbones` — backbone nào macro-F1 val cao nhất, đánh đổi với params/GMAC/độ trễ?
- **Công thức**: bảng `Training` — yếu tố nào giúp (Δ > std), yếu tố nào không? Kết hợp có cộng dồn không?
- **Suy luận**: bảng `Inference` — TTA/ensemble tốn bao nhiêu lần độ trễ, đáng giá không? Temperature scaling giảm ECE bao nhiêu?
- **Chung kết**: `Final`/`PerClass` — so với mốc `T00`, Δ có vượt nhiễu (std) không? Hai lớp khó Chinee apple / Snake weed ra sao?
- **Hạn chế**: một fold, số seed hữu hạn, chia ngẫu nhiên không theo địa điểm nên điểm test có thể lạc quan.
""")


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
