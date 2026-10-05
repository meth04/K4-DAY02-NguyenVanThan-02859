"""Build a self-contained Colab notebook from the current source snapshot."""
from __future__ import annotations

import base64
import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def build():
    files = {str(p.relative_to(ROOT)).replace("\\", "/"): p.read_text(encoding="utf-8")
             for p in sorted((ROOT / "code").glob("*.py"))}
    for name in ("eval.py", "README.md", "GUIDE.md", "RUBRIC.md", "SUBMISSION_README.md", "LAB_STATUS.md", "requirements.txt"):
        files[name] = (ROOT / name).read_text(encoding="utf-8")
    payload = base64.b64encode(gzip.compress(json.dumps(files, ensure_ascii=False).encode("utf-8"), mtime=0)).decode("ascii")
    cells = []

    def cell(kind, source):
        value = {"cell_type": kind, "metadata": {}, "source": source.strip() + "\n"}
        if kind == "code":
            value.update(execution_count=None, outputs=[])
        cells.append(value)

    cell("markdown", """# DeepWeeds — Lab Day 2, Colab ưu tiên tốc độ

**Cách chạy:** tải notebook này lên [Google Colab](https://colab.research.google.com/),
chọn **Runtime → Change runtime type → GPU**, rồi **Run all**.
Notebook chứa toàn bộ code hiện tại; không cần clone repo hay push GitHub trước.

Lab cần: fold 0 nguyên bản; ≥5 backbone; ≥3 trục huấn luyện; ≥4 phương pháp suy luận
ngoài mốc; chung kết và mốc ≥3 seed; chọn mọi thứ trên val; test chỉ đánh giá cuối cùng.
Sản phẩm: `results.xlsx`, `report.md` bản nháp, `curves/`, `predictions/`, `logs/`, `eval_out/`.

| PROFILE | Khảo sát / chung kết | Mục đích |
|---|---|---|
| `fast` (mặc định) | 3 / 10 epoch | Ít thí nghiệm nhất trong kế hoạch này, đủ số nhóm yêu cầu; chất lượng cần kiểm chứng |
| `full` | 12 / 15 epoch | Nhiều epoch và thêm scratch/CutMix/focal, mất thời gian hơn |
| `smoke` | 1 epoch, tập con train/val | Kiểm tra pipeline; không chạy test, không dùng nộp |

Tăng tốc: pretrained, AMP FP16 trên T4/BF16 trên GPU hỗ trợ, channels-last, fused AdamW,
DataLoader có prefetch/persistent workers, ảnh trên `/content`, dùng lại cấu hình nền khảo sát,
bỏ qua thí nghiệm đã hoàn tất. Không bật `torch.compile` mặc định vì chi phí compile nhiều
backbone với vòng chạy ngắn có thể làm tổng thời gian tăng.
Tham khảo [PyTorch tuning guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html).

Không cam kết số phút hay điểm model khi chưa đo trên GPU được cấp. Sàng lọc 3 epoch có thể
xếp hạng khác khi train lâu. Mọi profile giữ đầy đủ tập test khi chạy chung kết.
""")
    cell("markdown", "## 1. Cấu hình — thường chỉ cần giữ mặc định")
    cell("code", '''PROFILE = "fast"              # fast | full | smoke
SAVE_TO_DRIVE = True           # Lưu thí nghiệm đã xong + best checkpoint; có bước xác thực Drive
SESSION = "deepweeds_day2_v1"   # Đổi tên nếu muốn bắt đầu một nghiên cứu mới
ROOT = f"/content/{SESSION}_{PROFILE}"
''')
    cell("markdown", "## 2. Cài thư viện và kiểm tra GPU")
    cell("code", '''import sys, os, json, platform, subprocess, time
from pathlib import Path
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "timm>=1.0,<1.1", "openpyxl", "scikit-learn"], check=True)
import torch, torchvision, timm, numpy as np, pandas as pd
assert torch.cuda.is_available(), "Chọn Runtime > Change runtime type > GPU trước khi train."
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
print("GPU:", torch.cuda.get_device_name(0))
print("torch:", torch.__version__, "torchvision:", torchvision.__version__, "timm:", timm.__version__)
print("Profile:", PROFILE, "| RAM workers:", min(4, os.cpu_count() or 2))
''')
    cell("markdown", "## 3. Bung code có sẵn trong notebook + lưu Drive")
    cell("code", f'''import base64, gzip
SOURCE_BUNDLE = {payload!r}
snapshot = json.loads(gzip.decompress(base64.b64decode(SOURCE_BUNDLE)))
root = Path(ROOT)
root.mkdir(parents=True, exist_ok=True)
for relative, source in snapshot.items():
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source, encoding="utf-8")
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "code"))
BACKUP = None
if SAVE_TO_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    BACKUP = f"/content/drive/MyDrive/{{SESSION}}"
from colab_fast import FastLab, write_json
lab = FastLab(ROOT, PROFILE, backup=BACKUP)
environment = dict(python=platform.python_version(), torch=torch.__version__,
                   torchvision=torchvision.__version__, timm=timm.__version__,
                   numpy=np.__version__, pandas=pd.__version__,
                   gpu=torch.cuda.get_device_name(0), profile=PROFILE, amp_dtype=lab.dtype,
                   snapshot_sha256=__import__("hashlib").sha256(gzip.decompress(base64.b64decode(SOURCE_BUNDLE))).hexdigest())
write_json(root / "environment.json", environment)
STARTED = time.perf_counter()
print("Đọc ảnh / train:", root, "| Backup:", lab.backup)
''')
    cell("markdown", """## 4. Tải dữ liệu, checksum, kiểm tra split và EDA

Ảnh đã tải được cache trên Drive; mỗi runtime đọc ảnh từ đĩa local.
Giữ nguyên CSV fold 0, kiểm tra giao rỗng, hợp 17.509 ảnh và nhãn đúng `labels.csv`.
""")
    cell("code", '''(train_df, val_df, test_df), split_info = lab.prepare_data()
import matplotlib.pyplot as plt
from PIL import Image
import dataset
counts = pd.DataFrame({s: pd.Series(split_info["per_class"][s]).sort_index()
                       for s in ("train", "val", "test")}).fillna(0)
counts.index = dataset.CLASS_NAMES
counts.plot.bar(figsize=(12, 4), title="DeepWeeds fold 0: phân bố lớp")
plt.ylabel("Số ảnh"); plt.tight_layout()
plt.savefig(lab.curves / "eda_classes.png"); plt.show()
display(counts.astype(int))
fig, axes = plt.subplots(3, 3, figsize=(9, 9))
for label, ax in enumerate(axes.ravel()):
    name = train_df[train_df.Label == label].iloc[0].Filename
    with Image.open(lab.images / name) as image:
        ax.imshow(image.convert("RGB"))
    ax.set_title(dataset.CLASS_NAMES[label]); ax.axis("off")
plt.tight_layout(); plt.savefig(lab.curves / "eda_samples.png"); plt.show()
print("Negative chiếm", round(100 * counts.loc["Negatives"].sum() / counts.values.sum(), 1), "% tổng ảnh.")
''')
    cell("markdown", "## 5. Kiểm tra loss, augmentation và overfit một batch trên GPU")
    cell("code", '''import self_test, model, train
train.set_seed(0)
checks = [self_test.test_focal_gamma0_equals_ce(), self_test.test_ls_eps0_equals_ce(),
          self_test.test_cutmix_area(), self_test.test_mixup_label(),
          self_test.test_fuse_conv_bn(), self_test.test_temperature()]
assert all(checks), "Dừng: có kiểm tra pipeline chưa đạt."
train.set_seed(0)
tf = dataset.build_transforms(True, 224, "basic")
loader = dataset.make_loader(train_df.iloc[:16], lab.images, tf, 8, True, num_workers=0)
x, y, _ = next(iter(loader))
fig, axes = plt.subplots(1, 4, figsize=(12, 3))
for image, ax in zip(x[:4], axes):
    visible = image.permute(1, 2, 0).numpy() * np.array(dataset.IMAGENET_STD) + np.array(dataset.IMAGENET_MEAN)
    ax.imshow(visible.clip(0, 1)); ax.axis("off")
plt.tight_layout(); plt.savefig(lab.curves / "augmentation.png"); plt.show()
m = model.build_model("resnet18", pretrained=False).cuda()
x, y = x.cuda(), y.cuda()
m.train()
with torch.no_grad():
    initial = torch.nn.functional.cross_entropy(m(x), y).item()
print("Initial loss:", initial, "| ln(9):", np.log(9))
assert abs(initial - np.log(9)) < 1.0
m.train(); optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)
first = None
for step in range(60):
    optimizer.zero_grad(set_to_none=True)
    loss = torch.nn.functional.cross_entropy(m(x), y)
    loss.backward(); optimizer.step()
    if first is None: first = loss.item()
    if step >= 15 and loss.item() < 0.05: break
print("Overfit 1 batch:", first, "->", loss.item(), "| steps:", step + 1)
assert loss.item() < 0.05, "Batch chưa overfit; kiểm tra dữ liệu/loss trước khi chạy tiếp."
write_json(lab.runs / "pipeline_checks.json", dict(initial_loss=initial, overfit_loss=loss.item(), steps=step+1, checks=checks))
del m, x, y, loader, optimizer
torch.cuda.empty_cache()
lab.persist()
''')
    cell("markdown", """## 6. So sánh 5 backbone trên toàn bộ train/val

Cùng seed 0, 224 px, batch 64, pretrained + fine-tune, CE, basic augmentation,
AdamW và số epoch. ResNet18/50, ConvNeXt, DeiT tiny, MobileNetV3 đáp ứng nhóm kiến trúc.
Mỗi epoch in thời gian; dùng số này để ước lượng thời gian còn lại.
""")
    cell("code", '''backbone_table = lab.backbones()
display(backbone_table[["exp_id", "backbone", "val_macro_f1", "val_top1", "train_time_per_epoch_s"]])
print("Chọn trên val:", lab.chosen["backbone"])
print("Thời gian đã chạy (phút):", round((time.perf_counter() - STARTED) / 60, 1))
''')
    cell("markdown", """## 7. Ba trục huấn luyện và một kết hợp

`TS00` dùng lại model backbone thắng với cùng cấu hình, không train thêm.
`T01`: frozen vs fine-tune; `T03`: color vs basic; `T07`: label smoothing vs CE.
`T14`: color + label smoothing để kiểm tra kết hợp. Mỗi thử nghiệm đơn chỉ thay một yếu tố.
`full` thêm scratch, CutMix, focal. Chọn công thức theo macro-F1 val, không xem test.
`T00` ở chung kết sẽ có ngân sách epoch bằng `F01`; bảng ablation so với `TS00`.
""")
    cell("code", '''ablation_table = lab.ablations()
display(ablation_table[["exp_id", "init", "aug", "loss", "val_macro_f1", "delta_vs_TS00"]])
print("Công thức thắng val:", {k: lab.best[k] for k in ("backbone", "init", "aug", "loss", "mix")})
''')
    cell("markdown", """## 8. Train chung kết seed 0 và chọn suy luận trên val

Thử ngoài mốc: hflip gộp xác suất, hflip gộp logit, temperature scaling, ensemble 2 backbone.
Ensemble dùng dự đoán val có sẵn; không đo latency ensemble và không đưa vào lựa chọn triển khai tự động.
Chọn 1-view hoặc hflip dựa trên val; ưu tiên 1-view nếu chênh lệch F1 < 0,001.
Độ trễ model: warmup 10, 50 lần đo, synchronize; batch 1, cùng AMP dtype train.
Chi phí decode/resize/hiệu chuẩn chưa nằm trong latency model. TTA là một lần đánh giá test
theo phương pháp đã chốt, với hai lượt forward/view.
""")
    cell("code", '''lab.train_final()
inference_table = lab.inference_study()
display(inference_table)
if PROFILE != "smoke":
    print("Phương pháp đã chốt:", lab.method)
''')
    cell("markdown", """## 9. Chung kết và mốc: ba seed, test cuối cùng

Train các seed còn lại trước khi đọc kết quả test. Giữ toàn bộ train/val/test fold 0.
T được fit trên val riêng từng seed. Dự đoán có/không T dùng cùng logits, không chạy test lại.
Mỗi lần đánh giá có manifest; chạy lại notebook sẽ dùng file đã hoàn tất.
Drive lưu sau mỗi thí nghiệm. Nếu bị ngắt giữa một lần train, thí nghiệm đang dở sẽ train lại;
các thí nghiệm đã xong được dùng lại. Nếu bị ngắt giữa test, notebook dừng để bạn kiểm tra
manifest/dự đoán, tránh tự chạy test lần nữa.
""")
    cell("code", '''lab.finals()
''')
    cell("markdown", "## 10. Đánh giá, workbook 7 sheet, báo cáo nháp và gói tải về")
    cell("code", '''archive = lab.products()
if PROFILE != "smoke":
    from eval import load_group
    group = load_group(str(lab.pred / "F01_seed*_test.csv"), str(lab.labels / "test_subset0.csv"), ref_what="test")
    cm = sum(metrics["confusion"] for metrics in group.metrics)
    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(9), dataset.CLASS_NAMES, rotation=60, ha="right")
    ax.set_yticks(range(9), dataset.CLASS_NAMES)
    ax.set_xlabel("Dự đoán"); ax.set_ylabel("Nhãn thật")
    fig.colorbar(image, ax=ax); plt.tight_layout()
    plt.savefig(lab.curves / "F01_confusion.png"); plt.show()
    errors = pd.read_csv(lab.pred / "F01_seed0_test.csv")
    errors = errors[errors.y_true != errors.y_pred]
    hard = errors[errors.y_true.isin([0, 7])]
    samples = pd.concat([hard, errors]).drop_duplicates("Filename").head(8)
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for ax in axes.ravel(): ax.axis("off")
    for (_, row), ax in zip(samples.iterrows(), axes.ravel()):
        with Image.open(lab.images / row.Filename) as im: ax.imshow(im.convert("RGB"))
        ax.set_title(f"Thật: {dataset.CLASS_NAMES[int(row.y_true)]}\\nĐoán: {dataset.CLASS_NAMES[int(row.y_pred)]}", fontsize=9)
    plt.tight_layout(); plt.savefig(lab.curves / "F01_errors.png"); plt.show()
    # Refresh archive to include the error-analysis figures.
    from zipfile import ZipFile
    with ZipFile(archive, "a") as out:
        for name in ("F01_confusion.png", "F01_errors.png"):
            out.write(lab.curves / name, f"curves/{name}")
    lab.persist()
    if lab.backup:
        __import__("shutil").copy2(archive, lab.backup / archive.name)
    print("Tổng thời gian (phút):", round((time.perf_counter() - STARTED) / 60, 1))
    print("Gói bài nộp:", archive)
    print("Đọc report.md và bổ sung kết luận, đánh đổi tốc độ/F1, giả thuyết lỗi trước khi nộp.")
''')
    cell("markdown", """## 11. Tải sản phẩm

Gói ZIP không chứa dataset hay checkpoint. Checkpoint để trong thư mục backup Drive.
Notebook không tự commit/push. Lưu notebook có output sau chạy bằng **File → Download → .ipynb**
để làm bằng chứng EDA, pipeline và quá trình huấn luyện.
""")
    cell("code", '''if PROFILE != "smoke":
    from google.colab import files
    files.download(str(archive))
''')
    notebook = dict(nbformat=4, nbformat_minor=5,
                    metadata=dict(colab=dict(name="lab_day2.ipynb"), accelerator="GPU",
                                  kernelspec=dict(name="python3", display_name="Python 3"),
                                  language_info=dict(name="python", version="3")), cells=cells)
    for i, item in enumerate(cells):
        item["id"] = f"deepweeds-{i:02d}"
    path = ROOT / "code" / "lab_day2.ipynb"
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Built {path.name}: {len(cells)} cells, {path.stat().st_size:,} bytes")
    return path


if __name__ == "__main__":
    build()
