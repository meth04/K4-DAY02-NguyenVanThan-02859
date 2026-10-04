# K4-DAY02 — Nguyễn Văn Thân (02859)

Bài nộp **Lab Day 2 — Backbone, công thức huấn luyện và suy luận trên DeepWeeds** (Track 4).

## Chạy lại (Google Colab, GPU T4 trở lên)

Notebook chạy end-to-end: **tự động clone repo → tải dataset → train → sinh sản phẩm**.

1. Mở [`code/lab_day2.ipynb`](code/lab_day2.ipynb) trên Colab:
   `https://colab.research.google.com/github/meth04/K4-DAY02-NguyenVanThan-02859/blob/main/code/lab_day2.ipynb`
2. `Runtime → Change runtime type → GPU (T4)`.
3. `Runtime → Run all`. Notebook tự mount Drive (lưu bền), clone repo, tải `images.zip`
   (~490 MB, kiểm MD5), giải nén, train và sinh:
   `results.xlsx`, `report.md`, `curves/`, `predictions/`, `eval_out/`.
4. Commit kết quả (không commit dataset/checkpoint):
   ```bash
   git add code results.xlsx report.md curves predictions SUBMISSION_README.md
   git commit -m "Lab Day 2: ket qua chung ket"
   git push
   ```

Chế độ chạy: biến `QUICK = True` (bản rút gọn, đủ ngưỡng điểm) / `False` (đầy đủ). Chỉnh trong cell
"Mount Google Drive + cấu hình".

## Thư viện

Xem [`requirements.txt`](requirements.txt). Colab đã có `torch`/`torchvision`; notebook cài thêm
`timm`, `openpyxl`, `scikit-learn`. Phiên bản đã dùng khi phát triển:

- Python 3.11, torch 2.x (CUDA), torchvision, timm ≥ 1.0, numpy, pandas, Pillow,
  scikit-learn, matplotlib, openpyxl.

## Cấu trúc bài nộp

```
.
├── SUBMISSION_README.md      # file này
├── results.xlsx              # 7 sheet: Backbones/Training/Inference/Final/PerClass/Latency/Summary
├── report.md                 # báo cáo kết luận (GUIDE.md mục 6.3)
├── curves/                   # mỗi exp_id một ảnh (loss/metric theo epoch)
├── predictions/              # <exp_id>_seed<k>_{test,val}.csv cho chung kết (F01) và mốc (T00)
├── code/                     # toàn bộ code (bộ khung starter/ đã hoàn thiện)
│   ├── lab_day2.ipynb        # notebook Colab chạy end-to-end
│   ├── dataset.py model.py losses.py train.py inference.py benchmark.py study.py
│   ├── self_test.py          # kiểm tra pipeline (loss ban đầu, overfit 1 batch, focal γ=0, CutMix, gộp BN, temperature)
│   ├── backbone_meta.py      # #params, GMAC, tag trọng số
│   ├── make_results.py       # sinh results.xlsx
│   ├── make_report.py        # sinh report.md
│   └── build_notebook.py     # sinh lại notebook
├── eval.py                   # công cụ chấm của lớp — KHÔNG sửa
├── tests/                    # test của repo
└── requirements.txt
```

`eval.py`, `GUIDE.md`, `README.md`, `RUBRIC.md`, `starter/`, `tests/` là file gốc của lớp, giữ nguyên.

## Lệnh chạy tay (không dùng notebook)

```bash
# 1) kiểm tra pipeline
python code/self_test.py

# 2) train một cấu hình
python code/train.py --set exp_id=T00 backbone=resnet50 seed=0 epochs=10

# 3) suy luận (Bước 3)
python code/study.py --study T00 --seed 0

# 4) chung kết: train 3 seed với save_test_predictions=True, rồi
python code/study.py --finalize F01 --seeds 0 1 2 --method hflip

# 5) sản phẩm
python code/make_results.py --labels-dir data/labels --out results.xlsx
python code/make_report.py  --labels-dir data/labels

# 6) chấm
python eval.py score --pred "predictions/F01_seed*_test.csv" \
    --test-csv data/labels/test_subset0.csv --labels data/labels/labels.csv --tag F01 --out eval_out
python eval.py grade --final "predictions/F01_seed*_test.csv" \
    --baseline "predictions/T00_seed*_test.csv" \
    --uncal "predictions/F01uncal_seed*_test.csv" --final-val "predictions/F01_seed*_val.csv" \
    --test-csv data/labels/test_subset0.csv --val-csv data/labels/val_subset0.csv \
    --labels data/labels/labels.csv --out eval_out
```

## Seed

Vòng chung kết dùng `seed ∈ {0, 1, 2}` (≥ 3 seed). Seed chỉ đổi khởi tạo head, thứ tự batch và
augmentation; **không** đổi cách chia dữ liệu (S5). Dùng đúng fold 0 chia sẵn (S1).

## Quy tắc đã tuân thủ

- Chọn mọi thứ (backbone, siêu tham số, phương pháp suy luận, nhiệt độ T) trên **val**.
- **Test chạy đúng một lần mỗi seed** ở Bước 4; không dùng thông tin test để quyết định.
- Không gộp val vào train, không huấn luyện trên test.
- Mọi số liệu trong `results.xlsx`/`report.md` sinh từ log chạy thật và khớp `eval.py`.
