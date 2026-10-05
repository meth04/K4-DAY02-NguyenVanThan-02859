# Bài nộp Lab Day 2 — Nguyễn Văn Thân (02859)

Kết quả thực nghiệm local: **accuracy 93,67% ± 0,30 điểm phần trăm; macro-F1 0,9139 ± 0,0044**, 3 seed trên toàn bộ test fold0. Báo cáo đã hoàn thiện từ lần train 36,18 phút; phần bổ sung chỉ validation và đo latency, không train/test lại.

- [Báo cáo](report.md), [workbook 8 sheet](results.xlsx), [biểu đồ](curves/), [dự đoán](predictions/), [minh chứng rubric](RUBRIC_EVIDENCE.md).
- [Mở notebook trên Google Colab](https://colab.research.google.com/github/meth04/K4-DAY02-NguyenVanThan-02859/blob/main/code/lab_day2.ipynb).
- Mã training thực dùng tại `9747314cb5aa60fc727fefaa089aacc3de544d15`; snapshot trong `code/`, hash core trong `evidence/runs/source_manifest.json`. Notebook hiện tải bản sửa renderer `cd6b0eb2aafe44b30cead7eb8fb20a0885c30c95`; không thay core training.

## Kiểm tra điểm từ CSV, không cần GPU hoặc checkpoint

Từ thư mục bài nộp này, cài `numpy` và `pandas` rồi chạy:

```powershell
python eval.py score --pred 'predictions/F01_seed*_test.csv' --test-csv evidence/labels/test_subset0.csv --labels evidence/labels/labels.csv --tag F01 --out verification
python eval.py grade --final 'predictions/F01_seed*_test.csv' --baseline 'predictions/T00_seed*_test.csv' --uncal 'predictions/F01uncal_seed*_test.csv' --final-val 'predictions/F01_seed*_val.csv' --test-csv evidence/labels/test_subset0.csv --val-csv evidence/labels/val_subset0.csv --labels evidence/labels/labels.csv --latency-p95-ms 26.6811200039 --out verification
```

`eval.py` nguyên bản, hash kiểm tra trong manifest. `grade` hiện cho **10/20 riêng mục I**, không tự chấm A–H hoặc điểm thưởng. `evidence/labels/` là bốn CSV metadata gốc để kiểm tra fold offline; không có ảnh dataset. Prediction test, val và uncal đều được giữ lại; bản uncal là cùng lượt dự đoán chưa temperature scaling.

## Chạy lại training

Từ root repository, xem [hướng dẫn local](https://github.com/meth04/K4-DAY02-NguyenVanThan-02859/blob/main/SUBMISSION_README.md). Môi trường đo: Windows11, Python3.13.13, torch2.7.1+cu126, torchvision0.22.1+cu126, timm1.0.30, GTX1650 4GB. `requirements-local.txt` ghi phiên bản runtime còn lại.

```powershell
python -m venv --system-site-packages .venv-gpu
.\.venv-gpu\Scripts\python.exe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
.\.venv-gpu\Scripts\python.exe -m pip install -r requirements.txt
.\.venv-gpu\Scripts\python.exe -X utf8 -u code/run_local.py --root local_runs/reproduction --minutes 60
```

Dùng thư mục mới cho nghiên cứu mới. CLI probe GPU và chọn batch/precision theo tốc độ, epoch chung kết theo ngân sách; có thể khác cấu hình bản nộp trên phần cứng khác. Để tái lập cấu hình đã nộp, dùng settings trong `evidence/environment.json` và config từng seed: 128px, batch64, FP32/NCHW, workers2, cudnn benchmark off; khảo sát1epoch, final12epoch, seed0/1/2. Test một lần sau khi chốt trên val; không sửa cấu hình dựa vào test.

Colab: chọn GPU, PROFILE=hour (128px), SAVE_TO_DRIVE=True, chạy cell theo thứ tự. Notebook hour mặc định final2epoch; thêm `lab.budget["final"] = 12` ngay sau khi tạo `lab`, trước phần backbone, nếu cần tái lập số epoch local. Để giữ precision/batch giống bản nộp, cập nhật `settings` trước khi tạo lab: `batch_size=64, amp=False, channels_last=False, num_workers=2, cudnn_benchmark=False`. Có thể cần giảm batch theo VRAM GPU được cấp, khi đó ghi rõ khác biệt. Giữ SESSION riêng cho mỗi nghiên cứu để không trộn nguồn/checkpoint. Thời gian và kết quả không thể bảo đảm giống trên phần cứng khác.

## Tái lập minh chứng bổ sung và đóng gói

Sau khi có checkpoint/dataset local cùng nghiên cứu:

```powershell
.\.venv-gpu\Scripts\python.exe -X utf8 -u code/submission_extras.py --root local_runs/hour
.\.venv-gpu\Scripts\python.exe -X utf8 code/finalize_submission.py --root local_runs/hour
```

Extras chỉ đo GPU latency và validation (không train/test). Packaging chỉ đọc CSV/log đã có; không gọi model. Dataset images.zip/ảnh/checkpoint **không commit** theo README lớp. Các checkpoint hiện nằm ở `local_runs/hour/runs/` trên máy chủ bài nộp; không có link checkpoint công khai. Người chấm kiểm tra toàn bộ metric test trực tiếp từ CSV, hoặc train lại bằng code/notebook.

Các phép đo bổ sung: 20 warmup/100 lượt/synchronize, batch1 FP32, gồm forward+aggregate+softmax; không gồm camera/decode/resize/H2D. Kết quả mới và forward-only gốc đều có evidence, tránh đánh đồng latency model với end-to-end robot.
