# K4-DAY02 — Nguyễn Văn Thân (02859)

Lab Day 2: so sánh backbone, công thức huấn luyện và suy luận trên DeepWeeds.

## Bài nộp hoàn thiện

[Bài nộp Nguyễn Văn Thân — 02859](submissions/02859_nguyen_van_than/README.md)
có workbook, báo cáo đã phân tích, curves, dự đoán từng seed, config/log và metadata fold gốc.
Đã thêm đo độ trễ đầy đủ các phương pháp, Grad-CAM và đánh giá tối/mờ/nhiễu trên validation.
Dataset ảnh và checkpoint giữ local theo yêu cầu lớp; các minh chứng nhỏ được commit.

Kiểm tra offline: `python code/verify_submission.py` (numpy, pandas, openpyxl).
`eval.py` gốc tính lại phần I: 10/20; A–H và điểm thưởng do giảng viên chấm.
Báo cáo mới ở `submissions/02859_nguyen_van_than/report.md`; bản trong `local_runs/hour/report.md`
là bản nháp tự sinh ban đầu. Gói hoàn thiện local: `local_runs/hour/deepweeds_submission_final.zip`.

## Chạy trên Colab

Dùng **[code/lab_day2.ipynb](code/lab_day2.ipynb)**.
Notebook tự tải code và eval.py từ GitHub theo một commit cố định, dùng các cell ngắn,
đọc được. Mã huấn luyện nằm trong thư mục `code/`.

1. Upload notebook lên https://colab.research.google.com/ qua File → Upload notebook.
2. Runtime → Change runtime type → GPU (T4 hoặc GPU tốt hơn được cấp).
3. Giữ PROFILE="fast", SAVE_TO_DRIVE=True, Run all và xác thực Drive.
4. Cuối notebook tải deepweeds_submission.zip. Backup nằm trong
   MyDrive/deepweeds_day2_v1/fast/. Ảnh train đọc từ /content.
5. Đọc report.md bản nháp, bổ sung kết luận/giả thuyết lỗi từ số liệu thật;
   tải thêm notebook có output để lưu bằng chứng thực nghiệm.

| Profile | Epoch khảo sát / chung kết | Mục đích |
|---|---|---|
| hour | 1 / 2 mặc định | 128 px, ResNet18/34, ConvNeXt atto, DeiT tiny, MobileNetV3; ngân sách ngắn, cần đo thời gian thật |
| fast | 3 / 10 | Khảo sát ngắn, đủ nhóm thí nghiệm của kế hoạch; chất lượng cần kiểm chứng |
| full | 12 / 15 | Thêm scratch/CutMix/focal, tăng ngân sách |
| smoke | 1, tập con train/val | Kiểm tra pipeline; không test, không nộp |

Notebook `code/lab_day2.ipynb` đã thay bản cũ bằng bản mới ưu tiên tốc độ.
Nếu không dùng Drive, đặt SAVE_TO_DRIVE=False; /content mất khi runtime bị xoá.
Nếu bắt đầu nghiên cứu với code/cấu hình mới, đổi SESSION để tránh trộn kết quả.

## Chạy trên máy local trong ngân sách khoảng một giờ

Máy đã kiểm tra: Windows 11, GTX 1650 4 GB, Ryzen 5 6600HS, RAM 16 GB.
Môi trường CUDA riêng nằm trong `.venv-gpu/`; Python hệ thống không bị thay đổi.
[PyTorch cung cấp cặp torch 2.7.1 / torchvision 0.22.1 cho CUDA 12.6](https://pytorch.org/get-started/previous-versions/).
Tạo môi trường từ Python đã có các thư viện trong requirements.txt:

```powershell
python -m venv --system-site-packages .venv-gpu
.\.venv-gpu\Scripts\python.exe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
.\.venv-gpu\Scripts\python.exe -X utf8 -u code/run_local.py --minutes 60
```

`run_local.py` thử forward/backward/AdamW thật trên cả 5 kiến trúc, so tốc độ
FP32/FP16 và channels-last, rồi chọn cùng một batch/precision cho các backbone.
Ảnh 128 px; ResNet18/34, ConvNeXt atto, DeiT tiny và MobileNetV3; khảo sát 1 epoch.
Sau khảo sát/ablation, ngân sách chung kết (1–12 epoch) dựa thời gian đo được,
giữ cùng số epoch cho F01/T00 và đủ 3 seed. Không cắt, lọc hoặc chia lại tập dữ liệu.
Test vẫn chỉ đọc sau khi chốt công thức trên val. Ngân sách ngắn có thể giảm điểm;
không đảm bảo hoàn tất chính xác 60 phút khi GPU nóng, máy bận hoặc mạng tải chậm.
Lần cài CUDA/tải pretrained đầu tiên có thể tốn thêm thời gian.

Log và sản phẩm ở `local_runs/hour/`: `training.log`, `local_plan.json`,
`cuda_probe.json`, `environment.json`, `results.xlsx`, `report.md`,
`deepweeds_submission.zip`. Mọi thư mục môi trường/dataset/checkpoint được bỏ qua khi commit.
Chạy lại lệnh trên dùng lại thí nghiệm hoàn tất. Khi đổi code/cấu hình, dùng `--root`
với thư mục mới để giữ kết quả cũ. Có thể chỉ đo GPU với `--probe-only`.

## Thí nghiệm và tăng tốc

- 5 backbone: ResNet18/50, ConvNeXt tiny, DeiT tiny, MobileNetV3 large.
  Cùng fold 0, seed 0, độ phân giải, batch và công thức nền.
  Notebook chọn batch 16 dưới 6 GB VRAM, 32 khi đủ bộ nhớ; tắt cuDNN benchmark
  và dùng API cuDNN cũ để tránh lỗi chọn engine trên GPU nhỏ/cũ.
- 3 trục: fine-tune/frozen; basic/color; CE/label smoothing.
  Thêm kết hợp color + label smoothing; full thêm scratch/CutMix/focal.
- 4 cách suy luận ngoài mốc: hflip gộp xác suất/logit, temperature scaling,
  ensemble 2 backbone. Ensemble chỉ báo val, chưa đo latency, không chọn tự động.
  Triển khai tự động chọn 1-view/hflip trên val, rồi hiệu chuẩn T từng seed.
- Chung kết F01 và mốc T00 cùng epoch, 3 seed 0/1/2. Test cuối cùng, toàn bộ tập.
- AMP FP16 trên T4/BF16 khi hỗ trợ, channels-last, fused AdamW,
  DataLoader prefetch/persistent workers; ảnh/checkpoint trên đĩa local.
- TS00 dùng lại backbone thắng đúng cấu hình để tránh train nền khảo sát thêm.
  Nếu công thức chung kết giống nền, dùng lại trọng số mốc cùng seed.
- Drive lưu sau mỗi thí nghiệm. Chạy lại dùng lại thí nghiệm đã hoàn tất;
  lần train đang dở phải bắt đầu lại, không resume optimizer giữa epoch.
- Manifest bảo vệ test: dùng lại predictions đã xong; nếu ngắt giữa test,
  dừng để kiểm tra manifest/file đã lưu, tránh tự đánh giá test lần nữa.

Không cam kết thời gian/điểm model khi chưa chạy Colab GPU. Khảo sát 3 epoch
có thể xếp hạng khác train lâu. Ablation so với TS00 cùng ngân sách;
T00 là mốc chung kết cùng ngân sách F01.

## Sản phẩm và tái lập

ZIP có results.xlsx (7 sheet), report.md, curves/, predictions/, logs/, eval_out/,
selection.json, environment.json, code/ và eval.py nguyên bản.
Không chứa dataset/checkpoint. Không đưa dataset/checkpoint vào Git.
GMAC đếm Conv2d + Linear, xét grouped conv/số token, chưa tính attention matmul.
Latency model: batch 1, warmup 10, 50 lần, synchronize, cùng AMP dtype suy luận;
chưa gồm decode/resize/hiệu chuẩn; latency ensemble chưa đo.

Checkpoint chọn bằng macro-F1 val. Nhiệt độ T khớp val từng seed.
Dữ liệu gốc có một nhãn lệch: `20170714-110407-3.jpg` là Label 0 trong train fold 0,
nhưng Label 1 trong labels.csv. Giữ nguyên CSV fold theo S1, ghi chênh lệch vào
`logs/split_check.json` và báo cáo; các lỗi nhãn khác vẫn làm kiểm tra dừng.
Dự đoán có/không T dùng cùng logits, không chạy model trên test thêm.
Workbook/báo cáo chỉ sinh sau chạy thật; chưa có số liệu là chưa xong thực nghiệm.

Môi trường: Python 3, PyTorch 2.x/CUDA và torchvision Colab có sẵn,
timm==1.0.30, numpy, pandas, Pillow, scikit-learn, matplotlib, openpyxl.
Version/GPU/dtype/commit code thực tế lưu trong environment.json.

Sinh lại notebook sau khi sửa code:

```bash
python code/build_notebook.py
```

Kiểm tra offline:

```bash
python -m unittest discover -s tests
python code/verify_colab_fast.py
```

Kiểm tra mới dùng ảnh giả trên CPU: train/checkpoint, calibration,
CSV chuẩn eval.py, frozen BN, dùng lại lần chạy và không lặp test.
Hàm ghi JSON xử lý scalar NumPy (bao gồm `numpy.bool`), array, tensor và Path;
kiểm tra pipeline trả về bool Python để lưu kết quả an toàn.

Kiểm tra trực tiếp các cell notebook với ảnh DeepWeeds gốc:

```bash
python code/verify_notebook.py --labels-dir _smoke/deepweeds_labels
```

Đặt `images.zip` gốc ở thư mục repo và bốn CSV gốc (`labels.csv`,
`train_subset0.csv`, `val_subset0.csv`, `test_subset0.csv`) trong thư mục chỉ định.
Kiểm tra này chạy chuẩn bị/EDA trên đủ 17.509 ảnh, cell 5, cả 5 kiến trúc,
ablation, suy luận, chung kết 3 seed, eval.py, workbook 7 sheet và xuất ZIP hai lần.
Các lượt train kiểm tra dùng CPU, trọng số ngẫu nhiên, 18/9/9 ảnh và 1 epoch
trong thư mục tạm; không thay đổi dữ liệu gốc. Chưa thay thế đo hiệu năng/chất lượng,
AMP trên Colab GPU, tải trọng số pretrained và giao diện Drive.

Xem tiến độ và sản phẩm hiện tại trong [LAB_STATUS.md](LAB_STATUS.md).
README/GUIDE/RUBRIC, eval.py và starter/tests gốc của lớp được giữ nguyên.
