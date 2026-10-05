# Tiến độ Lab Day 2 (2026-10-05)

## Bài lab cần làm

1. DeepWeeds: 17.509 ảnh, 9 lớp; dùng CSV fold 0 nguyên bản.
   Kiểm tra số ảnh/lớp, giao train/val/test rỗng, hợp đủ ảnh; làm EDA.
2. So sánh ≥5 backbone, có ResNet, ResNeXt hoặc ConvNeXt, transformer, mạng nhẹ.
   Cùng công thức nền, seed, split và ngân sách.
3. Thử ≥3 trục huấn luyện, mỗi trục ≥2 giá trị; có loss/augmentation.
   Thử nghiệm đơn chỉ khác nền một yếu tố; thêm một kết hợp.
4. So sánh ≥4 cách suy luận ngoài mốc 1-view; temperature khớp trên val,
   ECE trước/sau; latency warmup/synchronize, ≥50 lần, p50/p95/p99.
5. Chốt trên val; chung kết và mốc ≥3 seed, test cuối cùng trên toàn bộ tập test.
   Mean ± std mẫu, chỉ số hai lớp khó Chinee Apple/Snake Weed.
6. Nộp code/notebook, workbook 7 sheet, báo cáo, curves/logs/predictions đúng `eval.py`.
   Có phân tích lỗi/ma trận nhầm lẫn và hạn chế.

Nguồn: [README.md](README.md), [GUIDE.md](GUIDE.md), [RUBRIC.md](RUBRIC.md).

## Đã có và còn thiếu

- Đã triển khai module dataset/model/losses/train/inference/benchmark, study,
  self-test, tạo workbook/báo cáo và notebook trong `code/`.
- `images.zip`: 491.516.047 byte, 17.509 ảnh, MD5 đúng
  `b7b30f96d466fba86016aa5a26606e0f`.
- Kiểm tra ban đầu: notebook cũ chưa có output; chưa có labels, log train,
  checkpoint, predictions, curves, `results.xlsx` hoặc `report.md` trong workspace.
  Nếu đã có kết quả riêng trên Drive/Colab thì chưa kiểm tra được từ đây.
- Notebook mới [code/lab_day2.ipynb](code/lab_day2.ipynb)
  tải code từ GitHub theo commit cố định, tải dữ liệu, chạy kế hoạch ưu tiên tốc độ và sinh sản phẩm.

**Trạng thái cập nhật:** đã chạy xong thực nghiệm GPU local với profile `hour`.
Sản phẩm train gốc ở `local_runs/hour/`; bài nộp đã hoàn thiện báo cáo và minh chứng trong
[`submissions/02859_nguyen_van_than/`](submissions/02859_nguyen_van_than/README.md).

## Kết quả local đã chạy thật (2026-10-05)

- Windows 11, GTX 1650 4 GB, Ryzen 5 6600HS, RAM 16 GB; môi trường `.venv-gpu/`,
  torch 2.7.1+cu126, torchvision 0.22.1+cu126, timm 1.0.30.
- Đã đo forward/backward/AdamW trên 5 kiến trúc: FP32, batch 64, layout mặc định
  nhanh hơn AMP/channels-last trên máy này. FP16 batch 64 bị loại vì loss không hữu hạn.
- 128 px; giữ đủ 10.501 train, 3.501 val, 3.507 test và CSV fold 0 gốc.
- 5 backbone, 3 trục ablation + kết hợp, 5 phương pháp suy luận; khảo sát 1 epoch.
  DeiT tiny + fine-tune/basic/CE thắng trên val. Chung kết 12 epoch, seed 0/1/2.
  F01/T00 dùng chung trọng số cùng seed vì công thức train giống hệt nhau.
- Hflip gộp logits chọn trên val; temperature fit riêng từng seed trước test.
  Sáu manifest F01/T00 đều hoàn tất, mỗi file có đủ 3.507 ảnh test.
- Toàn bộ lượt chạy **36,18 phút**, không tính cài CUDA/tải trọng số/đo cấu hình ban đầu.
- Test F01 (mean ± std mẫu, 3 seed): accuracy **93,67% ± 0,30 điểm phần trăm**;
  macro-F1 **0,9139 ± 0,0044**; ECE **0,0090 ± 0,0025**.
- T00 macro-F1 **0,9104 ± 0,0058**; mức tăng 0,0035 nhỏ hơn độ lệch chuẩn,
  chưa đủ để kết luận cải thiện chắc chắn. Recall Chinee Apple 79,2%, Snake Weed 88,9%.
- P95 model hflip trên GPU này **19,67 ms**, chưa gồm đọc/resize ảnh.
- Đã kiểm tra ZIP không trùng entry, không chứa dataset/checkpoint; workbook đủ 7 sheet,
  5 backbone, 8 training, 5 inference, 2 final, 18 per-class rows. eval.py gốc chạy thành công.

Sản phẩm: [ZIP](local_runs/hour/deepweeds_submission.zip),
[workbook](local_runs/hour/results.xlsx), [báo cáo nháp](local_runs/hour/report.md),
[log](local_runs/hour/training.log). Đây là các sản phẩm tự sinh ban đầu.
Bản hoàn thiện của workbook, báo cáo, curves, predictions và log nhỏ được đưa vào thư mục
bài nộp để commit; dataset ảnh và checkpoint tiếp tục giữ local.

## Hoàn thiện bài nộp và minh chứng bổ sung

- Báo cáo không còn TODO; phân tích backbone/ablation/TTA, Δ so với std, confusion,
  ảnh sai và giới hạn; có bảng đối chiếu từng mục rubric.
- Workbook 8 sheet (7 bắt buộc + Robustness), Final đủ 6 dòng từng seed + 2 mean±std;
  PerClass thêm std, training thêm F1 hai lớp khó; định dạng/freeze/filter/đơn vị.
- Đo lại đủ 5 phương pháp suy luận và F01: 20 warmup/100 lượt, synchronize,
  GPU forward + gộp + softmax. F01 p95 **26,68 ms** trên GTX1650;
  phép đo forward-only gốc 19,67 ms được giữ riêng, không tính decode/resize/H2D.
- Giữ byte-identical mọi CSV dự đoán gốc (đối chiếu archive trước khi bổ sung), không train/test lại.
- Bonus trên toàn bộ 3.501 ảnh validation, seed0: F1 sạch **0,9182**, tối **0,8167**,
  mờ **0,6101**, nhiễu **0,8763**; temperature cố định từ validation sạch, có ECE trước/sau.
- Grad-CAM 6 lỗi validation, hook trước attention cuối; không dùng test để chọn mẫu/đổi model.
- `eval.py` gốc tự chấm **10/20 riêng mục I**; điểm A–H và hai phần bonus do giảng viên xác nhận.

Bài nộp: [README](submissions/02859_nguyen_van_than/README.md),
[báo cáo](submissions/02859_nguyen_van_than/report.md),
[workbook](submissions/02859_nguyen_van_than/results.xlsx),
[rubric evidence](submissions/02859_nguyen_van_than/RUBRIC_EVIDENCE.md).
ZIP hoàn thiện local: `local_runs/hour/deepweeds_submission_final.zip`.
Kiểm tra offline: `python code/verify_submission.py`; tái lập theo README bài nộp.

## Notebook có output

Đã xuất lại `code/lab_day2.ipynb` bằng một phiên Jupyter đọc artifact thật:
13 cell có output và execution_count, 22 ảnh nhúng, bảng/log của nghiên cứu gốc.
Mặc định `VIEW_SAVED_RESULTS=True` để xem kết quả; đặt False để train lại trên Colab.
Output hiển thị được đánh dấu rõ là kết quả local đã lưu; không train hoặc chạy test lại.
Nhánh training trong notebook có output được kiểm tra trên CPU với tập con trong thư mục tạm.
