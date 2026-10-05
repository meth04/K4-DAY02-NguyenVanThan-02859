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
Sản phẩm ở `local_runs/hour/`; xem chi tiết bên dưới. Báo cáo vẫn là bản nháp cần
đọc và bổ sung phân tích trước khi nộp.

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
[log](local_runs/hour/training.log). File kết quả lớn và trọng số được giữ local,
không đưa vào Git. Code/nhật ký môi trường có trong ZIP.

## Tiếp theo

Đọc báo cáo nháp, phân tích ảnh sai và Δ so với nhiễu trước khi nộp.
Chạy lại local: `.venv-gpu/Scripts/python.exe -X utf8 -u code/run_local.py --minutes 60`;
dùng lại các lần train/test đã hoàn tất. Trên Colab có thể chọn `hour`, `fast` hoặc `full`,
nhưng thời gian đã đo ở trên chỉ áp dụng cho máy local và cấu hình đã ghi.
