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

**Trạng thái:** đã chuẩn bị ảnh và triển khai pipeline; cần chạy thực nghiệm GPU,
xem kết quả, bổ sung phân tích và nộp. Chưa thể suy ra phần trăm hoàn thành hay điểm số.

## Tiếp theo

Upload notebook mới lên Colab, bật GPU, chạy `PROFILE="fast"`.
Notebook in thời gian mỗi epoch và lưu thí nghiệm đã xong về Drive.
Đọc báo cáo nháp, phân tích ảnh sai và Δ so với nhiễu trước khi nộp.
Profile `full` tăng ngân sách để khảo sát/chung kết lâu hơn.
