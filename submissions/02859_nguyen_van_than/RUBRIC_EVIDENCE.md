# Đối chiếu rubric và minh chứng

Các mục dưới đây là vị trí minh chứng, **không phải cam kết điểm do sinh viên tự chấm**. Tham chiếu [RUBRIC lớp](evidence/lab_docs/RUBRIC.md). Mọi lựa chọn đã chốt trước test; báo cáo giữ các kết quả yếu/thí nghiệm không có lợi.

| Mục | Minh chứng | Giới hạn cần người chấm xem |
|---|---|---|
| P1 — 6 sản phẩm | README.md, results.xlsx, report.md, curves/, code/, predictions/ | Có đủ trong thư mục bài nộp |
| P2 — số liệu thật | [training.log](evidence/training.log), [runs](evidence/runs/), [manifest](manifest.json) | Mỗi exp/seed có config/history/summary; T00 và TS00 là alias có provenance |
| P3 — fold 0 | [split_check.json](evidence/runs/split_check.json), [CSV gốc](evidence/labels/) | Giữ nguyên khác biệt 1 nhãn catalog/fold ở train; không tự chia/sửa CSV |
| P4 — dự đoán đủ seed | [predictions](predictions/), [grade_I.json](evidence/evaluation/grade_I.json) | F01/T00 đều đủ seed0/1/2, 3507 ảnh test/seed; có val và uncal |
| A — pipeline (12) | Report §2, EDA/augmentation PNG, [pipeline_checks.json](evidence/runs/pipeline_checks.json), environment/config | Loss đầu/overfit/checks có số thật; test manifests complete |
| B — backbone (12) | Backbones sheet, report §3, B01–B05 curves/config/summary | 5 mạng, ResNet/ConvNeXt/transformer/MobileNet; khảo sát chỉ 1epoch/1seed; GMAC transformer là cận dưới |
| C — training (16) | Training sheet, report §4, TS00/T01/T03/T07/T14 | 3 trục × 2 giá trị; các thử đơn đổi 1 yếu tố; kết hợp color+LS không thắng; chưa có std ablation |
| D — inference (12) | Inference/Latency sheets, report §5, [timings](evidence/extras/inference_timings.json), tradeoff PNG | 4 phương pháp ngoài I00; 20warmup/100 synchronized iterations; ensemble chỉ 1epoch nên không so ngang ngân sách final |
| E — workbook (8) | [results.xlsx](results.xlsx) | 7 sheet bắt buộc + Robustness; Final có từng seed và dòng mean±std; header/freeze/filter/units |
| F — curves (4) | [curves](curves/), evidence/runs/*/seed*/history.csv | Mỗi B/T/F có ảnh; F01/T00 còn có từng seed; 1epoch screening chỉ một điểm |
| G — báo cáo (12) | [report.md](report.md) §1–9, confusion/errors/Grad-CAM | Có kết luận, Δ so std, F1/recall lớp khó, lỗi ảnh, hạn chế; không còn TODO |
| H — tái lập (4) | [README](README.md), [code](code/), [requirements-local](requirements-local.txt), eval.py gốc | Code hoàn thiện trong code/, không sửa starter hay evaluator lớp; notebook Colab và lệnh kiểm tra offline |
| I — model (20) | [grade_I.json](evidence/evaluation/grade_I.json), Final/PerClass, predictions | eval.py tự chấm 10/20 (ngưỡng tạm thời); không phải tổng bài |

## Phần bổ sung xin xem xét điểm thưởng

| Thí nghiệm thật | Minh chứng | Mức trong rubric / giới hạn |
|---|---|---|
| Grad-CAM giải thích lỗi | Report §8.2, [ảnh](curves/gradcam_val.png), [thông tin](evidence/extras/gradcam_val.json), code/submission_extras.py | Mục +1; 6 lỗi validation, single-view predicted class, không dùng test để chọn mẫu |
| Phân tích lệch phân phối | Report §8.1, Robustness sheet, [protocol](evidence/extras/robustness_protocol.json), 4 CSV val đầy đủ | Mục +2; 3501 ảnh/điều kiện, seed0, tối/mờ/nhiễu, ECE trước/sau T cố định từ sạch |

Không nhận điểm nhiều fold/DINOv2/distillation/adaptation/ONNX vì chưa có thực nghiệm. Giảng viên xác nhận điểm A–H và thưởng; số epoch khảo sát ngắn và những thiếu sót nêu trong báo cáo có thể ảnh hưởng điểm.
