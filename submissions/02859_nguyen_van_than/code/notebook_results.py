"""Display verified results of the completed local study, without model inference.

Used by the notebook's saved-results mode. Training cells remain available when
VIEW_SAVED_RESULTS=False. All data come from the committed submission artifacts.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
from IPython.display import Image, Markdown, display


class SavedResults:
    def __init__(self, root):
        self.root = Path(root)
        self.environment = self.read("evidence/environment.json")
        self.sheets = pd.read_excel(self.root / "results.xlsx", sheet_name=None)
        self.selection = self.read("evidence/selection.json")
        self.manifest = self.read("manifest.json")
        self.verified = set()
        self.log = (self.root / "evidence/training.log").read_text(encoding="utf-8")

    def read(self, name):
        return json.loads((self.root / name).read_text(encoding="utf-8"))

    def verify(self, names):
        for name in names:
            if name in self.verified:
                continue
            path = self.root / name
            expected = self.manifest[name]
            if path.stat().st_size != expected["bytes"] or hashlib.sha256(path.read_bytes()).hexdigest() != expected["sha256"]:
                raise ValueError(f"Saved artifact checksum differs: {name}")
            self.verified.add(name)

    def image(self, name, width=900):
        filename = "curves/" + name
        self.verify([filename])
        display(Image(filename=str(self.root / filename), width=width))

    def heading(self, text):
        display(Markdown("**Kết quả lượt train local đã lưu — không train/test lại.**\n\n" + text))

    def stage_log(self, start, end):
        begin = self.log.rfind("STAGE: " + start)
        if begin < 0:
            raise ValueError(f"Missing recorded stage: {start}")
        finish = self.log.find("STAGE: " + end, begin) if end else len(self.log)
        if finish < 0:
            raise ValueError(f"Missing next recorded stage: {end}")
        print(self.log[begin:finish].strip())

    def show(self, section):
        self.verify(["results.xlsx", "evidence/training.log", "evidence/environment.json", "evidence/selection.json"])
        if section == "environment":
            self.heading("Phần cứng/phiên bản dưới đây thuộc lần train gốc trên GTX 1650; kernel hiện tại chỉ đọc kết quả.")
            display(pd.DataFrame([{k: self.environment.get(k) for k in ["gpu","python","torch","torchvision","timm","numpy","profile","source_commit"]}]))
            display(pd.DataFrame([self.environment["settings"]]))
        elif section == "source":
            self.heading("Code train gốc: `9747314cb5aa60fc727fefaa089aacc3de544d15`. Artifact trong manifest được kiểm tra SHA-256 trước khi hiển thị.")
            print("Nguồn kết quả:", self.root)
            print("Đã đọc workbook:", ", ".join(self.sheets))
        elif section == "initialize":
            self.heading("Profile hour: khảo sát 1 epoch, chung kết 12 epoch; 128px, batch64, FP32/NCHW, 2 workers; seed0/1/2.")
            self.verify(["evidence/local_plan.json", "evidence/local_complete.json"])
            display(pd.DataFrame([self.read("evidence/local_plan.json")]))
            print("Nghiên cứu hoàn tất sau", round(self.read("evidence/local_complete.json")["elapsed_s"] / 60, 2), "phút.")
        elif section == "data":
            self.verify(["evidence/runs/split_check.json"])
            split = self.read("evidence/runs/split_check.json")
            self.heading("DeepWeeds fold 0 gốc; giao rỗng và hợp đủ 17.509 ảnh. Một nhãn catalog/fold lệch ở train được giữ nguyên theo CSV fold.")
            print("Số ảnh:", split["n"], "| Giao:", split["overlap"], "| Hợp:", split["union"])
            names = self.sheets["PerClass"].query("`cấu hình` == 'F01'")["lớp"].tolist()
            frame = pd.DataFrame(split["per_class"]).reindex([str(i) for i in range(9)])
            frame.index = names
            display(frame)
            display(pd.DataFrame(split["label_audit"]["catalog_label_differences"]))
            self.image("eda_classes.png")
            self.image("eda_samples.png", 700)
        elif section == "checks":
            self.verify(["evidence/runs/pipeline_checks.json"])
            checks = self.read("evidence/runs/pipeline_checks.json")
            self.heading("Các kiểm tra và overfit batch dưới đây được thực hiện trước huấn luyện, lấy từ pipeline_checks.json và log gốc.")
            self.stage_log("checks =", "backbone_table =")
            display(pd.DataFrame([{"initial_loss": checks["initial_loss"], "ln(9)": 2.1972245773362196,
                                   "overfit_loss": checks["overfit_loss"], "steps": checks["steps"],
                                   "checks_passed": sum(checks["checks"]), "checks_total": len(checks["checks"])}]))
            self.image("augmentation.png")
        elif section == "backbones":
            self.heading("Năm backbone: cùng fold/seed0/công thức/128px/1epoch. DeiT tiny đi tiếp theo validation và ngân sách GPU.")
            self.stage_log("backbone_table =", "ablation_table =")
            display(self.sheets["Backbones"])
            self.image("backbone_tradeoff.png")
            for _, row in self.sheets["Backbones"].iterrows():
                self.image(f"{row.exp_id}_{row.backbone}.png")
        elif section == "ablations":
            self.heading("So ablation với TS00 ở cùng 1epoch; T00 là mốc 12epoch cuối cùng. Color+LS không thắng; giữ finetune/basic/CE.")
            self.stage_log("ablation_table =", "lab.train_final()")
            display(self.sheets["Training"])
            for exp in ["TS00","T01","T03","T07","T14"]:
                self.image(f"{exp}_deit_tiny_patch16_224.png")
        elif section == "inference":
            self.heading("Chọn suy luận trên val seed0. Bảng dùng phép đo bổ sung 20warmup/100 lượt (forward+aggregate+softmax); log gốc bên dưới dùng forward-only 10/50. Không trộn phạm vi đo.")
            self.stage_log("lab.train_final()", "lab.finals()")
            display(self.sheets["Inference"])
            display(self.sheets["Latency"])
            print("Phương pháp đã chốt trước test:", self.selection["method"])
            self.image("inference_tradeoff.png")
            self.image("F01_seed0.png")
        elif section == "finals":
            self.heading("Log train seed1/2 và test cuối cùng (cũ). Mỗi CSV test đủ 3.507 ảnh; bản uncal dùng cùng logits. Kernel này không gọi model hoặc test.")
            self.stage_log("lab.finals()", 'if PROFILE != "smoke":')
            display(self.sheets["Final"])
            display(self.sheets["PerClass"])
            for seed in [1,2]:
                self.image(f"F01_seed{seed}.png")
        elif section == "products":
            self.heading("Kết quả được tính lại từ dự đoán đã lưu bằng eval.py gốc. Mục I: 10/20; không phải tổng điểm bài lab.")
            self.verify(["evidence/evaluation/grade_I.json"])
            grade = self.read("evidence/evaluation/grade_I.json")
            display(pd.DataFrame(grade["items"]))
            print("Mục I:", grade["total"], "/", grade["max_scored"])
            print("Workbook có", len(self.sheets), "sheet | dự đoán 3 seed | thời gian train 36,18 phút")
            self.image("F01_confusion.png", 700)
            self.image("F01_errors.png")
            display(Markdown("### Kết luận\n\nAccuracy test **93,67% ±0,30 điểm phần trăm**; macro-F1 **0,9139 ±0,0044**. Δ so T00 = **+0,0035 < std0,0058**; chưa chứng minh cải thiện chắc chắn. Recall Chinee apple **79,2%**, Snake weed **88,9%**. Độ trễ F01 bổ sung p95 **26,68ms** trên GTX1650, chưa gồm decode/resize/H2D. Xem report.md để đọc đầy đủ phân tích và hạn chế."))
        elif section == "extras":
            self.heading("Bonus sau nghiên cứu: validation sạch/tối/mờ/nhiễu, không train/refit T/test lại; Grad-CAM 6 lỗi validation.")
            display(self.sheets["Robustness"])
            self.image("robustness_val.png")
            self.image("gradcam_val.png", 700)
        elif section == "download":
            self.heading("Notebook này đã có output sẵn để xem trên GitHub/Colab. Tải .ipynb bằng File → Download; bộ bài nộp đầy đủ nằm trong repository.")
            display(Markdown("[Bài nộp đầy đủ](https://github.com/meth04/K4-DAY02-NguyenVanThan-02859/tree/main/submissions/02859_nguyen_van_than) · [Báo cáo](https://github.com/meth04/K4-DAY02-NguyenVanThan-02859/blob/main/submissions/02859_nguyen_van_than/report.md)"))
        else:
            raise ValueError(section)
