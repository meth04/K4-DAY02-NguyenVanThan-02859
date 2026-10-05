"""make_report.py - sinh report.md từ results.xlsx + eval_out/ + runs/.

Chạy:  python code/make_report.py --labels-dir data/labels
Ghi:   report.md  (bản nháp có số liệu thật; bạn đọc và bổ sung nhận xét/giả thuyết)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS))
sys.path.insert(0, str(_THIS.parent))
import eval as ev  # noqa: E402


def _md_table(df: pd.DataFrame) -> str:
    if df is None or len(df) == 0:
        return "_(không có dữ liệu)_\n"
    cols = list(df.columns)
    out = ["| " + " | ".join(str(c) for c in cols) + " |",
           "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in r.values) + " |")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-dir", default="data/labels")
    ap.add_argument("--xlsx", default="results.xlsx")
    ap.add_argument("--final", default="F01")
    ap.add_argument("--baseline", default="T00")
    ap.add_argument("--out", default="report.md")
    args = ap.parse_args()

    labels_csv = str(Path(args.labels_dir) / "labels.csv")
    test_csv = str(Path(args.labels_dir) / "test_subset0.csv")
    val_csv = str(Path(args.labels_dir) / "val_subset0.csv")
    names = ev.load_names(labels_csv)
    baseline_config = Path("runs") / args.baseline / "seed0" / "config.json"
    baseline = json.loads(baseline_config.read_text(encoding="utf-8")) if baseline_config.exists() else {}
    resolution = baseline.get("img_size", 224)
    precision = "AMP" if baseline.get("amp", True) else "FP32"

    lines = ["# Báo cáo Lab Day 2 — Backbone, công thức huấn luyện và suy luận trên DeepWeeds", ""]

    # --- Tóm tắt ---
    lines += ["## 1. Tóm tắt", ""]
    try:
        g_final = ev.load_group(f"predictions/{args.final}_seed*_test.csv", test_csv, ref_what="test")
        mf, sdf = g_final.summary["macro_f1"]
        t1, sdt = g_final.summary["top1"]
        lines += [
            f"- Cấu hình chung kết: **{args.final}** (backbone + công thức + suy luận tốt nhất trên val).",
            f"- Test (mean ± std qua {len(g_final.preds)} seed): **macro-F1 {mf:.4f} ± {sdf:.4f}**, "
            f"**top-1 {t1:.4f} ± {sdt:.4f}**.",
        ]
        g_base = ev.load_group(f"predictions/{args.baseline}_seed*_test.csv", test_csv, ref_what="test")
        mb = g_base.summary["macro_f1"][0]
        lines += [f"- Mốc `{args.baseline}` + I00: macro-F1 {mb:.4f}. "
                  f"Δ = {mf - mb:+.4f} so với mốc.", ""]
    except Exception as e:
        lines += [f"_(chưa có predictions/{args.final}_seed*_test.csv: {e})_", ""]

    # --- Dữ liệu & thiết lập ---
    lines += ["## 2. Dữ liệu và thiết lập", "",
              "- Dataset: DeepWeeds, 17.509 ảnh 256×256, 9 lớp; chia sẵn fold 0 (60/20/20).",
              "- Chỉ số chính: macro-F1 (9 lớp). Phụ: top-1, balanced accuracy, F1 từng lớp, ECE (15 bin).",
              "- Chọn mọi thứ trên **val**; test chạy **một lần mỗi seed** ở cuối.",
              f"- Công thức nền `T00`: ImageNet pretrain, RandomResizedCrop({resolution})+flip, AdamW "
              f"(backbone 1e-4 / head 1e-3), wd 0.05 (không áp cho norm/bias), warmup+cosine, CE, {precision}.",
              ""]

    # --- Bảng từ xlsx ---
    xlsx = Path(args.xlsx)
    if xlsx.exists():
        xl = pd.ExcelFile(xlsx)
        for sheet, title in [("Backbones", "## 3. So sánh backbone"),
                             ("Training", "## 4. Công thức huấn luyện"),
                             ("Inference", "## 5. Phương pháp suy luận"),
                             ("Final", "## 6. Cấu hình tốt nhất (test)"),
                             ("PerClass", "### 6.1. Chỉ số theo lớp"),
                             ("Latency", "### 6.2. Độ trễ")]:
            if sheet in xl.sheet_names:
                lines += [title, "", _md_table(pd.read_excel(xl, sheet)), ""]
    else:
        lines += [f"_(chưa có {xlsx}; chạy code/make_results.py trước)_", ""]

    # --- Kết luận / hạn chế (khung để người viết bổ sung) ---
    lines += [
        "## 7. Kết luận và khuyến nghị", "",
        "> TODO (bạn viết, dựa số trong results.xlsx): yếu tố nào đóng góp nhiều nhất — backbone, "
        "công thức huấn luyện hay suy luận? Cấu hình tốt nhất tốt hơn mốc bao nhiêu, có vượt nhiễu (std) không? "
        "Nếu triển khai trên robot (30–100 ms/khung) chọn gì?", "",
        "## 8. Hạn chế và việc tiếp theo", "",
        "- Một fold (fold 0); số seed hữu hạn; chia ngẫu nhiên không theo địa điểm nên điểm test có thể lạc quan.",
        "- Ngân sách GPU hạn chế -> số epoch/ablation có thể rút gọn (ghi rõ ở đây).",
        "- Chưa kiểm chứng trên miền khác (mùa/ánh sáng khác).", "",
        "## 9. Phụ lục", "",
        "- Danh sách `exp_id` và cấu hình đầy đủ: xem `runs/<exp_id>/seed<k>/config.json`.",
        "- Notebook Colab/Kaggle chạy lại: xem README riêng (`SUBMISSION_README.md`).", "",
    ]

    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print(f"Đã ghi {args.out}")


if __name__ == "__main__":
    main()
