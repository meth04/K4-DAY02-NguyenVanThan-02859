"""make_results.py - tổng hợp mọi lần chạy thành results.xlsx (GUIDE.md mục 6.1).

Đọc:
  - runs/<exp_id>/seed<k>/summary.json, config.json, history.csv
  - predictions/<exp_id>_seed<k>_{val,test}.csv
  - runs/latency.json, runs/inference.json, runs/backbone_meta.json (nếu có)

Ghi: results.xlsx với các sheet Backbones, Training, Inference, Final, PerClass,
     Latency, Summary.

Chạy:  python code/make_results.py --labels-dir data/labels --out results.xlsx
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS))
sys.path.insert(0, str(_THIS.parent))
import eval as ev  # noqa: E402

# Ánh xạ trục ablation (GUIDE.md mục 3) - có thể chỉnh cho khớp kế hoạch của bạn.
AXIS_OF = {
    "T00": "nền", "T01": "A. khởi tạo (frozen)", "T02": "A. khởi tạo (scratch)",
    "T03": "B. augmentation (color)", "T04": "B. augmentation (trivial)",
    "T05": "B. mixup", "T06": "B. cutmix",
    "T07": "C. loss (label smoothing)", "T08": "C. loss (focal)",
    "T09": "C. loss (ce_weighted)", "T10": "D. sampler (balanced)",
    "T11": "E. optimizer/LR", "T12": "F. EMA", "T13": "G. độ phân giải",
    "T14": "kết hợp tốt nhất",
    "TS00": "nền khảo sát (cùng ngân sách ablation)",
}


def _load_json(p: Path):
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


def collect_summaries(out_dir: Path) -> list[dict]:
    rows = []
    for s in sorted(out_dir.glob("*/seed*/summary.json")):
        d = _load_json(s)
        if d:
            cfg = _load_json(s.parent / "config.json") or {}
            d.update({f"cfg_{k}": v for k, v in cfg.items() if k not in d})
            rows.append(d)
    return rows


def _fmt(mean, std, digits=4):
    if std is None or (isinstance(std, float) and (np.isnan(std))):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def group_stats(pattern: str, csv: str, labels: str, what: str = "test"):
    try:
        names = ev.load_names(labels)
        g = ev.load_group(pattern, csv, ref_what=what)
    except Exception as e:
        return None, None, f"({e})"
    return g, names, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-dir", default="data/labels")
    ap.add_argument("--out-dir", default="runs")
    ap.add_argument("--pred-dir", default="predictions")
    ap.add_argument("--out", default="results.xlsx")
    ap.add_argument("--final", default="F01")
    ap.add_argument("--baseline", default="T00")
    args = ap.parse_args()

    labels_csv = str(Path(args.labels_dir) / "labels.csv")
    test_csv = str(Path(args.labels_dir) / "test_subset0.csv")
    val_csv = str(Path(args.labels_dir) / "val_subset0.csv")
    out_dir = Path(args.out_dir)
    pred_dir = Path(args.pred_dir)

    summaries = collect_summaries(out_dir)
    meta = _load_json(out_dir / "backbone_meta.json") or {}
    latency = _load_json(out_dir / "latency.json") or []
    inference = _load_json(out_dir / "inference.json") or []
    bb_latency = _load_json(out_dir / "backbone_latency.json") or {}

    def gmac_of(backbone):
        return meta.get(backbone, {}).get("gmac", "")

    def tag_of(row):
        return row.get("weight_tag") or meta.get(row["backbone"], {}).get("weight_tag", "")

    # ---------------- Backbones ----------------
    b_rows = []
    for r in summaries:
        if not str(r["exp_id"]).startswith("B"):
            continue
        lat = next((l for l in latency if l.get("exp_id") == r["exp_id"]), {})
        bl = bb_latency.get(r["exp_id"], {})
        p50 = lat.get("p50") or bl.get("p50", "")
        b_rows.append({
            "exp_id": r["exp_id"], "backbone": r["backbone"], "tag trọng số": tag_of(r),
            "#params (M)": round(r["params_m"], 2), "GMAC": gmac_of(r["backbone"]),
            "độ phân giải": r.get("cfg_img_size", 224), "epoch": r["epochs"], "seed": r["seed"],
            "macro-F1 val": round(r["val_macro_f1"], 4), "top-1 val": round(r["val_top1"], 4),
            "train/epoch (s)": round(r["train_time_per_epoch_s"], 1),
            "độ trễ batch-1 (ms)": round(p50, 2) if p50 != "" else "",
            "ghi chú": f"best epoch {r['best_epoch']}",
        })
    backbones = pd.DataFrame(b_rows)

    # ---------------- Training ----------------
    t00 = next((r for r in summaries if r["exp_id"] == "T00" and r["seed"] == 0), None)
    t_rows = []
    for r in summaries:
        if not str(r["exp_id"]).startswith("T"):
            continue
        reference_id = r.get("comparison_baseline", "T00")
        reference = next((s for s in summaries if s["exp_id"] == reference_id and s["seed"] == 0), t00)
        delta = "" if reference is None else round(r["val_macro_f1"] - reference["val_macro_f1"], 4)
        t_rows.append({
            "exp_id": r["exp_id"], "backbone": r["backbone"],
            "trục thay đổi": AXIS_OF.get(r["exp_id"], "?"),
            "khác T00 ở": f"init={r['init']} aug={r['aug']} loss={r['loss']} mix={r['mix']} "
                          f"sampler={r.get('cfg_sampler')} ema={r.get('cfg_ema_decay')}",
            "seed": r["seed"], "macro-F1 val": round(r["val_macro_f1"], 4),
            "top-1 val": round(r["val_top1"], 4), "mốc so sánh": reference_id, "Δ so với mốc": delta,
            "epoch tối đa": r["epochs"], "epoch đã chạy": r.get("epochs_run", r["epochs"]),
            "ghi chú": r.get("provenance", ""),
        })
    training = pd.DataFrame(t_rows)

    # ---------------- Inference ----------------
    inference_config = _load_json(out_dir / args.final / "seed0" / "config.json") or {}
    for row in inference:
        if row.get("exp_id") == "I00":
            precision = ("AMP BF16" if inference_config.get("amp_dtype") == "bfloat16" else "AMP FP16") \
                        if inference_config.get("amp", True) else "FP32"
            row["phương pháp"] = f"1-view {precision}"
    inference_df = pd.DataFrame(inference) if inference else pd.DataFrame(
        columns=["exp_id", "phương pháp", "mô hình/checkpoint", "K", "macro-F1 val",
                 "top-1 val", "ECE val", "p50 (ms)", "p95 (ms)", "p99 (ms)",
                 "ảnh/s", "chi phí so với I00"])

    # ---------------- Final + PerClass ----------------
    final_rows, perclass_rows = [], []
    groups = {}
    for exp in [args.final, args.baseline]:
        g_test, names, err = group_stats(f"{pred_dir}/{exp}_seed*_test.csv", test_csv, labels_csv, "test")
        g_val, _, _ = group_stats(f"{pred_dir}/{exp}_seed*_val.csv", val_csv, labels_csv, "val")
        if g_test is None:
            continue
        groups[exp] = (g_test, g_val, names)
        mf = g_test.summary["macro_f1"]
        final_rows.append({
            "exp_id": exp, "cấu hình": exp, "seeds": ",".join(map(str, g_test.seeds)),
            "macro-F1 val": _fmt(*g_val.summary["macro_f1"]) if g_val else "",
            "macro-F1 test": _fmt(*mf),
            "top-1 test": _fmt(*g_test.summary["top1"]),
            "ECE test": _fmt(*g_test.summary["ece"]),
        })
        rm = g_test.summary["recall"][0]
        for i, n in enumerate(names):
            perclass_rows.append({
                "cấu hình": exp, "lớp": n, "số ảnh test": int(g_test.metrics[0]["support"][i]),
                "precision": round(float(g_test.summary["precision"][0][i]), 4),
                "recall": round(float(rm[i]), 4),
                "F1": round(float(g_test.summary["f1"][0][i]), 4),
            })
    final_df = pd.DataFrame(final_rows)
    perclass = pd.DataFrame(perclass_rows)

    # ---------------- Latency ----------------
    latency_df = pd.DataFrame(latency) if latency else pd.DataFrame(
        columns=["cấu hình", "GPU", "dtype", "batch", "gộp BN", "p50", "p95", "p99", "ảnh/s"])

    # ---------------- Summary ----------------
    all_rows = []
    for r in summaries:
        all_rows.append({
            "exp_id": r["exp_id"], "backbone": r["backbone"],
            "macro-F1 val": round(r["val_macro_f1"], 4), "top-1 val": round(r["val_top1"], 4),
            "params (M)": round(r["params_m"], 2), "GMAC": gmac_of(r["backbone"]),
            "train/epoch (s)": round(r["train_time_per_epoch_s"], 1),
            "cấu hình": f"init={r['init']} aug={r['aug']} loss={r['loss']} mix={r['mix']}",
        })
    summary = pd.DataFrame(all_rows).sort_values("macro-F1 val", ascending=False).head(10)

    # ---------------- Ghi xlsx ----------------
    out = Path(args.out)
    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        backbones.to_excel(xw, sheet_name="Backbones", index=False)
        training.to_excel(xw, sheet_name="Training", index=False)
        inference_df.to_excel(xw, sheet_name="Inference", index=False)
        final_df.to_excel(xw, sheet_name="Final", index=False)
        perclass.to_excel(xw, sheet_name="PerClass", index=False)
        latency_df.to_excel(xw, sheet_name="Latency", index=False)
        summary.to_excel(xw, sheet_name="Summary", index=False)
    print(f"Đã ghi {out} với {len(backbones)} backbone, {len(training)} training, "
          f"{len(inference_df)} inference, {len(final_df)} final, {len(perclass)} per-class rows.")

    # phụ trợ cho report
    if groups:
        Path("eval_out").mkdir(exist_ok=True)
        for exp, (g_test, g_val, names) in groups.items():
            (Path("eval_out") / f"{exp}_final.json").write_text(
                json.dumps({k: {"mean": float(v[0]) if np.ndim(v[0]) == 0 else np.asarray(v[0]).tolist(),
                                "std": float(v[1]) if np.ndim(v[1]) == 0 else np.asarray(v[1]).tolist()}
                            for k, v in g_test.summary.items()}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
