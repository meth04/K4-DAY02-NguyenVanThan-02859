"""study.py - Bước 3 (so sánh phương pháp suy luận) và Bước 4 (finalize test).

Dùng các hàm trong inference.py / benchmark.py, đọc checkpoint đã train trong runs/.

Entry point cho notebook:
    inference_study(exp_id="T00", seed=0)   -> so sánh I00..I08 trên VAL + đo độ trễ
    finalize_test(exp_id="F01", seeds=[0,1,2], method="hflip", temperature=True)
        -> nạp checkpoint, tính logits TTA trên val+test, khớp T trên VAL, ghi predictions/
           (test có T, test chưa T cho I4a, val cho I4b)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS))
sys.path.insert(0, str(_THIS.parent))

import benchmark as bm          # noqa: E402
import dataset as ds            # noqa: E402
import inference as inf         # noqa: E402
import model as md              # noqa: E402
import train as tr              # noqa: E402
from eval import compute_metrics, save_predictions  # noqa: E402

_CACHE = {}


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_cfg(exp_id: str, seed: int, out_dir: str = "runs") -> tr.Config:
    p = Path(out_dir) / exp_id / f"seed{seed}" / "config.json"
    return tr.Config(**json.loads(p.read_text(encoding="utf-8")))


def _load_model(cfg: tr.Config, seed: int, out_dir: str = "runs", fuse_bn: bool = False):
    ckpt = Path(out_dir) / cfg.exp_id / f"seed{seed}" / "best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"không thấy checkpoint {ckpt}")
    model = md.build_model(cfg.backbone, pretrained=False, num_classes=ds.NUM_CLASSES,
                           drop_rate=cfg.drop_rate, init="finetune")
    model.load_state_dict(torch.load(ckpt, map_location="cpu")["state_dict"])
    model.eval()
    if fuse_bn:
        model = inf.fuse_conv_bn(model)
    return model.to(_device())


def _eval_loader(cfg: tr.Config, split: str):
    df = ds.load_split(cfg.labels_dir, cfg.fold)[{"train": 0, "val": 1, "test": 2}[split]]
    tf = ds.build_transforms(False, cfg.img_size, cfg.aug)
    return ds.make_loader(df, cfg.images_dir, tf, cfg.batch_size, False, None,
                          cfg.num_workers, cfg.seed)


def _view_logits(model, cfg, split, mode, sizes=(256, 288)):
    """Trả về (filenames, y_true, list_logits_per_view)."""
    loader = _eval_loader(cfg, split)
    if mode == "identity":
        fn, yt, lg = inf.predict_logits(model, loader, _device(), None)
        return fn, yt, [lg]
    if mode == "hflip":
        fn, yt, lg = inf.predict_logits(model, loader, _device(), inf.view_identity)
        fn2, yt2, lg2 = inf.predict_logits(model, loader, _device(), inf.view_hflip)
        return fn, yt, [lg, lg2]
    if mode == "multicrop":
        acc, fn, yt = [], None, None
        for ci in range(5):
            def vf(x, ci=ci):
                return inf.views_multicrop(x, 224)[ci]
            fn, yt, l = inf.predict_logits(model, loader, _device(), vf)
            acc.append(l)

        def vf_mid_flip(x):
            return torch.flip(inf.views_multicrop(x, 224)[2], dims=[3])
        fn, yt, l = inf.predict_logits(model, loader, _device(), vf_mid_flip)
        acc.append(l)
        return fn, yt, acc
    if mode == "multiscale":
        acc, fn, yt = [], None, None
        for s in sizes:
            def vf(x, s=s):
                return inf.views_multiscale(x, [s])[0]
            fn, yt, l = inf.predict_logits(model, loader, _device(), vf)
            acc.append(l)
        return fn, yt, acc
    raise ValueError(mode)


def _metrics_from_probs(y_true, probs):
    m = compute_metrics(np.asarray(y_true), probs.argmax(1), probs)
    return {k: float(m[k]) for k in ("top1", "macro_f1", "balanced_acc", "ece", "nll")}


def _seeds_of(exp_id, out_dir):
    d = Path(out_dir) / exp_id
    return sorted(int(p.name.replace("seed", "")) for p in d.glob("seed*")
                  if (p / "best.pt").exists())


def _append_json(out_dir, name, rows):
    p = Path(out_dir) / name
    old = json.loads(p.read_text(encoding="utf-8")) if p.exists() else []
    p.write_text(json.dumps(old + rows, indent=2, ensure_ascii=False), encoding="utf-8")


def inference_study(exp_id: str = "T00", seed: int = 0, out_dir: str = "runs",
                    ensemble_ids: list[str] | None = None,
                    iters: int = 50, warmup: int = 10) -> list[dict]:
    """So sánh các phương pháp suy luận trên VAL, đo độ trễ, ghi runs/inference.json."""
    cfg = _load_cfg(exp_id, seed, out_dir)
    model = _load_model(cfg, seed, out_dir)
    rows, latency_rows = [], []

    def add(code, method, K, probs, yt, latency=None, note=""):
        m = _metrics_from_probs(yt, probs)
        row = {"exp_id": code, "phương pháp": method, "mô hình/checkpoint": f"{exp_id}/seed{seed}",
               "K": K, "macro-F1 val": round(m["macro_f1"], 4), "top-1 val": round(m["top1"], 4),
               "ECE val": round(m["ece"], 4), "p50 (ms)": "", "p95 (ms)": "", "p99 (ms)": "",
               "ảnh/s": "", "chi phí so với I00": "", "ghi chú": note}
        if latency:
            row.update({"p50 (ms)": round(latency["p50"], 2), "p95 (ms)": round(latency["p95"], 2),
                        "p99 (ms)": round(latency["p99"], 2),
                        "ảnh/s": round(latency["images_per_s"], 1)})
        rows.append(row)
        return m

    # I00 - 1 view (mốc)
    fn, yt, lg00 = _view_logits(model, cfg, "val", "identity")
    lat00 = bm.latency_report(model, 1, cfg.img_size, "fp32", _device().type, warmup, iters)
    p00 = inf.aggregate_views(lg00, "prob")
    m00 = add("I00", "1 view (mốc)", 1, p00, yt, lat00)
    latency_rows.append({"cấu hình": f"{exp_id} I00 fp32", "exp_id": exp_id, "GPU": lat00["gpu"],
                         "dtype": "fp32", "batch": 1, "gộp BN": "không", "p50": round(lat00["p50"], 2),
                         "p95": round(lat00["p95"], 2), "p99": round(lat00["p99"], 2),
                         "ảnh/s": round(lat00["images_per_s"], 1)})

    # I01 - TTA lật ngang (K=2)
    _, _, lgh = _view_logits(model, cfg, "val", "hflip")
    lat_hflip = bm.tta_latency(model, 2, 1, cfg.img_size, "fp32", _device().type, warmup, iters)
    add("I01", "TTA lật ngang", 2, inf.aggregate_views(lgh, "prob"), yt, lat_hflip)
    latency_rows.append({"cấu hình": f"{exp_id} I01 fp32 (K=2)", "exp_id": exp_id,
                         "GPU": lat_hflip["gpu"], "dtype": "fp32", "batch": 1, "gộp BN": "không",
                         "p50": round(lat_hflip["p50"], 2), "p95": round(lat_hflip["p95"], 2),
                         "p99": round(lat_hflip["p99"], 2), "ảnh/s": round(lat_hflip["images_per_s"], 1)})

    # I03 - gộp logit thay vì gộp xác suất (cùng 2 view lật)
    add("I03", "gộp logit (K=2)", 2, inf.aggregate_views(lgh, "logit"), yt, lat_hflip,
        note="so sánh với I01 gộp xác suất")

    # I02 - multi-crop (5 crop + lật giữa)
    try:
        _, _, lgc = _view_logits(model, cfg, "val", "multicrop")
        lat_crop = bm.tta_latency(model, 6, 1, cfg.img_size, "fp32", _device().type, warmup, iters)
        add("I02", "TTA 5 crop + lật", 6, inf.aggregate_views(lgc, "prob"), yt, lat_crop)
    except Exception as e:
        print(f"  (bỏ qua I02 multicrop: {e})")

    # I04 - dò độ phân giải kiểm tra (chỉ đổi kích thước đầu vào)
    loader = _eval_loader(cfg, "val")
    for s in (256, 288):
        try:
            def vf_resize(x, s=s):
                return inf.views_multiscale(x, [s])[0]
            _, yt_s, lg_s = inf.predict_logits(model, loader, _device(), vf_resize)
            lat_s = bm.latency_report(model, 1, s, "fp32", _device().type, warmup, iters)
            add(f"I04@{s}", f"kiểm tra ở {s}px", 1, inf.aggregate_views([lg_s], "prob"), yt_s, lat_s)
        except Exception as e:
            print(f"  (bỏ qua I04@{s}: {e})")

    # I07 - temperature scaling (khớp T trên val)
    T = inf.fit_temperature(lg00[0], yt)
    p_T = inf.apply_temperature(lg00[0], T)
    add("I07", f"temperature scaling (T={T:.3f})", 1, p_T, yt, None,
        note=f"ECE {m00['ece']:.4f} -> {_metrics_from_probs(yt, p_T)['ece']:.4f}")

    # I08 - gộp BN / FP16 (chỉ CNN)
    try:
        fused = _load_model(cfg, seed, out_dir, fuse_bn=True)
        _, _, lg_f = inf.predict_logits(fused, loader, _device(), None)
        p_f = inf.aggregate_views([lg_f], "prob")
        lat_f = bm.latency_report(fused, 1, cfg.img_size, "fp32", _device().type, warmup, iters)
        add("I08", "gộp BatchNorm", 1, p_f, yt, lat_f,
            note=f"accuracy gần như không đổi; max|ΔF1|={abs(m00['macro_f1']-_metrics_from_probs(yt, p_f)['macro_f1']):.4f}")
        if _device().type == "cuda":   # FP16 trên CPU bị emulate, cực chậm và vô nghĩa
            lat_fp16 = bm.latency_report(fused, 1, cfg.img_size, "fp16", _device().type, warmup, iters)
            rows.append({"exp_id": "I08fp16", "phương pháp": "gộp BN + FP16",
                         "mô hình/checkpoint": f"{exp_id}/seed{seed}", "K": 1,
                         "macro-F1 val": "", "top-1 val": "", "ECE val": "",
                         "p50 (ms)": round(lat_fp16["p50"], 2), "p95 (ms)": round(lat_fp16["p95"], 2),
                         "p99 (ms)": round(lat_fp16["p99"], 2), "ảnh/s": round(lat_fp16["images_per_s"], 1),
                         "chi phí so với I00": "", "ghi chú": "chỉ đo độ trễ"})
    except Exception as e:
        print(f"  (bỏ qua I08 fuse BN: {e})")

    # I05 - ensemble nhiều seed / nhiều cấu hình
    if ensemble_ids:
        try:
            probs_list = []
            for eid in ensemble_ids:
                for sd in _seeds_of(eid, out_dir):
                    m = _load_model(_load_cfg(eid, sd, out_dir), sd, out_dir)
                    _, _, lg = _view_logits(m, cfg, "val", "identity")
                    probs_list.append(inf.aggregate_views(lg, "prob"))
            add("I05", f"ensemble {len(probs_list)} mô hình", len(probs_list),
                inf.ensemble_probs(probs_list), yt, None)
        except Exception as e:
            print(f"  (bỏ qua I05 ensemble: {e})")

    # chi phí tương đối so với I00
    if lat00["p50"] > 0:
        for r in rows:
            if r["p50 (ms)"] != "":
                r["chi phí so với I00"] = f"{r['p50 (ms)'] / lat00['p50']:.2f}x"
            elif r["K"] != "":
                r["chi phí so với I00"] = f"{r['K']}x"

    _append_json(out_dir, "inference.json", rows)
    _append_json(out_dir, "latency.json", latency_rows)
    print(f"inference_study({exp_id} seed{seed}) xong: {len(rows)} phương pháp.")
    return rows


def finalize_test(exp_id: str = "F01", seeds=(0, 1, 2), method: str = "hflip",
                  temperature: bool = True, out_dir: str = "runs", pred_dir: str = "predictions"):
    """Nạp checkpoint, tính logits TTA trên val+test, khớp T trên val, ghi predictions/.

    Ghi:
      predictions/<exp_id>_seed<k>_test.csv      (TTA + T)  -> chung kết
      predictions/<exp_id>uncal_seed<k>_test.csv (TTA, chưa T) -> I4(a)
      predictions/<exp_id>_seed<k>_val.csv       (TTA + T)  -> I4(b)
    """
    pred_dir = Path(pred_dir)
    for sd in seeds:
        cfg = _load_cfg(exp_id, sd, out_dir)
        model = _load_model(cfg, sd, out_dir)
        fn_v, y_v, lv = _view_logits(model, cfg, "val", method)
        fn_t, y_t, lt = _view_logits(model, cfg, "test", method)
        pv = inf.aggregate_views(lv, "prob")
        pt = inf.aggregate_views(lt, "prob")
        if temperature:
            # log(mean softmax) preserves the uncalibrated prediction exactly at T=1.
            zv = np.log(np.clip(pv, 1e-12, 1.0))
            zt = np.log(np.clip(pt, 1e-12, 1.0))
            T = inf.fit_temperature(zv, y_v)
            pv_T = inf.apply_temperature(zv, T)
            pt_T = inf.apply_temperature(zt, T)
        else:
            T, pv_T, pt_T = 1.0, pv, pt
        save_predictions(pred_dir / f"{exp_id}_seed{sd}_test.csv", fn_t, y_t, pt_T)
        save_predictions(pred_dir / f"{exp_id}uncal_seed{sd}_test.csv", fn_t, y_t, pt)
        save_predictions(pred_dir / f"{exp_id}_seed{sd}_val.csv", fn_v, y_v, pv_T)
        print(f"finalize {exp_id} seed{sd}: T={T:.3f}, đã ghi test/uncal/val predictions")
    return 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", help="exp_id để chạy inference_study")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--finalize", help="exp_id để finalize_test")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--method", default="hflip")
    args = ap.parse_args()
    if args.study:
        inference_study(args.study, args.seed)
    if args.finalize:
        finalize_test(args.finalize, tuple(args.seeds), args.method)
