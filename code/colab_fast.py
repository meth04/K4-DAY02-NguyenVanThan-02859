"""Orchestrate a short, controlled DeepWeeds study. No test-based selection."""
from __future__ import annotations

import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

import numpy as np
import pandas as pd
import torch
import dataset as ds
import inference as inf
import model as md
import train as tr
import benchmark as bm
from eval import compute_metrics, save_predictions

PROFILES = {
    "fast": dict(screen=3, final=10),
    "full": dict(screen=12, final=15),
    "smoke": dict(screen=1, final=1),
}
BACKBONES = [
    ("B01", "resnet18"), ("B02", "resnet50"),
    ("B03", "convnext_tiny"), ("B04", "deit_tiny_patch16_224"),
    ("B05", "mobilenetv3_large_100"),
]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def config_key(cfg):
    return {k: v for k, v in asdict(cfg).items()
            if k not in {"images_dir", "labels_dir", "out_dir", "pred_dir", "curves_dir", "num_workers"}}


class FastLab:
    def __init__(self, root, profile="fast", backup=None):
        if profile not in PROFILES:
            raise ValueError(f"Profile phải thuộc {list(PROFILES)}")
        self.root = Path(root).resolve()
        self.profile = profile
        self.budget = PROFILES[profile]
        self.backup = Path(backup) / profile if backup else None
        self.runs = self.root / "runs"
        self.pred = self.root / "predictions"
        self.curves = self.root / "curves"
        self.labels = self.root / "data" / "labels"
        self.images = self.root / "images"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = "bfloat16" if self.device.type == "cuda" and torch.cuda.is_bf16_supported() else "float16"
        self.workers = min(4, os.cpu_count() or 2)
        for path in (self.runs, self.pred, self.curves, self.labels, self.images):
            path.mkdir(parents=True, exist_ok=True)
        if self.backup and self.backup.exists():
            for name in ("runs", "predictions", "curves"):
                if (self.backup / name).exists():
                    shutil.copytree(self.backup / name, self.root / name, dirs_exist_ok=True)
        os.chdir(self.root)
        source_hashes = {name: hashlib.sha256((self.root / "code" / name).read_bytes()).hexdigest()
                         for name in ("train.py", "dataset.py", "model.py", "losses.py", "colab_fast.py")}
        manifest = self.runs / "source_manifest.json"
        if manifest.exists() and read_json(manifest) != source_hashes:
            raise ValueError("Code khác lần chạy đã lưu; đổi tên SESSION để tránh trộn kết quả.")
        write_json(manifest, source_hashes)

    def persist(self):
        if not self.backup:
            return
        self.backup.mkdir(parents=True, exist_ok=True)
        # Only changed files, including best checkpoints; images stay on /content.
        for name in ("runs", "predictions", "curves", "eval_out", "logs"):
            source = self.root / name
            if not source.exists():
                continue
            for src in source.rglob("*"):
                if not src.is_file():
                    continue
                dst = self.backup / name / src.relative_to(source)
                if not dst.exists() or src.stat().st_size != dst.stat().st_size or src.stat().st_mtime > dst.stat().st_mtime + 1:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
        for name in ("results.xlsx", "report.md", "selection.json", "environment.json"):
            if (self.root / name).exists():
                shutil.copy2(self.root / name, self.backup / name)

    def prepare_data(self):
        zip_path = self.root / "images.zip"
        cached = self.backup.parent / "images.zip" if self.backup else None
        if not zip_path.exists():
            if cached and cached.exists():
                shutil.copy2(cached, zip_path)
            else:
                print("Tải images.zip (~490 MB)...", flush=True)
                temporary = zip_path.with_suffix(".download")
                urlretrieve("https://zenodo.org/records/7939060/files/images.zip?download=1", temporary)
                temporary.replace(zip_path)
        with zip_path.open("rb") as stream:
            h = hashlib.md5()
            for chunk in iter(lambda: stream.read(4 << 20), b""):
                h.update(chunk)
        if h.hexdigest() != "b7b30f96d466fba86016aa5a26606e0f":
            raise ValueError("images.zip sai MD5; thay file tải lỗi rồi chạy lại cell.")
        if not (self.images / ".extracted").exists():
            with ZipFile(zip_path) as archive:
                archive.extractall(self.images)
            (self.images / ".extracted").touch()
        if cached and not cached.exists():
            cached.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(zip_path, cached)
        base = "https://raw.githubusercontent.com/AlexOlsen/DeepWeeds/master/labels"
        for name in ("labels", "train_subset0", "val_subset0", "test_subset0"):
            path = self.labels / f"{name}.csv"
            if not path.exists():
                temp = path.with_suffix(".download")
                urlretrieve(f"{base}/{name}.csv", temp)
                temp.replace(path)
        frames = ds.load_split(self.labels)
        info = ds.check_split(*frames, self.images)
        write_json(self.runs / "split_check.json", info)
        full = pd.read_csv(self.labels / "labels.csv")
        observed = full.Label.value_counts().sort_index().reindex(range(9), fill_value=0)
        union = pd.concat(frames, ignore_index=True)
        assert not union.Filename.duplicated().any(), "CSV chứa ảnh trùng"
        ref = full.set_index("Filename").Label
        assert union.Filename.isin(ref.index).all()
        assert np.array_equal(ref.loc[union.Filename].to_numpy(), union.Label.to_numpy())
        expected = [1125, 1064, 1031, 1022, 1062, 1009, 1074, 1016, 9106]
        print(pd.DataFrame({"lớp": ds.CLASS_NAMES, "đếm thật": observed.values, "Table 1": expected}))
        return frames, info

    def cfg(self, **kwargs):
        base = dict(images_dir=str(self.images), labels_dir=str(self.labels), out_dir=str(self.runs),
                    pred_dir=str(self.pred), curves_dir=str(self.curves), num_workers=self.workers,
                    amp=self.device.type == "cuda", amp_dtype=self.dtype, channels_last=True,
                    fused_optimizer=True, save_last=False, batch_size=64,
                    epochs=self.budget["screen"], save_test_predictions=False)
        if self.profile == "smoke":
            base.update(limit_train=64, limit_val=32, batch_size=16)
        base.update(kwargs)
        return tr.Config(**base)

    def run_exp(self, cfg, comparison="TS00"):
        folder = tr.run_dir(cfg)
        complete = folder / "summary.json"
        if complete.exists() and (folder / "best.pt").exists():
            old = tr.Config(**read_json(folder / "config.json"))
            if config_key(old) != config_key(cfg):
                raise ValueError(f"{cfg.exp_id}/seed{cfg.seed} đã chạy cấu hình khác; dùng thư mục phiên mới.")
            # Remap paths when restoring a completed run on another runtime.
            write_json(folder / "config.json", asdict(cfg))
            print(f"Dùng lại {cfg.exp_id}/seed{cfg.seed}")
            return read_json(complete)
        start = time.perf_counter()
        try:
            result = tr.run(cfg)
        finally:
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        result.update(comparison_baseline=comparison, wall_time_s=time.perf_counter() - start)
        write_json(complete, result)
        self.persist()
        return result

    def alias(self, source, cfg, source_seed=0, comparison="TS00"):
        target = tr.run_dir(cfg)
        if (target / "summary.json").exists():
            return self.run_exp(cfg)
        origin = self.runs / source / f"seed{source_seed}"
        old_cfg = tr.Config(**read_json(origin / "config.json"))
        old_key, new_key = config_key(old_cfg), config_key(cfg)
        old_key.pop("exp_id"); new_key.pop("exp_id")
        if old_key != new_key:
            raise ValueError("Chỉ dùng lại model khi mọi cấu hình huấn luyện giống nhau.")
        shutil.copytree(origin, target, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("test_manifest.json", "test_logits.npy", "test_labels.npy", "test_filenames.txt"))
        write_json(target / "config.json", asdict(cfg))
        summary = read_json(target / "summary.json")
        summary.update(exp_id=cfg.exp_id, comparison_baseline=comparison, provenance=f"Dùng lại {source}/seed{source_seed}; cấu hình huấn luyện giống hệt")
        write_json(target / "summary.json", summary)
        shutil.copy2(self.pred / f"{source}_seed{source_seed}_val.csv", self.pred / f"{cfg.exp_id}_seed{cfg.seed}_val.csv")
        tr.plot_curves(pd.read_csv(target / "history.csv").to_dict("records"), tr.curve_path(cfg), f"{cfg.exp_id} · {cfg.backbone} · seed{cfg.seed}")
        self.persist()
        return summary

    def backbones(self):
        rows, meta, latency = [], {}, {}
        for eid, name in BACKBONES:
            rows.append(self.run_exp(self.cfg(exp_id=eid, backbone=name)))
            model = md.build_model(name, pretrained=False).to(self.device).eval()
            meta[name] = dict(params_m=md.count_params(model), gmac=md.count_gmacs(model),
                              weight_tag=md.pretrained_tag(model),
                              gmac_method="Conv2d + Linear; excludes attention matmuls (lower bound)")
            latency[eid] = bm.latency_report(model, 1, 224, "fp32", self.device.type, 10, 50)
            del model
            gc.collect()
        write_json(self.runs / "backbone_meta.json", meta)
        write_json(self.runs / "backbone_latency.json", latency)
        table = pd.DataFrame(rows).sort_values(["val_macro_f1", "train_time_per_epoch_s"], ascending=[False, True])
        self.chosen = table.iloc[0].to_dict()
        self.persist()
        return table

    def ablations(self):
        baseline = self.cfg(exp_id="TS00", backbone=self.chosen["backbone"])
        rows = [self.alias(self.chosen["exp_id"], baseline)]
        trials = [("T01", dict(init="frozen")), ("T03", dict(aug="color")),
                  ("T07", dict(loss="ls", label_smoothing=0.1))]
        if self.profile == "full":
            trials += [("T02", dict(init="scratch")), ("T06", dict(mix="cutmix")),
                       ("T08", dict(loss="focal", focal_gamma=2.0))]
        for eid, change in trials:
            rows.append(self.run_exp(self.cfg(exp_id=eid, backbone=baseline.backbone, **change)))
        rows.append(self.run_exp(self.cfg(exp_id="T14", backbone=baseline.backbone,
                                         aug="color", loss="ls", label_smoothing=0.1)))
        table = pd.DataFrame(rows).sort_values("val_macro_f1", ascending=False)
        winning_id = table.iloc[0].exp_id
        self.best = read_json(self.runs / winning_id / "seed0" / "config.json")
        self.persist()
        return table.assign(delta_vs_TS00=table.val_macro_f1 - rows[0]["val_macro_f1"])

    def train_final(self):
        if self.profile == "smoke":
            print("SMOKE: dừng trước chung kết/test; kết quả không dùng để nộp.")
            return
        recipe = {k: self.best[k] for k in ("backbone", "init", "aug", "loss", "label_smoothing", "focal_gamma", "mix")}
        self.run_exp(self.cfg(exp_id="F01", seed=0, epochs=self.budget["final"], **recipe))
        # No test prediction yet. Inference is selected on F01 seed0 validation.

    def load_model(self, eid, seed):
        cfg = tr.Config(**read_json(self.runs / eid / f"seed{seed}" / "config.json"))
        model = md.build_model(cfg.backbone, pretrained=False, init=cfg.init).to(self.device).eval()
        state = torch.load(self.runs / eid / f"seed{seed}" / "best.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state["state_dict"])
        if cfg.channels_last:
            model.to(memory_format=torch.channels_last)
        return cfg, model

    def predict(self, model, cfg, split, flipped=False, amp=True):
        frame = ds.load_split(self.labels)[{"train": 0, "val": 1, "test": 2}[split]]
        loader = ds.make_loader(frame, self.images, ds.build_transforms(False, cfg.img_size),
                                cfg.batch_size, False, num_workers=self.workers, seed=cfg.seed)
        view = inf.view_hflip if flipped else None
        with torch.autocast(self.device.type, enabled=amp and self.device.type == "cuda",
                            dtype=torch.bfloat16 if self.dtype == "bfloat16" else torch.float16):
            return inf.predict_logits(model, loader, self.device, view)

    def inference_study(self):
        if self.profile == "smoke":
            return pd.DataFrame()
        cfg, model = self.load_model("F01", 0)
        fn, yt, identity = self.predict(model, cfg, "val")
        fn2, yt2, flipped = self.predict(model, cfg, "val", flipped=True)
        assert fn == fn2 and np.array_equal(yt, yt2)
        p0 = inf.aggregate_views([identity])
        ph = inf.aggregate_views([identity, flipped])
        pl = inf.aggregate_views([identity, flipped], "logit")
        precision = "amp_bf16" if self.dtype == "bfloat16" else "amp"
        lat0 = bm.latency_report(model, 1, 224, precision, self.device.type, 10, 50)
        lat2 = bm.tta_latency(model, 2, 1, 224, precision, self.device.type, 10, 50)
        rows = []

        def add(eid, method, p, lat, note="", views=1):
            metrics = compute_metrics(yt, p.argmax(1), p)
            rows.append({"exp_id": eid, "phương pháp": method, "mô hình/checkpoint": "F01/seed0",
                         "K": views, "macro-F1 val": metrics["macro_f1"], "top-1 val": metrics["top1"],
                         "ECE val": metrics["ece"], "p50 (ms)": lat["p50"], "p95 (ms)": lat["p95"],
                         "p99 (ms)": lat["p99"], "ảnh/s": lat["images_per_s"],
                         "chi phí so với I00": lat["p50"] / lat0["p50"], "ghi chú": note})

        add("I00", "1-view AMP", p0, lat0)
        add("I01", "hflip / mean probability", ph, lat2, views=2)
        add("I03", "hflip / mean logits", pl, lat2, views=2)
        T = inf.fit_temperature(np.log(np.clip(p0, 1e-12, 1)), yt)
        add("I07", "temperature scaling", inf.apply_temperature(np.log(np.clip(p0, 1e-12, 1)), T), lat0,
            note=f"T={T:.4f}; fitted on val; calibration overhead excluded from model timing")
        # Fifth method uses predictions already produced during backbone training.
        ranked = sorted(BACKBONES, key=lambda item: read_json(self.runs / item[0] / "seed0" / "summary.json")["val_macro_f1"], reverse=True)[:2]
        frames = [pd.read_csv(self.pred / f"{eid}_seed0_val.csv").set_index("Filename").loc[fn] for eid, _ in ranked]
        probs = inf.ensemble_probs([frame[[f"p{i}" for i in range(9)]].to_numpy() for frame in frames])
        add("I05", "ensemble 2 backbones", probs, lat0,
            note=f"{[r[0] for r in ranked]}; latency not measured for this ensemble", views=2)
        rows[-1]["mô hình/checkpoint"] = "+".join(f"{eid}/seed0" for eid, _ in ranked)
        for key in ("p50 (ms)", "p95 (ms)", "p99 (ms)", "ảnh/s", "chi phí so với I00"):
            rows[-1][key] = ""
        write_json(self.runs / "inference.json", rows)
        # Select deployable method on validation; prefer one view for gains below 0.001.
        options = {"identity": p0, "hflip_prob": ph, "hflip_logit": pl}
        scores = {name: compute_metrics(yt, p.argmax(1), p)["macro_f1"] for name, p in options.items()}
        winner = max(scores, key=scores.get)
        self.method = winner if scores[winner] > scores["identity"] + 0.001 else "identity"
        self.final_lat = lat0 if self.method == "identity" else lat2
        write_json(self.root / "selection.json", dict(recipe=self.best, method=self.method, val_scores=scores,
                                                       profile=self.profile, seeds=[0, 1, 2]))
        # The selected model precision is autocast float16, matching prediction on T4.
        write_json(self.runs / "latency.json", [{"cấu hình": f"F01 {self.method}", "GPU": self.final_lat["gpu"],
                   "dtype": precision, "batch": 1, "gộp BN": "không", "p50": self.final_lat["p50"],
                   "p95": self.final_lat["p95"], "p99": self.final_lat["p99"],
                   "ảnh/s": self.final_lat["images_per_s"], "n_iters": 50}])
        del model
        self.persist()
        return pd.DataFrame(rows)

    def finalize_one(self, eid, seed, method, calibrate):
        folder = self.runs / eid / f"seed{seed}"
        manifest = folder / "test_manifest.json"
        expected = dict(method=method, calibrate=calibrate)
        if manifest.exists():
            saved = read_json(manifest)
            if any(saved[k] != v for k, v in expected.items()):
                raise ValueError("Test đã chốt phương pháp khác; không dùng lại test để chọn cấu hình.")
            if saved.get("status") != "complete":
                raise RuntimeError("Test từng bị ngắt; kiểm tra file dự đoán và manifest trước khi tiếp tục.")
            if not (self.pred / f"{eid}_seed{seed}_test.csv").exists():
                raise RuntimeError("Manifest complete nhưng thiếu predictions; khôi phục file đã lưu từ Drive.")
            print(f"Dùng lại dự đoán test {eid}/seed{seed}")
            return
        cfg, model = self.load_model(eid, seed)
        fnv, yv, zv = self.predict(model, cfg, "val")
        views = [zv]
        if method != "identity":
            fn2, y2, z2 = self.predict(model, cfg, "val", flipped=True)
            assert fnv == fn2 and np.array_equal(yv, y2)
            views.append(z2)
        space = "logit" if method == "hflip_logit" else "prob"
        pv = inf.aggregate_views(views, space)
        T = inf.fit_temperature(np.log(np.clip(pv, 1e-12, 1)), yv) if calibrate else 1.0
        # Written BEFORE accessing test: no accidental repeat after a disconnected runtime.
        write_json(manifest, dict(**expected, status="started", temperature=T))
        self.persist()
        fnt, yt, zt = self.predict(model, cfg, "test")
        test_views = [zt]
        if method != "identity":
            fn2, y2, z2 = self.predict(model, cfg, "test", flipped=True)
            assert fnt == fn2 and np.array_equal(yt, y2)
            test_views.append(z2)
        pt = inf.aggregate_views(test_views, space)
        save_predictions(self.pred / f"{eid}_seed{seed}_test.csv", fnt, yt,
                         inf.apply_temperature(np.log(np.clip(pt, 1e-12, 1)), T))
        save_predictions(self.pred / f"{eid}_seed{seed}_val.csv", fnv, yv,
                         inf.apply_temperature(np.log(np.clip(pv, 1e-12, 1)), T))
        if calibrate:
            save_predictions(self.pred / f"{eid}uncal_seed{seed}_test.csv", fnt, yt, pt)
        write_json(manifest, dict(**expected, status="complete", temperature=T, n_test=len(yt)))
        del model
        gc.collect()
        self.persist()

    def finals(self):
        if self.profile == "smoke":
            return
        recipe = {k: self.best[k] for k in ("backbone", "init", "aug", "loss", "label_smoothing", "focal_gamma", "mix")}
        # Train all remaining models before reading any test outcome.
        for seed in (0, 1, 2):
            final_cfg = self.cfg(exp_id="F01", seed=seed, epochs=self.budget["final"], **recipe)
            self.run_exp(final_cfg)
            baseline_cfg = self.cfg(exp_id="T00", seed=seed, backbone=recipe["backbone"], epochs=self.budget["final"])
            final_key, baseline_key = config_key(final_cfg), config_key(baseline_cfg)
            final_key.pop("exp_id"); baseline_key.pop("exp_id")
            if final_key == baseline_key:
                self.alias("F01", baseline_cfg, source_seed=seed, comparison="T00")
            else:
                self.run_exp(baseline_cfg, comparison="T00")
        for seed in (0, 1, 2):
            self.finalize_one("F01", seed, self.method, True)
            self.finalize_one("T00", seed, "identity", False)

    def products(self):
        if self.profile == "smoke":
            print("SMOKE không sinh bảng chung kết/test.")
            return
        def command(*args):
            subprocess.run([sys.executable, "-X", "utf8", *map(str, args)], check=True)
        command("eval.py", "score", "--pred", self.pred / "F01_seed*_test.csv",
                "--test-csv", self.labels / "test_subset0.csv", "--labels", self.labels / "labels.csv",
                "--tag", "F01", "--out", "eval_out")
        command("eval.py", "grade", "--final", self.pred / "F01_seed*_test.csv",
                "--baseline", self.pred / "T00_seed*_test.csv", "--uncal", self.pred / "F01uncal_seed*_test.csv",
                "--final-val", self.pred / "F01_seed*_val.csv", "--test-csv", self.labels / "test_subset0.csv",
                "--val-csv", self.labels / "val_subset0.csv", "--labels", self.labels / "labels.csv",
                "--latency-p95-ms", self.final_lat["p95"], "--out", "eval_out")
        command("code/make_results.py", "--labels-dir", self.labels)
        command("code/make_report.py", "--labels-dir", self.labels)
        command("code/build_notebook.py")
        # Export small logs separately; keep checkpoints in Drive backup, outside submission zip.
        for src in self.runs.glob("*/seed*/*"):
            if src.suffix in {".json", ".csv", ".txt"}:
                dst = self.root / "logs" / src.relative_to(self.runs)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
        with (self.root / "report.md").open("a", encoding="utf-8") as stream:
            stream.write(f"\n## Ngân sách và lựa chọn tự động\n\nProfile `{self.profile}`: khảo sát "
                         f"{self.budget['screen']} epoch, chung kết {self.budget['final']} epoch; 3 seed. "
                         "TS00 dùng lại đúng lần chạy backbone cùng công thức, không train thêm. "
                         "Sàng lọc ngắn có thể xếp hạng khác khi train lâu; không đảm bảo ngưỡng điểm chất lượng. "
                         "GMAC là tổng Conv2d + Linear, chưa tính attention matmul. "
                         "Ablation dùng TS00 cùng ngân sách; T00 là mốc chung kết với cùng ngân sách F01. "
                         "Khảo sát chỉ có 1 seed nên chưa thể kết luận cải thiện nhỏ vượt nhiễu. "
                         f"Suy luận chọn trên val: `{self.method}`, hiệu chuẩn T riêng mỗi seed trên val.\n")
        self.persist()
        archive = self.root / "deepweeds_submission.zip"
        with ZipFile(archive, "w") as out:
            for name in ("code", "logs", "predictions", "curves", "eval_out"):
                for file in (self.root / name).rglob("*"):
                    if file.is_file() and file.suffix not in {".pt", ".pyc"}:
                        out.write(file, str(file.relative_to(self.root)))
            for name in ("eval.py", "results.xlsx", "report.md", "selection.json", "environment.json",
                         "SUBMISSION_README.md", "LAB_STATUS.md", "README.md", "GUIDE.md", "RUBRIC.md", "requirements.txt"):
                if (self.root / name).exists():
                    out.write(self.root / name, name)
        if self.backup:
            shutil.copy2(archive, self.backup / archive.name)
        print("Sản phẩm:", archive, "| report.md là bản nháp cần bổ sung phân tích lỗi/kết luận.")
        return archive
