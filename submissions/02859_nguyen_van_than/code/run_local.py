"""Run the complete lab locally, measuring CUDA settings and an epoch budget.

python code/run_local.py --minutes 60
All original fold CSVs/images remain intact. A short screening study and three
final/baseline seeds use the same training code as the Colab notebook.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLBACKEND", "Agg")
# Select the older cuDNN API before torch is imported; avoid cuDNN v8 FIND
# engine selection on this 4 GB Turing GPU. Record this in environment.json.
os.environ.setdefault("TORCH_CUDNN_V8_API_DISABLED", "1")
os.environ.setdefault("HF_HOME", str(REPO_ROOT / "local_runs" / "weights-cache"))
os.environ.setdefault("TORCH_HOME", str(REPO_ROOT / "local_runs" / "torch-cache"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch
import torchvision
import timm
import matplotlib.pyplot as plt
import model
import train
import build_notebook
from colab_fast import FastLab, HOUR_BACKBONES, read_json, write_json


class Tee:
    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, value):
        self.stream.write(value)
        self.log.write(value)
        self.log.flush()

    def flush(self):
        self.stream.flush()
        self.log.flush()


def cuda_settings(plan, img_size, output):
    """Exercise forward, backward and AdamW state on all real architectures.

    Benchmark FP32/FP16 instead of assuming AMP is faster on a GTX 1650.
    Select ONE common configuration for comparable backbone experiments.
    """
    report = []
    for amp, channels_last in ((False, True), (True, True), (False, False), (True, False)):
        batch = 64
        while batch >= 8:
            rows, failed = [], False
            for _, name in plan:
                network = optimizer = x = y = loss = None
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    torch.backends.cudnn.benchmark = False
                    train.set_seed(0)
                    torch.backends.cudnn.benchmark = False
                    network = model.build_model(name, pretrained=False, img_size=img_size).cuda().train()
                    if channels_last:
                        network.to(memory_format=torch.channels_last)
                    x = torch.randn(batch, 3, img_size, img_size, device="cuda")
                    if channels_last:
                        x = x.contiguous(memory_format=torch.channels_last)
                    y = torch.arange(batch, device="cuda") % 9
                    optimizer = torch.optim.AdamW(network.parameters(), lr=1e-4, fused=True)
                    scaler = torch.amp.GradScaler("cuda", enabled=amp)
                    times = []
                    for step in range(5):
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        optimizer.zero_grad(set_to_none=True)
                        with torch.autocast("cuda", enabled=amp, dtype=torch.float16):
                            loss = torch.nn.functional.cross_entropy(network(x), y)
                        if not torch.isfinite(loss).item():
                            raise RuntimeError("Nonfinite loss during CUDA preflight")
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                        torch.cuda.synchronize()
                        if step >= 2:
                            times.append(time.perf_counter() - start)
                    if not all(torch.isfinite(p).all().item() for p in network.parameters()):
                        raise RuntimeError("Nonfinite parameters during CUDA preflight")
                    rows.append(dict(backbone=name, step_s=float(np.median(times)),
                                     peak_mb=torch.cuda.max_memory_allocated() / 2**20))
                    print(f"CUDA probe: {name}, batch={batch}, amp={amp}, channels_last={channels_last}: "
                          f"{rows[-1]['step_s']:.3f}s/step, {rows[-1]['peak_mb']:.0f} MB", flush=True)
                except RuntimeError as error:
                    if not any(token in str(error) for token in ("out of memory", "engine", "Nonfinite")):
                        raise
                    print(f"CUDA probe rejected: {name}: {error}", flush=True)
                    failed = True
                    break
                finally:
                    del network, optimizer, x, y, loss
                    gc.collect()
                    torch.cuda.empty_cache()
            report.append(dict(batch_size=batch, amp=amp, channels_last=channels_last,
                               valid=not failed, models=rows))
            write_json(output, report)
            if not failed:
                break
            batch //= 2
    valid = [r for r in report if r["valid"]]
    if not valid:
        raise RuntimeError("No stable common CUDA configuration; see cuda_probe.json")
    selected = min(valid, key=lambda r: sum(m["step_s"] for m in r["models"]) / r["batch_size"])
    return {k: selected[k] for k in ("batch_size", "amp", "channels_last")}


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError("Run with .venv-gpu/Scripts/python.exe; this interpreter has no CUDA")
    torch.set_num_threads(4)
    workspace = args.root.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    # Copy the source snapshot; never train from changing files in the repo.
    shutil.copytree(REPO_ROOT / "code", workspace / "code", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__"))
    for name in ("eval.py", "README.md", "GUIDE.md", "RUBRIC.md", "SUBMISSION_README.md", "LAB_STATUS.md", "requirements.txt"):
        shutil.copy2(REPO_ROOT / name, workspace / name)
    if not (workspace / "images.zip").exists() and (REPO_ROOT / "images.zip").exists():
        os.link(REPO_ROOT / "images.zip", workspace / "images.zip")
    labels = workspace / "data" / "labels"
    labels.mkdir(parents=True, exist_ok=True)
    cached_labels = REPO_ROOT / "_smoke" / "deepweeds_labels"
    if cached_labels.exists():
        for file in cached_labels.glob("*.csv"):
            shutil.copy2(file, labels / file.name)
    plan_file = workspace / "local_plan.json"
    if args.reprobe and any((workspace / "runs").glob("*/seed*/summary.json")):
        raise ValueError("Completed training exists; reprobe in a new --root to preserve comparable configurations")
    if plan_file.exists() and not args.reprobe:
        settings = read_json(plan_file)["settings"]
    else:
        settings = cuda_settings(HOUR_BACKBONES, 128, workspace / "cuda_probe.json")
        settings.update(num_workers=args.workers, cudnn_benchmark=False, amp_dtype="float16")
        write_json(plan_file, dict(settings=settings, target_minutes=args.minutes))
    print("Selected common GPU settings:", settings, flush=True)
    if args.probe_only:
        return
    lab = FastLab(workspace, "hour", settings=settings)
    lab.workers = settings["num_workers"]
    import subprocess
    source_commit = subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True).strip()
    write_json(workspace / "environment.json", dict(python=sys.version, torch=torch.__version__,
               torchvision=torchvision.__version__, timm=timm.__version__, numpy=np.__version__,
               gpu=torch.cuda.get_device_name(), vram_gb=torch.cuda.get_device_properties(0).total_memory / 2**30,
               cudnn=torch.backends.cudnn.version(), cudnn_v8_disabled=os.environ["TORCH_CUDNN_V8_API_DISABLED"],
               profile="hour", source_repo="meth04/K4-DAY02-NguyenVanThan-02859", source_commit=source_commit,
               target_minutes=args.minutes, settings=settings))
    # Execute the actual notebook's EDA, pipeline checks and final analysis cells.
    nb = json.loads((workspace / "code/lab_day2.ipynb").read_text(encoding="utf-8"))
    cells = [cell["source"] for cell in nb["cells"] if cell["cell_type"] == "code"]
    started = time.perf_counter()
    namespace = dict(lab=lab, PROFILE="hour", torch=torch, np=np, pd=pd, plt=plt, time=time,
                     STARTED=started, write_json=write_json, display=lambda value: print(value.to_string(), flush=True))

    def execute(fragment):
        print(f"STAGE: {fragment}", flush=True)
        source = next(cell for cell in cells if fragment in cell)
        exec(compile(source, "lab_day2.ipynb", "exec"), namespace)
        plt.close("all")
        gc.collect()
        torch.cuda.empty_cache()

    execute("lab.prepare_data()")
    execute("checks =")
    execute("backbone_table =")
    execute("ablation_table =")
    plan = read_json(plan_file)
    if "final_epochs" not in plan:
        # Budget choices use ONLY training wall time and validation-selected models.
        final_epoch = float(lab.chosen["train_time_per_epoch_s"])
        recipe_epoch = float(read_json(lab.runs / lab.best["exp_id"] / "seed0" / "summary.json")["train_time_per_epoch_s"])
        final_runs = 3 if all(lab.best[k] == lab.cfg(backbone=lab.best["backbone"]).__dict__[k]
                             for k in ("init", "aug", "loss", "mix")) else 6
        elapsed = time.perf_counter() - started
        # Reserve 8 minutes for inference, calibration, test and product export.
        available = max(0, args.minutes * 60 - elapsed - 480)
        per_epoch_all_seeds = 3 * recipe_epoch + (3 * final_epoch if final_runs == 6 else 0)
        epochs = max(1, min(12, int(available / max(1, per_epoch_all_seeds))))
        plan.update(final_epochs=epochs, elapsed_before_final_s=elapsed,
                    expected_final_runs=final_runs, measured_epoch_s=final_epoch,
                    measured_recipe_epoch_s=recipe_epoch, estimated_all_seeds_epoch_s=per_epoch_all_seeds)
        write_json(plan_file, plan)
    lab.budget["final"] = plan["final_epochs"]
    print(f"FINAL BUDGET: {lab.budget['final']} epoch per seed; target {args.minutes} minutes", flush=True)
    execute("lab.train_final()")
    execute("lab.finals()")
    execute('if PROFILE != "smoke":\n    from eval import load_group')
    elapsed = time.perf_counter() - started
    write_json(workspace / "local_complete.json", dict(status="complete", elapsed_s=elapsed,
               archive=str(namespace["archive"]), target_minutes=args.minutes))
    print(f"COMPLETE: {elapsed/60:.1f} minutes. {namespace['archive']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "local_runs" / "hour")
    parser.add_argument("--minutes", type=float, default=60)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--reprobe", action="store_true")
    args = parser.parse_args()
    if args.minutes <= 0 or args.workers < 0:
        parser.error("minutes must be positive; workers must be nonnegative")
    args.root.mkdir(parents=True, exist_ok=True)
    with (args.root / "training.log").open("a", encoding="utf-8", buffering=1) as log:
        original_stdout, original_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = Tee(sys.stdout, log), Tee(sys.stderr, log)
        try:
            run(args)
        except Exception:
            traceback.print_exc()
            raise SystemExit(1)
        finally:
            sys.stdout, sys.stderr = original_stdout, original_stderr


if __name__ == "__main__":
    # Windows DataLoader uses spawn; protect all training under this guard.
    main()
