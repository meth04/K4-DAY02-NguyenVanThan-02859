"""Execute the notebook workflow on CPU with bounded training and real architectures.

Use the original images.zip and four original CSVs. Setup/download UI is replaced
with a local archive; all EDA, pipeline, model study, inference, final and export
cells execute as written. CPU test training uses 18/9/9 samples, one epoch,
and randomly initialized weights. It does not measure Colab GPU performance.

python code/verify_notebook.py --labels-dir _smoke/deepweeds_labels
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
import timm

import build_notebook
import model
from colab_fast import read_json
from verify_colab_fast import restore_cwd

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--profile", choices=("fast", "hour"), default="fast")
    args = parser.parse_args()
    labels = args.labels_dir.resolve()
    for name in ("labels", "train_subset0", "val_subset0", "test_subset0"):
        if not (labels / f"{name}.csv").exists():
            raise FileNotFoundError(f"Download the original {name}.csv into {labels}")
    if not (ROOT / "images.zip").exists():
        raise FileNotFoundError("Place original DeepWeeds images.zip at the repository root")
    torch.set_num_threads(2)
    build_notebook.build()
    nb = json.loads((ROOT / "code/lab_day2.ipynb").read_text(encoding="utf-8"))
    cells = [cell["source"] for cell in nb["cells"] if cell["cell_type"] == "code"]
    for index, source in enumerate(cells):
        compile(source, f"notebook-cell-{index}", "exec")
    original_build = model.build_model
    original_cwd = Path.cwd()
    temporary_root = ROOT / "_smoke"
    temporary_root.mkdir(exist_ok=True)
    executed = []
    start = time.perf_counter()

    def execute(fragment, namespace):
        source = next(cell for cell in cells if fragment in cell)
        print(f"EXECUTE {fragment}", flush=True)
        exec(compile(source, fragment, "exec"), namespace)
        plt.close("all")
        gc.collect()
        executed.append(fragment)

    with tempfile.TemporaryDirectory(dir=temporary_root) as tmp, restore_cwd(original_cwd):
        workspace = Path(tmp) / "runtime"
        workspace.mkdir()
        fixture = Path(tmp) / "source.zip"
        with ZipFile(fixture, "w") as archive:
            for name in ("code",):
                for file in (ROOT / name).glob("*.py"):
                    archive.write(file, "repo/" + str(file.relative_to(ROOT)).replace("\\", "/"))
            for name in ("eval.py", "README.md", "GUIDE.md", "RUBRIC.md", "SUBMISSION_README.md", "LAB_STATUS.md", "requirements.txt"):
                archive.write(ROOT / name, "repo/" + name)
        namespace = dict(sys=sys, os=os, json=json, Path=Path, torch=torch, torchvision=torchvision,
                         timm=timm, np=np, pd=pd, time=time, platform=__import__("platform"),
                         display=lambda value: print(f"DISPLAY {type(value).__name__}", flush=True))
        execute('PROFILE = "fast"', namespace)
        namespace.update(ROOT=str(workspace), SAVE_TO_DRIVE=False, PROFILE=args.profile)
        with patch("urllib.request.urlretrieve", side_effect=lambda url, destination: shutil.copy2(fixture, destination)):
            execute("archive_path =", namespace)
        with patch.object(torch.cuda, "get_device_name", return_value="CPU verification"), \
             patch.object(torch.cuda, "get_device_properties", return_value=type("GPU", (), {"total_memory": 4 * 2**30})()):
            execute("lab = FastLab", namespace)
        lab = namespace["lab"]
        assert lab.device.type == "cpu", "This bounded verifier is intended for CPU execution"
        os.link(ROOT / "images.zip", workspace / "images.zip")
        for file in labels.glob("*.csv"):
            shutil.copy2(file, lab.labels / file.name)
        execute("lab.prepare_data()", namespace)
        assert namespace["split_info"]["union"] == 17509
        assert len(namespace["split_info"]["label_audit"]["catalog_label_differences"]) == 1
        with patch.object(torch.nn.Module, "cuda", lambda module, *a, **kw: module), \
             patch.object(torch.Tensor, "cuda", lambda tensor, *a, **kw: tensor):
            execute("checks =", namespace)
        saved = read_json(lab.runs / "pipeline_checks.json")
        assert all(type(value) is bool for value in saved["checks"])
        assert saved["overfit_loss"] < 0.05
        # Select bounded verification data only inside the disposable workspace.
        for split, frame in zip(("train", "val", "test"), (namespace["train_df"], namespace["val_df"], namespace["test_df"])):
            subset = frame.groupby("Label", sort=True).head(2 if split == "train" else 1)
            subset.to_csv(lab.labels / f"{split}_subset0.csv", index=False)
        original_cfg = lab.cfg

        def bounded_cfg(**kwargs):
            return replace(original_cfg(**kwargs), epochs=1, batch_size=3, num_workers=0,
                           limit_train=18, limit_val=9, amp=False)

        def without_weight_download(name, pretrained=True, **kwargs):
            return original_build(name, pretrained=False, **kwargs)

        lab.cfg = bounded_cfg
        lab.workers = 0
        lab.budget = dict(screen=1, final=1)
        with patch.object(model, "build_model", side_effect=without_weight_download):
            execute("backbone_table =", namespace)
            assert len(namespace["backbone_table"]) == 5
            execute("ablation_table =", namespace)
            execute("lab.train_final()", namespace)
            assert len(namespace["inference_table"]) == 5
            execute("lab.finals()", namespace)
            execute('if PROFILE != "smoke":\n    from eval import load_group', namespace)
            # Rerun export/analysis cell: no duplicates in the resulting ZIP.
            execute('if PROFILE != "smoke":\n    from eval import load_group', namespace)
        archive_path = namespace["archive"]
        with ZipFile(archive_path) as archive:
            names = archive.namelist()
            assert len(names) == len(set(names))
            for required in ("code/lab_day2.ipynb", "curves/F01_confusion.png", "curves/F01_errors.png", "results.xlsx"):
                assert required in names, required
            assert not any(name.endswith(".pt") for name in names)
        with pd.ExcelFile(workspace / "results.xlsx") as workbook:
            counts = {name: len(pd.read_excel(workbook, name)) for name in workbook.sheet_names}
        assert counts["Backbones"] == 5 and counts["Training"] >= 5 and counts["Inference"] == 5
        assert counts["Final"] == 2 and counts["PerClass"] == 18
        for eid in ("F01", "T00"):
            for seed in (0, 1, 2):
                assert read_json(lab.runs / eid / f"seed{seed}" / "test_manifest.json")["status"] == "complete"
        print("PASS notebook cells: real 17,509-image preparation/EDA and overfit/JSON, all 5 architectures,", flush=True)
        print("     all 3 ablation axes + combination, 5 inference methods, 3-seed final/baseline,", flush=True)
        print("     evaluator CLI, seven workbook sheets, error figures and rerunnable ZIP export.", flush=True)
        print(f"CPU verification took {time.perf_counter() - start:.1f}s; pretrained download/Colab GPU/Drive UI were not exercised.")


if __name__ == "__main__":
    main()
