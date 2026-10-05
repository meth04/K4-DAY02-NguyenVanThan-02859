"""Validate published deliverables offline, without torch/GPU/checkpoints.

Run from repo root: python code/verify_submission.py
Run from submission: python code/verify_submission.py --submission .
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=Path,
                        default=Path(__file__).resolve().parents[1] / "submissions/02859_nguyen_van_than")
    args = parser.parse_args()
    root = args.submission.resolve()
    spec = importlib.util.spec_from_file_location("submission_evaluator", root / "eval.py")
    ev = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = ev
    spec.loader.exec_module(ev)
    read = lambda p: json.loads(p.read_text(encoding="utf-8"))
    manifest = read(root / "manifest.json")
    files = {path.relative_to(root).as_posix() for path in root.rglob("*")
             if path.is_file() and "__pycache__" not in path.parts}
    assert files == set(manifest) | {"manifest.json"}, "Extra or missing artifacts"
    for name, item in manifest.items():
        path = root / name
        assert path.stat().st_size == item["bytes"] and hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"], name
    labels = root / "evidence/labels"
    parts = [pd.read_csv(labels / f"{split}_subset0.csv") for split in ("train", "val", "test")]
    assert [len(p) for p in parts] == [10501,3501,3507]
    union = pd.concat(parts)
    assert union.Filename.is_unique and len(union) == 17509
    assert set(union.Filename) == set(pd.read_csv(labels / "labels.csv").Filename)
    for path in (root / "predictions").glob("*.csv"):
        split = "test" if path.stem.endswith("_test") else "val"
        ev.check_against_csv(ev.read_pred(path), str(labels / f"{split}_subset0.csv"), split)
    for path in (root / "evidence/extras").glob("V_*_val.csv"):
        ev.check_against_csv(ev.read_pred(path), str(labels / "val_subset0.csv"), "val")
    book = pd.read_excel(root / "results.xlsx", sheet_name=None)
    assert set(book) == {"Backbones","Training","Inference","Final","PerClass","Latency","Summary","Robustness"}
    assert len(book["Backbones"]) == 5 and len(book["Inference"]) == 5
    final = book["Final"]
    for exp in ("F01", "T00"):
        for split in ("test", "val"):
            group = ev.load_group(str(root / f"predictions/{exp}_seed*_{split}.csv"),
                                  str(labels / f"{split}_subset0.csv"), ref_what=split)
            assert group.seeds == [0,1,2]
            if split == "test":
                saved = read(root / f"evidence/evaluation/{exp}_summary.json")
                for key in ev.SCALARS:
                    np.testing.assert_allclose([saved[key]["mean"],saved[key]["std"]], group.summary[key], atol=1e-12)
            for index, seed in enumerate(group.seeds):
                row = final[(final.exp_id == exp) & (final.seed == seed)].iloc[0]
                np.testing.assert_allclose(row[f"macro-F1 {split}"], group.metrics[index]["macro_f1"], atol=1e-12)
                if split == "test":
                    np.testing.assert_allclose(row["top-1 test"], group.metrics[index]["top1"], atol=1e-12)
                    np.testing.assert_allclose(row["ECE test"], group.metrics[index]["ece"], atol=1e-12)
        for seed in (0,1,2):
            manifest = read(root / f"evidence/runs/{exp}/seed{seed}/test_manifest.json")
            assert manifest["status"] == "complete" and manifest["n_test"] == 3507
            cfg = read(root / f"evidence/runs/{exp}/seed{seed}/config.json")
            assert cfg["seed"] == seed and cfg["epochs"] == 12
            assert (root / f"curves/{exp}_seed{seed}.png").exists()
    core = read(root / "evidence/runs/source_manifest.json")
    for name, expected in core.items():
        assert hashlib.sha256((root / "code" / name).read_bytes()).hexdigest() == expected, name
    baseline = read(root / "evidence/runs/TS00/seed0/config.json")
    for exp, expected in [("T01", {"init"}), ("T03", {"aug"}), ("T07", {"loss","label_smoothing"}), ("T14", {"aug","loss","label_smoothing"})]:
        cfg = read(root / f"evidence/runs/{exp}/seed0/config.json")
        changes = {key for key in cfg if cfg[key] != baseline[key]} - {"exp_id"}
        assert changes == expected, (exp, changes)
    for path in (root / "evidence/runs").glob("*/seed0/summary.json"):
        summary = read(path)
        assert list((root / "curves").glob(f"{summary['exp_id']}_*.png")), path
        history = pd.read_csv(path.parent / "history.csv")
        assert {"train_loss","val_loss","val_macro_f1"} <= set(history)
    latency = book["Latency"]
    assert (latency.batch == 1).all() and (latency.warmup >= 10).all() and (latency.n_iters >= 50).all()
    robustness = book["Robustness"]
    for _, row in robustness.iterrows():
        prediction = ev.read_pred(root / f"evidence/extras/V_{row.condition}_seed0_val.csv")
        metrics = ev.compute_metrics(prediction.y_true, prediction.y_pred, prediction.probs)
        np.testing.assert_allclose([row.macro_f1,row.top1,row.ece_after],
                                   [metrics["macro_f1"],metrics["top1"],metrics["ece"]], atol=1e-12)
    cam = read(root / "evidence/extras/gradcam_val.json")
    assert len(cam) == 6 and all(row["gradient_l1"] > 0 and row["cam_max"] == 1 for row in cam)
    for path in (root / "code").glob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    notebook = read(root / "code/lab_day2.ipynb")
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            source = cell["source"]
            source = "".join(source) if isinstance(source, list) else source
            ast.parse(source)
            assert "SOURCE_BUNDLE" not in source
    if notebook.get("metadata", {}).get("saved_results"):
        import nbformat
        nbformat.validate(nbformat.from_dict(notebook))
        code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        assert len(code_cells) == 13
        assert [cell["execution_count"] for cell in code_cells] == list(range(1,14))
        assert all(cell["outputs"] for cell in code_cells)
        assert not any(output["output_type"] == "error" for cell in code_cells for output in cell["outputs"])
        assert sum("image/png" in output.get("data", {}) for cell in code_cells for output in cell["outputs"]) == 22
    report = (root / "report.md").read_text(encoding="utf-8")
    assert "TODO" not in report
    for path in root.glob("*.md"):
        for target in re.findall(r"\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
            if not target.startswith(("https://", "http://", "#")):
                assert (path.parent / target.split("#")[0]).exists(), (path, target)
    grade = read(root / "evidence/evaluation/grade_I.json")
    assert grade["total"] == 10 and grade["max_scored"] == 20 and not grade["warnings"]
    print(f"PASS: {len(files)} artifacts, hashes, fold0, all predictions, 8 sheets, 3 seeds, ablations, timings, robustness, Grad-CAM, code/notebook and links.")


if __name__ == "__main__":
    main()
