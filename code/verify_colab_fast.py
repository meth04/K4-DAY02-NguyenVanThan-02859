"""Offline regression checks for the accelerated pipeline (CPU, synthetic images)."""
import json
import os
import shutil
import tempfile
from pathlib import Path
from contextlib import contextmanager
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from PIL import Image

import model
import train
import dataset
from colab_fast import FastLab, read_json, write_json, audit_catalog_labels
from eval import load_group


@contextmanager
def restore_cwd(path):
    try:
        yield
    finally:
        os.chdir(path)


def main():
    torch.set_num_threads(2)
    root = Path(__file__).resolve().parent.parent
    original_cwd = Path.cwd()
    catalog = pd.DataFrame({"Filename": ["20170714-110407-3.jpg", "b.jpg", "c.jpg"], "Label": [1, 2, 3]})
    train_frame = pd.DataFrame({"Filename": ["20170714-110407-3.jpg"], "Label": [0]})
    frames = [train_frame, catalog.iloc[1:2].copy(), catalog.iloc[2:3].copy()]
    original_frames = [frame.copy() for frame in frames]
    audit = audit_catalog_labels(catalog, frames)
    assert len(audit["catalog_label_differences"]) == 1
    for original, current in zip(original_frames, frames):
        pd.testing.assert_frame_equal(original, current)
    unknown = [frame.copy() for frame in frames]
    unknown[1].loc[:, "Label"] = 4
    try:
        audit_catalog_labels(catalog, unknown)
    except ValueError:
        pass
    else:
        raise AssertionError("Unexpected label disagreement must still fail")
    with patch.object(model.timm, "create_model", wraps=model.timm.create_model) as create:
        m = model.build_model("resnet18", pretrained=False)
        assert create.call_args.kwargs["pretrained"] is False
    # Analytic operation count covers grouped convolutions and token-wise Linear.
    toy = torch.nn.Sequential(torch.nn.Conv2d(3, 6, 3, groups=3),
                              torch.nn.Flatten(), torch.nn.Linear(6 * 6 * 6, 9))
    expected_mac = (9 * 1 * 6 * 6 * 6 + 6 * 6 * 6 * 9) / 1e9
    assert abs(model.count_gmacs(toy, 8) - expected_mac) < 1e-12
    del m
    try:
        with tempfile.TemporaryDirectory(dir=root / "_smoke") as tmp, restore_cwd(original_cwd):
            work = Path(tmp)
            json_path = work / "json_scalars.json"
            write_json(json_path, {"checks": [np.bool_(True), np.bool_(False)], "count": np.int64(3),
                                  "loss": np.float32(0.25), "array": np.array([1, 2]),
                                  "tensor": torch.tensor([3, 4]), "path": work})
            decoded = read_json(json_path)
            assert decoded["checks"] == [True, False] and decoded["count"] == 3
            assert decoded["loss"] == 0.25 and decoded["array"] == [1, 2] and decoded["tensor"] == [3, 4]
            try:
                write_json(json_path, {"unsupported": object()})
            except TypeError:
                pass
            else:
                raise AssertionError("Unsupported values must still raise an error")
            assert read_json(json_path) == decoded, "Failed JSON write must preserve the previous file"
            shutil.copytree(root / "code", work / "code", ignore=shutil.ignore_patterns("__pycache__", "*.ipynb"))
            for name in ("eval.py", "README.md", "GUIDE.md", "RUBRIC.md", "SUBMISSION_README.md", "LAB_STATUS.md", "requirements.txt"):
                shutil.copy2(root / name, work / name)
            lab = FastLab(work, "smoke")
            # A failed pre-training setup must not lock the user out after a code fix.
            manifest = lab.runs / "source_manifest.json"
            manifest.write_text(json.dumps({"old_failed_setup": True}))
            lab = FastLab(work, "smoke")
            assert "old_failed_setup" not in read_json(manifest)
            lab.workers = 0
            rng = np.random.default_rng(0)
            for split, count in (("train", 18), ("val", 9), ("test", 9)):
                rows = []
                for index in range(count):
                    name = f"{split}{index}.jpg"
                    Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)).save(lab.images / name)
                    rows.append(dict(Filename=name, Label=index % 9))
                pd.DataFrame(rows).to_csv(lab.labels / f"{split}_subset0.csv", index=False)
            pd.DataFrame({"Label": range(9), "Species": dataset.CLASS_NAMES}).to_csv(lab.labels / "labels.csv", index=False)
            cfg = lab.cfg(exp_id="F01", backbone="resnet18", init="scratch", img_size=32,
                          batch_size=9, limit_train=18, limit_val=9, num_workers=0)
            result = lab.run_exp(cfg)
            saved_manifest = manifest.read_text()
            manifest.write_text(json.dumps({"old_trained_code": True}))
            try:
                FastLab(work, "smoke")
            except ValueError:
                pass
            else:
                raise AssertionError("Changed code must still reject existing training runs")
            manifest.write_text(saved_manifest)
            assert result["epochs_run"] == 1
            assert not (lab.pred / "F01_seed0_test.csv").exists()
            assert not (lab.runs / "F01" / "seed0" / "last.pt").exists()
            with patch.object(train, "run", side_effect=AssertionError("must reuse completed run")):
                assert lab.run_exp(cfg)["val_macro_f1"] == result["val_macro_f1"]
            # Verify no test is touched until validation calibration is complete;
            # T=1 leaves averaged probabilities unchanged, and rerun skips test.
            calls = []
            original_predict = lab.predict

            def tracked_predict(*args, **kwargs):
                calls.append((args[2], kwargs.get("flipped", False)))
                return original_predict(*args, **kwargs)

            with patch.object(lab, "predict", side_effect=tracked_predict):
                lab.finalize_one("F01", 0, "hflip_prob", True)
                lab.finalize_one("F01", 0, "hflip_prob", True)
            assert calls == [("val", False), ("val", True), ("test", False), ("test", True)], calls
            calibrated = load_group(str(lab.pred / "F01_seed0_test.csv"), str(lab.labels / "test_subset0.csv"), ref_what="test")
            uncalibrated = load_group(str(lab.pred / "F01uncal_seed0_test.csv"), str(lab.labels / "test_subset0.csv"), ref_what="test")
            assert len(calibrated.preds) == 1
            assert calibrated.metrics[0]["top1"] == uncalibrated.metrics[0]["top1"]
            assert read_json(lab.runs / "F01" / "seed0" / "test_manifest.json")["status"] == "complete"
            with patch.object(lab, "predict", side_effect=AssertionError("must not touch test")):
                try:
                    lab.finalize_one("F01", 0, "identity", True)
                except ValueError:
                    pass
                else:
                    raise AssertionError("must reject changed method after test")
            # Frozen backbone: BN running stats stay fixed, classifier stays train.
            frozen = model.build_model("resnet18", pretrained=False)
            model.freeze_backbone(frozen)
            bn = next(module for module in frozen.modules() if isinstance(module, torch.nn.BatchNorm2d))
            before = bn.running_mean.clone()
            loader = [(torch.randn(4, 3, 32, 32), torch.tensor([0, 1, 2, 3]), ["a", "b", "c", "d"])]
            frozen_cfg = train.Config(epochs=1, amp=False)
            optimizer = train.build_optimizer(frozen, frozen_cfg)
            scheduler = train.build_scheduler(optimizer, frozen_cfg, 1)
            train.train_one_epoch(frozen, loader, torch.nn.CrossEntropyLoss(), optimizer, scheduler,
                                  None, frozen_cfg, torch.device("cpu"), frozen=True)
            assert torch.equal(before, bn.running_mean)
            assert frozen.get_classifier().training
            # Exercise the real evaluator and workbook/report/ZIP export on synthetic data.
            for seed in (1, 2):
                final_cfg = lab.cfg(exp_id="F01", seed=seed, backbone="resnet18", init="scratch", img_size=32,
                                    batch_size=9, limit_train=18, limit_val=9, num_workers=0)
                lab.run_exp(final_cfg)
                lab.finalize_one("F01", seed, "hflip_prob", True)
            for seed in (0, 1, 2):
                baseline_cfg = lab.cfg(exp_id="T00", seed=seed, backbone="resnet18", init="scratch", img_size=32,
                                       batch_size=9, limit_train=18, limit_val=9, num_workers=0)
                lab.alias("F01", baseline_cfg, source_seed=seed, comparison="T00")
                lab.finalize_one("T00", seed, "identity", False)
            lab.profile = "fast"  # Enable artifact export; all data remain synthetic.
            lab.final_lat = {"p95": 1.0}
            lab.method = "hflip_prob"
            (work / "environment.json").write_text(json.dumps({"synthetic_test": True}))
            archive = lab.products()
            assert archive.exists()
            with pd.ExcelFile(work / "results.xlsx") as workbook:
                assert set(workbook.sheet_names) == {
                    "Backbones", "Training", "Inference", "Final", "PerClass", "Latency", "Summary"}
            from zipfile import ZipFile
            with ZipFile(archive) as bundle:
                assert "code/lab_day2.ipynb" in bundle.namelist()
                assert not any(name.endswith(".pt") for name in bundle.namelist())
            print("PASS: offline training, loader/checkpoint outputs, completed-run reuse, frozen BN, GMAC,")
            print("      validation-before-test, hflip calibration, official CSV validation, no repeated test,")
            print("      three-seed evaluator CLI, all seven workbook sheets, report and submission ZIP.")
            os.chdir(original_cwd)  # Windows cannot remove the process's current directory.
    finally:
        os.chdir(original_cwd)


if __name__ == "__main__":
    (Path(__file__).resolve().parent.parent / "_smoke").mkdir(exist_ok=True)
    main()
