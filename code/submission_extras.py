"""Measure missing inference timings and analyze robustness/Grad-CAM on VAL only.

Uses completed checkpoints; does not train, select a new recipe, or evaluate test.
Run: python code/submission_extras.py --root local_runs/hour
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TORCH_CUDNN_V8_API_DISABLED", "1")
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
import torch
from torch.utils.data import DataLoader, Dataset

import benchmark as bm
import dataset as ds
import inference as inf
import model as md
import eval as ev


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def load(root, eid):
    folder = root / "runs" / eid / "seed0"
    cfg = read(folder / "config.json")
    model = md.build_model(cfg["backbone"], pretrained=False, img_size=cfg["img_size"])
    state = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state["state_dict"])
    return cfg, model.cuda().eval()


class CorruptedValidation(Dataset):
    def __init__(self, root, frame, size, corruption):
        self.root, self.frame, self.corruption = root, frame, corruption
        self.transform = ds.build_transforms(False, size)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(self.root / "images" / row.Filename) as source:
            image = source.convert("RGB")
        if self.corruption == "dark":
            image = ImageEnhance.Brightness(image).enhance(0.55)
        elif self.corruption == "blur":
            image = image.filter(ImageFilter.GaussianBlur(radius=1.5))
        elif self.corruption == "noise":
            rng = np.random.default_rng(20261005 + index)
            pixels = np.asarray(image, dtype=np.float32)
            image = Image.fromarray(np.clip(pixels + rng.normal(0, 20, pixels.shape), 0, 255).astype(np.uint8))
        return self.transform(image), int(row.Label), row.Filename


def measure_timings(root, output):
    # Includes probability/logit aggregation and temperature scaling, tensor input
    # already on GPU. This is distinct from the earlier forward-only timings.
    cfg, model = load(root, "F01")
    _, other = load(root, "B03")
    _, screening = load(root, "B04")
    temperature = read(root / "runs/F01/seed0/test_manifest.json")["temperature"]
    x = torch.randn(1, 3, cfg["img_size"], cfg["img_size"], device="cuda")
    def forward(method):
        with torch.inference_mode():
            if method == "I05":
                return (screening(x).softmax(1) + other(x).softmax(1)) / 2
            logits = model(x)
            if method == "I01":
                return (logits.softmax(1) + model(x.flip(3)).softmax(1)) / 2
            if method == "I03":
                return ((logits + model(x.flip(3))) / 2).softmax(1)
            if method == "F01":
                return ((logits + model(x.flip(3))) / (2 * temperature)).softmax(1)
            if method == "I07":
                # I07 was calibrated separately for single-view inference.
                return (logits / 1.3548).softmax(1)
            return logits.softmax(1)
    results = []
    for method in ["I00", "I01", "I03", "I05", "I07", "F01"]:
        measured = bm.bench(lambda: forward(method), warmup=20, iters=100, sync=torch.cuda.synchronize)
        row = dict(exp_id=method, **measured, gpu=torch.cuda.get_device_name(0), dtype="fp32",
                   batch=1, img_size=cfg["img_size"], warmup=20,
                   scope="GPU forward + aggregation + softmax; excludes decode/resize/H2D",
                   temperature=temperature if method == "F01" else 1.3548 if method == "I07" else 1.0)
        results.append(row)
        print("TIMING", method, measured, flush=True)
    write(output / "inference_timings.json", results)
    del model, other, screening
    torch.cuda.empty_cache()


def robustness(root, output):
    cfg, model = load(root, "F01")
    frame = pd.read_csv(root / "data/labels/val_subset0.csv")
    temperature = read(root / "runs/F01/seed0/test_manifest.json")["temperature"]
    rows = []
    for corruption in ["clean", "dark", "blur", "noise"]:
        loader = DataLoader(CorruptedValidation(root, frame, cfg["img_size"], corruption),
                            batch_size=64, num_workers=2, pin_memory=True)
        labels, logits, filenames = [], [], []
        with torch.inference_mode():
            for x, y, fn in loader:
                x = x.cuda(non_blocking=True)
                logits.append(((model(x) + model(x.flip(3))) / 2).cpu().numpy())
                labels.extend(y.tolist())
                filenames.extend(fn)
        y = np.asarray(labels)
        z = np.concatenate(logits)
        before, after = inf.apply_temperature(z, 1), inf.apply_temperature(z, temperature)
        ev.save_predictions(output / f"V_{corruption}_seed0_val.csv", filenames, y, after)
        metrics = ev.compute_metrics(y, after.argmax(1), after)
        rows.append(dict(condition=corruption, n=len(y), seed=0, split="val", temperature=temperature,
                         macro_f1=metrics["macro_f1"], top1=metrics["top1"],
                         ece_before=ev.compute_metrics(y, before.argmax(1), before)["ece"],
                         ece_after=metrics["ece"], nll=metrics["nll"]))
        print("ROBUSTNESS", rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(output / "robustness.csv", index=False)
    write(output / "robustness_protocol.json", dict(checkpoint="F01/seed0/best.pt", split="val only",
          n=3501, seed=0, method="hflip_logit", temperature_source="original clean validation fit",
          brightness_factor=0.55, blur_radius_native_px=1.5, noise_std_native_8bit=20,
          noise_seed="20261005 + row index", adaptation=False, model_selection=False,
          checkpoint_sha256=hashlib.sha256((root / "runs/F01/seed0/best.pt").read_bytes()).hexdigest()))
    clean = pd.read_csv(root / "predictions/F01_seed0_val.csv").set_index("Filename")
    regenerated = pd.read_csv(output / "V_clean_seed0_val.csv").set_index("Filename").loc[clean.index]
    # Independent validation that the corruption experiment uses the saved recipe.
    np.testing.assert_allclose(regenerated[[f"p{i}" for i in range(9)]],
                               clean[[f"p{i}" for i in range(9)]], rtol=2e-4, atol=2e-5)
    gradcam(root, output, cfg, model)


def gradcam(root, output, cfg, model):
    predictions = pd.read_csv(root / "predictions/F01_seed0_val.csv")
    errors = predictions[predictions.y_true != predictions.y_pred]
    chosen = pd.concat([errors[errors.y_true == 0].head(3), errors[errors.y_true == 7].head(3)])
    names = ds.CLASS_NAMES
    fig, axes = plt.subplots(len(chosen), 3, figsize=(10, 3 * len(chosen)), squeeze=False)
    evidence = []
    # For a class-token ViT, use patch tokens BEFORE the last attention operation;
    # patch gradients after final attention are zero for the class-token score.
    saved = {}
    def capture(module, inputs, value):
        saved["features"] = value
        value.retain_grad()
    hook = model.blocks[-1].norm1.register_forward_hook(capture)
    try:
        for index, (_, row) in enumerate(chosen.iterrows()):
            with Image.open(root / "images" / row.Filename) as original:
                x = ds.build_transforms(False, cfg["img_size"])(original.convert("RGB")).unsqueeze(0).cuda()
            model.zero_grad(set_to_none=True)
            x.requires_grad_(True)
            logits = model(x)
            predicted = int(logits.argmax(1).item())
            logits[0, predicted].backward()
            features = saved["features"][0, 1:]
            gradients = saved["features"].grad[0, 1:]
            assert torch.isfinite(gradients).all() and gradients.abs().sum() > 0
            weights = gradients.mean(0)
            grid = int(features.shape[0] ** 0.5)
            cam = (features * weights).sum(1).relu().reshape(grid, grid).detach().cpu().numpy()
            cam = cam / max(float(cam.max()), 1e-12)
            rgb = x.detach().cpu()[0].permute(1, 2, 0).numpy()
            rgb = np.clip(rgb * np.asarray(ds.IMAGENET_STD) + np.asarray(ds.IMAGENET_MEAN), 0, 1)
            heat = np.asarray(Image.fromarray(cam).resize((cfg["img_size"], cfg["img_size"]), Image.Resampling.BILINEAR))
            axes[index, 0].imshow(rgb)
            axes[index, 1].imshow(rgb)
            axes[index, 1].imshow(heat, cmap="jet", alpha=0.45, vmin=0, vmax=1)
            axes[index, 2].imshow(cam, cmap="jet", vmin=0, vmax=1)
            axes[index, 0].set_title(f"{row.Filename}\nTrue: {names[int(row.y_true)]}", fontsize=9)
            axes[index, 1].set_title(f"Single-view predicted: {names[predicted]}", fontsize=9)
            axes[index, 2].set_title("Grad-CAM, last norm1 / 8x8 patches", fontsize=9)
            for ax in axes[index]:
                ax.axis("off")
            evidence.append(dict(Filename=row.Filename, true=int(row.y_true), single_view_pred=predicted,
                                 saved_tta_pred=int(row.y_pred), target="single-view predicted class",
                                 gradient_l1=float(gradients.abs().sum()), cam_max=float(cam.max())))
    finally:
        hook.remove()
    fig.suptitle("Validation error analysis: F01 seed0, single-view Grad-CAM", fontsize=12)
    fig.tight_layout()
    fig.savefig(output / "gradcam_val.png", dpi=130)
    plt.close(fig)
    write(output / "gradcam_val.json", evidence)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("local_runs/hour"))
    args = parser.parse_args()
    root = args.root.resolve()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    if not torch.cuda.is_available():
        raise RuntimeError("This measured experiment requires the declared CUDA device.")
    output = root / "extras"
    output.mkdir(exist_ok=True)
    measure_timings(root, output)
    robustness(root, output)
    print("Extras complete; test predictions unchanged.", flush=True)


if __name__ == "__main__":
    main()
