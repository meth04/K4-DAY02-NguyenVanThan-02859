"""train.py - vòng huấn luyện cho mọi thí nghiệm (B, T, F).

Hoàn thiện từ bộ khung starter/. Dùng MỘT hàm `run(cfg)` cho mọi cấu hình (RUBRIC mục H):
đổi thí nghiệm chỉ bằng cách đổi `Config`.

Chạy từ dòng lệnh:
    python train.py --set exp_id=B01 backbone=resnet50 seed=0

Chỉ số chọn checkpoint (macro-F1 val) tính bằng eval.compute_metrics để cùng định nghĩa lúc chấm.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

# --- import các module cùng thư mục code/ ---
_THIS = Path(__file__).resolve().parent
if str(_THIS) not in sys.path:
    sys.path.insert(0, str(_THIS))
# eval.py nằm ở gốc repo (một cấp trên code/)
_ROOT = _THIS.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import dataset as ds          # noqa: E402
import losses as ls           # noqa: E402
import model as md            # noqa: E402
from eval import compute_metrics, save_predictions  # noqa: E402

try:  # eval.py ở gốc repo; nếu chạy trong code/ thì vẫn import được nhờ sys.path ở trên
    import eval as ev
except Exception:  # pragma: no cover
    ev = None


@dataclass
class Config:
    # --- định danh ---
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0
    # --- mô hình ---
    backbone: str = "resnet50"
    init: str = "finetune"            # scratch | frozen | finetune
    drop_rate: float = 0.0
    # --- dữ liệu / augmentation ---
    img_size: int = 224
    aug: str = "basic"                # basic | color | trivial | randaug
    sampler: str | None = None        # None | balanced
    mix: str | None = None            # None | mixup | cutmix
    mix_alpha: float = 1.0
    # --- loss ---
    loss: str = "ce"                  # ce | ls | focal | ce_weighted
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None
    # --- tối ưu (công thức nền, GUIDE.md mục 1.4) ---
    epochs: int = 12
    batch_size: int = 64
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema_decay: float | None = None
    amp: bool = True
    num_workers: int = 2
    # --- đường dẫn ---
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    out_dir: str = "runs"
    pred_dir: str = "predictions"
    curves_dir: str = "curves"
    # --- chỉ bật ở Bước 4 (chung kết): ghi predictions trên TEST. Mặc định TẮT (quy tắc S4). ---
    save_test_predictions: bool = False
    # --- tăng tốc / smoke-test tuỳ chọn ---
    channels_last: bool = False
    amp_dtype: str = "float16"       # bfloat16 on supported GPUs; float16 on T4
    fused_optimizer: bool = False
    patience: int | None = None      # val macro-F1 early stopping; None = all epochs
    save_last: bool = True
    limit_train: int | None = None    # chỉ dùng cho smoke-test; None = toàn bộ train
    limit_val: int | None = None      # chỉ dùng cho smoke-test; None = toàn bộ val


# --------------------------------------------------------------------------- #
# Đường dẫn
# --------------------------------------------------------------------------- #
def run_dir(cfg: Config) -> Path:
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg: Config, split: str) -> Path:
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def curve_path(cfg: Config) -> Path:
    return Path(cfg.curves_dir) / f"{cfg.exp_id}_{cfg.backbone}.png"


# --------------------------------------------------------------------------- #
# Seed
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    """Cố định mọi nguồn ngẫu nhiên (random, numpy, torch CPU+CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False   # đổi thành True nếu cần tái lập tuyệt đối (chậm hơn)
    torch.backends.cudnn.benchmark = True


# --------------------------------------------------------------------------- #
# Optimizer / scheduler / EMA
# --------------------------------------------------------------------------- #
def build_optimizer(model, cfg: Config):
    groups = md.param_groups(model, cfg.lr_backbone, cfg.lr_head, cfg.weight_decay)
    options = {"fused": True} if cfg.fused_optimizer and next(model.parameters()).is_cuda else {}
    return torch.optim.AdamW(groups, **options)


def build_scheduler(optimizer, cfg: Config, steps_per_epoch: int):
    """Warmup tuyến tính (theo bước) rồi cosine về ~0 (theo bước)."""
    total = max(1, cfg.epochs * steps_per_epoch)
    warmup = max(1, int(cfg.warmup_epochs * steps_per_epoch))

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        prog = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class EMA:
    """Trung bình động trọng số: W_ema <- d * W_ema + (1 - d) * W.

    Buffer của BatchNorm (running_mean/var) cũng được sao chép theo kiểu EMA.
    """

    def __init__(self, model, decay: float):
        self.decay = float(decay)
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model) -> None:
        msd = model.state_dict()
        for k, v in self.shadow.items():
            if k not in msd:
                continue
            src = msd[k].detach()
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(src, alpha=1.0 - self.decay)
            else:
                v.copy_(src)

    def copy_to(self, model) -> None:
        model.load_state_dict(self.shadow, strict=True)

    def state_dict(self):
        return copy.deepcopy(self.shadow)


# --------------------------------------------------------------------------- #
# Train / eval một epoch
# --------------------------------------------------------------------------- #
def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, cfg: Config,
                    device, ema: EMA | None = None, frozen: bool = False) -> dict:
    model.train()
    if frozen:
        # backbone đóng băng: giữ BN ở eval để không cập nhật running_mean/var
        model.eval()
        model.get_classifier().train()

    running = torch.zeros((), device=device)
    correct = torch.zeros((), device=device)
    seen = 0
    amp_dtype = torch.bfloat16 if cfg.amp_dtype == "bfloat16" else torch.float16
    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if cfg.channels_last:
            x = x.to(memory_format=torch.channels_last)

        targets = None
        if cfg.mix:
            x, (ya, yb, lam) = ls.mix_batch(x, y, cfg.mix_alpha, cfg.mix)
            targets = (ya, yb, lam)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=cfg.amp, dtype=amp_dtype):
            logits = model(x)
            loss = ls.mixed_loss(criterion, logits, targets) if targets else criterion(logits, y)

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
            optimizer.step()
        scheduler.step()
        if ema is not None:
            ema.update(model)

        running += loss.detach() * y.size(0)
        seen += y.size(0)
        if targets is None:  # accuracy chỉ có nghĩa khi nhãn không bị trộn
            correct += (logits.argmax(1) == y).sum().detach()

    return {"train_loss": running.item() / max(1, seen),
            "train_acc": correct.item() / max(1, seen),
            "lr": optimizer.param_groups[0]["lr"]}


@torch.no_grad()
def evaluate(model, loader, criterion, device, amp: bool = True, amp_dtype: str = "float16",
             channels_last: bool = False):
    """Trả về (filenames, y_true, logits, loss). Giữ đúng thứ tự của loader."""
    model.eval()
    filenames, ys, logits_all = [], [], []
    total_loss, n = 0.0, 0
    for x, y, f in loader:
        x = x.to(device, non_blocking=True)
        if channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        y = y.to(device, non_blocking=True)
        with autocast(device_type=device.type, enabled=amp,
                      dtype=torch.bfloat16 if amp_dtype == "bfloat16" else torch.float16):
            logits = model(x)
            loss = criterion(logits, y)
        total_loss += loss.item() * y.size(0)
        n += y.size(0)
        filenames.extend(list(f))
        ys.append(y.detach().cpu())
        logits_all.append(logits.float().detach().cpu())
    y_true = torch.cat(ys).numpy() if ys else np.array([], dtype=np.int64)
    logits_np = torch.cat(logits_all).numpy() if logits_all else np.zeros((0, ds.NUM_CLASSES))
    return filenames, y_true, logits_np, total_loss / max(1, n)


def softmax_np(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


# --------------------------------------------------------------------------- #
# Vẽ đường cong
# --------------------------------------------------------------------------- #
def plot_curves(history: list[dict], path: str | Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ep = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].plot(ep, [h["train_loss"] for h in history], "-o", label="train")
    axes[0].plot(ep, [h["val_loss"] for h in history], "-s", label="val")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss"); axes[0].set_title("Loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(ep, [h["val_macro_f1"] for h in history], "-o", color="tab:green", label="val macro-F1")
    axes[1].plot(ep, [h["val_top1"] for h in history], "-s", color="tab:blue", label="val top-1")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("metric"); axes[1].set_title("Val metric")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    axes[2].plot(ep, [h["lr"] for h in history], "-", color="tab:red")
    axes[2].set_xlabel("epoch"); axes[2].set_ylabel("lr"); axes[2].set_title("Learning rate")
    axes[2].grid(alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Chạy một cấu hình
# --------------------------------------------------------------------------- #
def run(cfg: Config) -> dict:
    """Huấn luyện một cấu hình và lưu mọi thứ cần thiết. Trả về dict tóm tắt."""
    set_seed(cfg.seed)
    rdir = run_dir(cfg)
    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / "config.json").write_text(json.dumps(asdict(cfg), indent=2, ensure_ascii=False))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. dữ liệu + kiểm tra split
    train_df, val_df, test_df = ds.load_split(cfg.labels_dir, cfg.fold)
    smoke = bool(cfg.limit_train or cfg.limit_val)
    ds.check_split(train_df, val_df, test_df, cfg.images_dir, strict=not smoke)
    if cfg.limit_train:  # smoke-test
        train_df = train_df.iloc[:cfg.limit_train].reset_index(drop=True)
    if cfg.limit_val:
        val_df = val_df.iloc[:cfg.limit_val].reset_index(drop=True)

    # 3. loader
    tf_train = ds.build_transforms(True, cfg.img_size, cfg.aug)
    tf_eval = ds.build_transforms(False, cfg.img_size, cfg.aug)
    train_loader = ds.make_loader(train_df, cfg.images_dir, tf_train, cfg.batch_size,
                                  True, cfg.sampler, cfg.num_workers, cfg.seed)
    val_loader = ds.make_loader(val_df, cfg.images_dir, tf_eval, cfg.batch_size,
                                False, None, cfg.num_workers, cfg.seed)

    # 4. model + loss + optimizer
    model = md.build_model(cfg.backbone, pretrained=True, num_classes=ds.NUM_CLASSES,
                           drop_rate=cfg.drop_rate, init=cfg.init)
    if cfg.channels_last:
        model = model.to(memory_format=torch.channels_last)
    model = model.to(device)
    frozen = cfg.init == "frozen"
    if frozen:
        md.freeze_backbone(model)

    # loss
    kw = {}
    if cfg.loss == "ls":
        kw["smoothing"] = cfg.label_smoothing if cfg.label_smoothing > 0 else 0.1
    elif cfg.loss == "focal":
        kw["gamma"] = cfg.focal_gamma
    elif cfg.loss == "ce_weighted":
        counts = train_df["Label"].value_counts().sort_index().to_numpy()
        beta = cfg.class_weight_beta or 0.0
        kw["weight"] = ls.class_weights(counts, beta).to(device)
    criterion = ls.build_criterion(cfg.loss, **kw)

    optimizer = build_optimizer(model, cfg)
    steps_per_epoch = max(1, len(train_loader))
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch)
    scaler = GradScaler(device.type, enabled=cfg.amp and cfg.amp_dtype != "bfloat16")
    ema = EMA(model, cfg.ema_decay) if cfg.ema_decay else None

    # 5. vòng epoch
    history, best_f1, best_epoch, best_state = [], -1.0, -1, None
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        tr = train_one_epoch(model, train_loader, criterion, optimizer, scheduler,
                             scaler, cfg, device, ema, frozen)
        # đánh giá bằng trọng số EMA nếu có
        if ema is not None:
            eval_model = copy.deepcopy(model)
            ema.copy_to(eval_model)
        else:
            eval_model = model
        fnames, y_true, logits, val_loss = evaluate(eval_model, val_loader, criterion, device,
                                                   cfg.amp, cfg.amp_dtype, cfg.channels_last)
        m = compute_metrics(y_true, logits.argmax(1), softmax_np(logits))
        dt = time.time() - t0
        history.append({"epoch": epoch, "train_loss": tr["train_loss"], "train_acc": tr["train_acc"],
                        "val_loss": val_loss, "val_macro_f1": m["macro_f1"], "val_top1": m["top1"],
                        "lr": tr["lr"], "epoch_time_s": dt})
        print(f"[{cfg.exp_id} s{cfg.seed}] epoch {epoch:2d}/{cfg.epochs} "
              f"train_loss {tr['train_loss']:.4f} val_loss {val_loss:.4f} "
              f"val_macroF1 {m['macro_f1']:.4f} val_top1 {m['top1']:.4f} ({dt:.1f}s)")

        # chọn checkpoint tốt nhất theo macro-F1 val (hòa -> epoch sớm hơn)
        if m["macro_f1"] > best_f1:
            best_f1, best_epoch = m["macro_f1"], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in eval_model.state_dict().items()}
            torch.save({"state_dict": best_state, "cfg": asdict(cfg), "epoch": epoch}, rdir / "best.pt")
        # lưu checkpoint mỗi epoch (chống ngắt phiên)
        if cfg.save_last:
            torch.save({"state_dict": eval_model.state_dict(), "epoch": epoch,
                        "macro_f1": m["macro_f1"], "cfg": asdict(cfg)}, rdir / "last.pt")
        pd.DataFrame(history).to_csv(rdir / "history.csv", index=False)
        if ema is not None:
            del eval_model
        if cfg.patience and epoch - best_epoch >= cfg.patience:
            print(f"Early stopping: val macro-F1 không cải thiện trong {cfg.patience} epoch.")
            break

    # 6. nạp checkpoint tốt nhất -> lưu val logits + predictions val
    model.load_state_dict(best_state)
    fnames, y_true, logits, _ = evaluate(model, val_loader, criterion, device, cfg.amp,
                                       cfg.amp_dtype, cfg.channels_last)
    probs = softmax_np(logits)
    np.save(rdir / "val_logits.npy", logits)
    np.save(rdir / "val_labels.npy", y_true)
    (rdir / "val_filenames.txt").write_text("\n".join(fnames), encoding="utf-8")
    save_predictions(pred_path(cfg, "val"), fnames, y_true, probs)
    val_metrics = compute_metrics(y_true, probs.argmax(1), probs)

    # 7. TEST: chỉ khi được bật tường minh (Bước 4). Chạy đúng MỘT lần.
    if cfg.save_test_predictions:
        test_loader = ds.make_loader(test_df, cfg.images_dir, tf_eval, cfg.batch_size,
                                     False, None, cfg.num_workers, cfg.seed)
        tf_names, ty_true, tlogits, _ = evaluate(model, test_loader, criterion, device, cfg.amp)
        tprobs = softmax_np(tlogits)
        np.save(rdir / "test_logits.npy", tlogits)
        np.save(rdir / "test_labels.npy", ty_true)
        (rdir / "test_filenames.txt").write_text("\n".join(tf_names), encoding="utf-8")
        save_predictions(pred_path(cfg, "test"), tf_names, ty_true, tprobs)

    # 8. ghi history, vẽ đường cong
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(rdir / "history.csv", index=False)
    try:
        plot_curves(history, curve_path(cfg), f"{cfg.exp_id} · {cfg.backbone} · seed{cfg.seed}")
    except Exception as e:  # matplotlib có thể thiếu ở môi trường tối giản
        print(f"  (cảnh báo) không vẽ được curve: {e}")

    torch.save({"state_dict": model.state_dict(), "cfg": asdict(cfg)}, rdir / "best.pt")

    summary = {
        "exp_id": cfg.exp_id, "seed": cfg.seed, "backbone": cfg.backbone,
        "init": cfg.init, "loss": cfg.loss, "aug": cfg.aug, "mix": cfg.mix,
        "epochs": cfg.epochs, "best_epoch": best_epoch,
        "epochs_run": len(history),
        "params_m": md.count_params(model),
        "val_macro_f1": float(val_metrics["macro_f1"]),
        "val_top1": float(val_metrics["top1"]),
        "val_balanced_acc": float(val_metrics["balanced_acc"]),
        "val_ece": float(val_metrics["ece"]),
        "train_time_per_epoch_s": float(np.mean([h["epoch_time_s"] for h in history])),
        "weight_tag": md.pretrained_tag(model),
    }
    (rdir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[{cfg.exp_id} s{cfg.seed}] XONG. val macro-F1={summary['val_macro_f1']:.4f} "
          f"(epoch {best_epoch})")
    return summary


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _coerce(field, value: str):
    if value.lower() in ("none", "null"):
        return None
    # `from __future__ import annotations` làm field.type là CHUỖI, ví dụ "int | None",
    # "str | None", "float | None". Chuẩn hoá về kiểu bên trong.
    t = field.type
    if isinstance(t, str):
        t = t.replace("Optional[", "").replace("]", "")
        parts = [p.strip() for p in t.split("|")]
        non_none = [p for p in parts if p not in ("None", "NoneType")]
        t = non_none[0] if len(non_none) == 1 else t
    else:
        args = getattr(t, "__args__", None)
        if args:
            non_none = [a for a in args if a is not type(None)]
            t = non_none[0] if len(non_none) == 1 else t
    if t is int or t == "int":
        return int(value)
    if t is float or t == "float":
        return float(value)
    if t is bool or t == "bool":
        return value.lower() in ("1", "true", "yes", "y")
    return value


def parse_overrides(pairs: list[str]) -> dict:
    """Biến ['seed=1', 'loss=focal', 'ema_decay=none'] thành dict, ép kiểu theo field của Config."""
    valid = {f.name: f for f in fields(Config)}
    out = {}
    for p in pairs:
        if "=" not in p:
            raise ValueError(f"cần dạng key=value, nhận '{p}'")
        k, v = p.split("=", 1)
        k = k.strip()
        if k not in valid:
            raise ValueError(f"Config không có field '{k}'. Các field hợp lệ: {sorted(valid)}")
        # Optional[...] -> lấy kiểu bên trong nếu có
        f = valid[k]
        out[k] = _coerce(f, v.strip())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Train một cấu hình Lab Day 2")
    ap.add_argument("--set", nargs="*", default=[], help="các override dạng KEY=VALUE")
    args = ap.parse_args()
    cfg = Config(**parse_overrides(args.set))
    run(cfg)


if __name__ == "__main__":
    main()
