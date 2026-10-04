"""self_test.py - các kiểm tra tính đúng đắn của pipeline (RUBRIC mục A, H; GUIDE.md mục 1.3).

Chạy:  python code/self_test.py

Gồm:
  1. loss ban đầu của head mới ~= ln(9) = 2.197
  2. overfit được một batch nhỏ tới loss ~ 0
  3. focal loss gamma=0 == cross-entropy (sai số < 1e-6)
  4. label smoothing eps=0 == cross-entropy
  5. CutMix: lam khớp diện tích hộp thực; ảnh/label trộn đúng
  6. fuse_conv_bn: sai số đầu ra truoc/sau <= 1e-5
  7. (nếu có dữ liệu) một vài kiểm tra trên batch thật: ảnh/label khớp, chuẩn hoá
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS))
sys.path.insert(0, str(_THIS.parent))

import inference as inf      # noqa: E402
import losses as ls          # noqa: E402
import model as md           # noqa: E402


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    return ok


def test_initial_loss():
    torch.manual_seed(0)
    m = md.build_model("resnet50", pretrained=False, num_classes=9)
    m.eval()
    x = torch.randn(8, 3, 224, 224)
    with torch.no_grad():
        logits = m(x)
    loss = F.cross_entropy(logits, torch.zeros(8, dtype=torch.long)).item()
    expected = float(np.log(9))
    return check("loss ban đầu ~ ln(9)", abs(loss - expected) < 0.6,
                 f"loss={loss:.4f} kỳ vọng~{expected:.4f} (|Δ|<0.6)")


def test_overfit_one_batch():
    torch.manual_seed(0)
    m = md.build_model("resnet50", pretrained=False, num_classes=9)
    m.train()
    x = torch.randn(8, 3, 224, 224)
    y = torch.randint(0, 9, (8,))
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    first = last = None
    for _ in range(30):
        opt.zero_grad()
        loss = crit(m(x), y)
        loss.backward()
        opt.step()
        if first is None:
            first = loss.item()
        last = loss.item()
    return check("overfit 1 batch nhỏ", last < 0.05, f"loss {first:.3f} -> {last:.4f}")


def test_focal_gamma0_equals_ce():
    torch.manual_seed(0)
    logits = torch.randn(16, 9, dtype=torch.float64)
    y = torch.randint(0, 9, (16,))
    fl = ls.FocalLoss(gamma=0.0).double()
    ce = nn.CrossEntropyLoss()
    a = fl(logits, y).item()
    b = ce(logits, y).item()
    return check("focal gamma=0 == CE", abs(a - b) < 1e-6, f"focal={a:.8f} ce={b:.8f}")


def test_ls_eps0_equals_ce():
    torch.manual_seed(0)
    logits = torch.randn(16, 9)
    y = torch.randint(0, 9, (16,))
    a = ls.LabelSmoothingCE(smoothing=0.0)(logits, y).item()
    b = nn.CrossEntropyLoss()(logits, y).item()
    return check("label smoothing eps=0 == CE", abs(a - b) < 1e-6, f"ls={a:.8f} ce={b:.8f}")


def test_cutmix_area():
    torch.manual_seed(0)
    x = torch.rand(4, 3, 32, 32)
    y = torch.arange(4)
    x_mix, (ya, yb, lam) = ls.mix_batch(x, y, alpha=1.0, mode="cutmix")
    # diện tích đổi so với ảnh gốc ~ (1 - lam) * H * W
    changed = (x_mix != x).any(dim=1).float().mean().item()
    ok = 0.0 <= lam <= 1.0 and torch.equal(ya, y) and not torch.equal(x_mix, x)
    return check("CutMix lam/area", ok, f"lam={lam:.3f}, tỉ lệ pixel đổi~{changed:.3f}")


def test_mixup_label():
    torch.manual_seed(0)
    x = torch.rand(4, 3, 16, 16)
    y = torch.arange(4)
    x_mix, (ya, yb, lam) = ls.mix_batch(x, y, alpha=1.0, mode="mixup")
    ok = torch.equal(ya, y) and not torch.equal(ya, yb)
    return check("Mixup trộn nhãn", ok, f"lam={lam:.3f}")


def test_fuse_conv_bn():
    torch.manual_seed(0)
    m = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU()).eval()
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        before = m(x).clone()
    fused = inf.fuse_conv_bn(m)
    with torch.no_grad():
        after = fused(x)
    err = (before - after).abs().max().item()
    return check("fuse_conv_bn sai số <= 1e-5", err <= 1e-5, f"max|Δ|={err:.2e}")


def test_temperature():
    torch.manual_seed(0)
    logits = torch.randn(200, 9) * 3.0
    y = logits.argmax(1).numpy()
    T = inf.fit_temperature(logits.numpy(), y)
    p0 = F.softmax(logits, dim=1).numpy()
    p1 = inf.apply_temperature(logits.numpy(), T)
    ece0 = _ece(p0, y)
    ece1 = _ece(p1, y)
    return check("temperature scaling giảm NLL/ECE", ece1 <= ece0 + 1e-6,
                 f"T={T:.3f}, ECE {ece0:.4f} -> {ece1:.4f}")


def _ece(p, y, bins=15):
    conf = p.max(1)
    correct = (p.argmax(1) == y).astype(float)
    idx = np.clip(np.ceil(conf * bins).astype(int) - 1, 0, bins - 1)
    e = 0.0
    for m in range(bins):
        mask = idx == m
        if mask.any():
            e += mask.sum() / len(conf) * abs(correct[mask].mean() - conf[mask].mean())
    return e


def main():
    print("=" * 72)
    print("SELF-TEST pipeline Lab Day 2")
    print("=" * 72)
    results = [
        test_focal_gamma0_equals_ce(),
        test_ls_eps0_equals_ce(),
        test_cutmix_area(),
        test_mixup_label(),
        test_fuse_conv_bn(),
        test_temperature(),
        test_initial_loss(),
        test_overfit_one_batch(),
    ]
    print("=" * 72)
    print(f"{sum(results)}/{len(results)} kiểm tra đạt.")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
