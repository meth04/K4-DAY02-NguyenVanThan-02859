"""losses.py - các hàm loss và trộn mẫu (Mixup, CutMix).

Hoàn thiện từ bộ khung starter/. Liên hệ slide Day 2: label smoothing (trang 56),
focal loss (trang 57), Mixup/CutMix (trang 48).

Giao diện giữ nguyên:
    build_criterion(kind, **kw)                 -> callable(logits, target) -> loss scalar
    class_weights(counts, beta)                 -> tensor trọng số lớp
    mix_batch(x, y, alpha, mode)                -> (x_mixed, (y_a, y_b, lam))
    mixed_loss(criterion, logits, targets)      -> loss scalar
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Loss
# --------------------------------------------------------------------------- #
class LabelSmoothingCE(nn.Module):
    """Cross-entropy với label smoothing: q'(k) = (1 - eps) * 1[k == y] + eps / K.

    Cài đặt trực tiếp bằng torch.nn.CrossEntropyLoss(label_smoothing=eps).
    Kiểm tra: eps = 0 phải cho đúng CE (xem tests trong code/self_test.py).
    """

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = float(smoothing)
        self._ce = nn.CrossEntropyLoss(label_smoothing=self.smoothing)

    def forward(self, logits, target):
        return self._ce(logits, target)


class FocalLoss(nn.Module):
    """Focal loss nhiều lớp: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t).

    gamma = 0 phải cho lại đúng cross-entropy (sai số < 1e-6) khi alpha = None.
    `alpha` là vector trọng số theo lớp (Tensor độ dài K) hoặc None.
    """

    def __init__(self, gamma: float = 2.0, alpha=None):
        super().__init__()
        self.gamma = float(gamma)
        if alpha is not None and not torch.is_tensor(alpha):
            alpha = torch.as_tensor(alpha, dtype=torch.float32)
        self.register_buffer("alpha", alpha if alpha is not None else torch.empty(0))

    def forward(self, logits, target):
        logp = F.log_softmax(logits, dim=1)                 # log p
        logpt = logp.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = logpt.exp()
        loss = -(1.0 - pt).pow(self.gamma) * logpt
        if self.alpha.numel() > 0:
            a = self.alpha.to(logits.device)
            loss = a.gather(0, target) * loss
        return loss.mean()


def build_criterion(kind: str = "ce", **kw):
    """Trả về callable(logits, target) -> loss scalar.

    kind: "ce" | "ls" | "focal" | "ce_weighted".
      - "ls": kw["smoothing"]
      - "focal": kw["gamma"], kw["alpha"]
      - "ce_weighted": kw["weight"] (Tensor độ dài K)
    """
    kind = (kind or "ce").lower()
    if kind == "ce":
        return nn.CrossEntropyLoss()
    if kind == "ls":
        return LabelSmoothingCE(smoothing=float(kw.get("smoothing", 0.1)))
    if kind == "focal":
        return FocalLoss(gamma=float(kw.get("gamma", 2.0)), alpha=kw.get("alpha"))
    if kind == "ce_weighted":
        weight = kw.get("weight")
        if weight is None:
            raise ValueError("ce_weighted cần kw['weight'] là Tensor trọng số theo lớp")
        return nn.CrossEntropyLoss(weight=weight)
    raise ValueError(f"loss không hỗ trợ: {kind}")


def class_weights(counts, beta: float = 0.0):
    """Trọng số theo lớp từ số ảnh mỗi lớp trong tập TRAIN.

    - beta = 0: trọng số tỉ lệ nghịch với số ảnh (1 / n_c), chuẩn hoá về trung bình 1.
    - beta > 0: class-balanced theo "số mẫu hiệu dụng":
        w_c = (1 - beta) / (1 - beta ** n_c), chuẩn hoá tổng về số lớp (Cui et al., 2019).

    `counts`: list/tuple/Tensor độ dài K, thứ tự lớp 0..K-1.
    """
    c = torch.as_tensor(counts, dtype=torch.float64).clamp_min(1.0)
    if beta and beta > 0:
        w = (1.0 - beta) / (1.0 - beta ** c)
        w = w / w.sum() * len(c)          # chuẩn hoá tổng = K
    else:
        w = 1.0 / c
        w = w / w.mean()                  # chuẩn hoá trung bình = 1
    return w.to(torch.float32)


# --------------------------------------------------------------------------- #
# Mixup / CutMix
# --------------------------------------------------------------------------- #
def mix_batch(x, y, alpha: float = 1.0, mode: str = "cutmix"):
    """Trộn một batch ảnh và nhãn.

    - lam ~ Beta(alpha, alpha)
    - mode="mixup": x_mix = lam * x + (1 - lam) * x[perm]
    - mode="cutmix": cắt một hộp chữ nhật từ x[perm] dán vào x, rồi điều chỉnh lam theo
      DIỆN TÍCH THỰC của hộp sau khi cắt ra ngoài biên (slide trang 48).
    Trả về (x_mix, (y_a, y_b, lam)) với y_a = y, y_b = y[perm].
    """
    if x.size(0) < 2:
        return x, (y, y, 1.0)

    device = x.device
    lam = float(torch.distributions.Beta(alpha, alpha).sample().item())
    perm = torch.randperm(x.size(0), device=device)
    mode = (mode or "cutmix").lower()

    if mode == "mixup":
        x_mix = lam * x + (1.0 - lam) * x[perm]
        return x_mix, (y, y[perm], lam)

    if mode == "cutmix":
        _, _, H, W = x.shape
        cut_rat = (1.0 - lam) ** 0.5
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        cx = int(torch.randint(W, (1,), device=device).item())
        cy = int(torch.randint(H, (1,), device=device).item())
        x1 = max(cx - cut_w // 2, 0)
        y1 = max(cy - cut_h // 2, 0)
        x2 = min(cx + cut_w // 2, W)
        y2 = min(cy + cut_h // 2, H)
        # Điều chỉnh lam theo diện tích hộp THỰC sau khi cắt vào biên.
        lam_eff = 1.0 - ((x2 - x1) * (y2 - y1) / (W * H))
        x_mix = x.clone()
        if x2 > x1 and y2 > y1:
            x_mix[:, :, y1:y2, x1:x2] = x[perm][:, :, y1:y2, x1:x2]
        return x_mix, (y, y[perm], lam_eff)

    raise ValueError(f"mode không hỗ trợ: {mode} (chỉ 'mixup' hoặc 'cutmix')")


def mixed_loss(criterion, logits, targets):
    """Loss cho batch đã trộn: lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)."""
    y_a, y_b, lam = targets
    return lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
