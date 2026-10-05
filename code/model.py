"""model.py - tạo backbone (timm), đóng băng, nhóm tham số, đếm params/GMAC.

Hoàn thiện từ bộ khung starter/.

Giao diện giữ nguyên:
    build_model(name, pretrained, num_classes, drop_rate, init) -> nn.Module
    freeze_backbone(model)                                        -> None
    param_groups(model, lr_backbone, lr_head, weight_decay)       -> list[dict]
    count_params(model) -> float (triệu)     count_gmacs(model, img_size) -> float
"""
from __future__ import annotations

import torch
import torch.nn as nn
import timm

# Gợi ý backbone (GUIDE.md mục 2.1). Tag trọng số của timm có thể đổi theo phiên bản.
SUGGESTED_BACKBONES = {
    "resnet50": "resnet50",
    "resnext50": "resnext50_32x4d",
    "convnext_tiny": "convnext_tiny",
    "deit_small": "deit_small_patch16_224",
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "efficientnet_b0": "efficientnet_b0",
    "mobilenetv3": "mobilenetv3_large_100",
}


def build_model(name: str, pretrained: bool = True, num_classes: int = 9,
                drop_rate: float = 0.0, init: str = "finetune"):
    """Tạo model phân loại `num_classes` lớp qua timm.

    `init` (trục A):
      - "scratch"  : pretrained=False, huấn luyện toàn bộ
      - "frozen"   : pretrained=True, đóng băng backbone, chỉ train head
      - "finetune" : pretrained=True, train toàn bộ
    """
    init = (init or "finetune").lower()
    use_pretrained = bool(pretrained) and init in ("frozen", "finetune")
    model = timm.create_model(name, pretrained=use_pretrained, num_classes=num_classes,
                              drop_rate=drop_rate)
    if init == "frozen":
        freeze_backbone(model)
    return model


def _backbone_params(model) -> list[nn.Parameter]:
    """Tham số của backbone (mọi thứ trừ head phân loại)."""
    head_ids = {id(p) for p in model.get_classifier().parameters()}
    return [p for p in model.parameters() if id(p) not in head_ids]


def freeze_backbone(model) -> None:
    """Đóng băng mọi tham số trừ head; backbone để eval (BN không cập nhật thống kê).

    Trong train loop phải gọi model.train() RỒI đặt lại các module backbone về eval()
    (xem train.train_one_epoch) để BN đóng băng giữ nguyên running_mean/var.
    """
    for p in _backbone_params(model):
        p.requires_grad = False
    model.eval()
    for p in model.get_classifier().parameters():
        p.requires_grad = True
    # head luôn ở train mode khi train
    model.get_classifier().train()


def param_groups(model, lr_backbone: float, lr_head: float, weight_decay: float):
    """Chia tham số thành 3 nhóm như slide Day 2, trang 52.

    - backbone, ndim > 1  : lr_backbone, weight_decay
    - backbone, ndim <= 1 : lr_backbone, weight_decay = 0 (norm + bias)
    - head                : lr_head, weight_decay
    """
    head_ids = {id(p) for p in model.get_classifier().parameters()}
    decay_b, nodecay_b, head = [], [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if id(p) in head_ids:
            head.append(p)
        elif p.ndim > 1:
            decay_b.append(p)
        else:
            nodecay_b.append(p)
    groups = [
        {"params": decay_b, "lr": lr_backbone, "weight_decay": weight_decay},
        {"params": nodecay_b, "lr": lr_backbone, "weight_decay": 0.0},
        {"params": head, "lr": lr_head, "weight_decay": weight_decay},
    ]
    return [g for g in groups if len(g["params"]) > 0]


def count_params(model) -> float:
    """Số tham số (triệu), đếm cả tham số bị đóng băng."""
    return sum(p.numel() for p in model.parameters()) / 1e6


@torch.no_grad()
def count_gmacs(model, img_size: int = 224) -> float:
    """GMAC cho một ảnh 3 x img_size x img_size (MAC, không phải FLOPs 2x).

    Đếm bằng hook trên Conv2d và Linear: MAC = k*k*Cin*Cout*Hout*Wout (conv) và
    in*out (linear). Số có thể lệch vài phần trăm so với fvcore/ptflops.
    """
    model = model.eval()
    original_device = next(model.parameters()).device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    total = {"mac": 0}

    def conv_hook(module, inp, out):
        o = out[0]
        k = module.kernel_size[0] * module.kernel_size[1]
        total["mac"] += k * (module.in_channels // module.groups) * module.out_channels * o.shape[-1] * o.shape[-2]

    def linear_hook(module, inp, out):
        total["mac"] += out.numel() * module.in_features

    handles = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            handles.append(m.register_forward_hook(linear_hook))
    try:
        x = torch.randn(1, 3, img_size, img_size, device=device)
        model(x)
    finally:
        for h in handles:
            h.remove()
        model.to(original_device)
    return total["mac"] / 1e9


def pretrained_tag(model) -> str:
    """Tag trọng số timm đã nạp (để ghi vào results.xlsx)."""
    cfg = getattr(model, "pretrained_cfg", {}) or {}
    return str(cfg.get("tag", cfg.get("hf_hub_id", "?")))
