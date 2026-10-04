"""inference.py - các phương pháp suy luận (Bước 3 của GUIDE.md).

Hoàn thiện từ bộ khung starter/. Liên hệ slide Day 2: TTA (trang 62-66, 75),
ensemble/EMA/soup (trang 67), độ phân giải kiểm tra (trang 68), temperature scaling
(trang 69), gộp BatchNorm (trang 71).

Mọi hàm chạy ở chế độ eval, không gradient. Nhiệt độ T khớp trên VAL rồi áp dụng sang TEST.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Chạy model
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict_logits(model, loader, device, view=None):
    """Chạy model trên loader và gom logit theo đúng thứ tự file.

    `view` là hàm biến đổi batch ảnh trước khi đưa vào model (ví dụ lật ngang), hoặc None.
    Trả về (filenames, y_true, logits[N, 9]).
    """
    model.eval()
    filenames, ys, logits_all = [], [], []
    for x, y, f in loader:
        x = x.to(device, non_blocking=True)
        if view is not None:
            x = view(x)
        with torch.inference_mode():
            logits = model(x)
        filenames.extend(list(f))
        ys.append(y.numpy())
        logits_all.append(logits.float().cpu().numpy())
    y_true = np.concatenate(ys) if ys else np.array([], dtype=np.int64)
    logits = np.concatenate(logits_all) if logits_all else np.zeros((0, 9), dtype=np.float32)
    return filenames, y_true, logits


def tta_logits(model, loader, device, view_fns):
    """Chạy K view, trả về (filenames, y_true, [logits_view_0, ..., logits_view_{K-1}]).

    Mỗi phần tử của `view_fns` là hàm x -> x (hoặc danh sách các batch, xem views_multicrop).
    """
    out, filenames, y_true = [], None, None
    for vf in view_fns:
        fn, yt, lg = predict_logits(model, loader, device, vf)
        filenames, y_true = fn, yt
        out.append(lg)
    return filenames, y_true, out


# --------------------------------------------------------------------------- #
# Các view cho TTA
# --------------------------------------------------------------------------- #
def view_identity(x):
    return x


def view_hflip(x):
    """Lật ngang batch (N, C, H, W) trên chiều rộng."""
    return torch.flip(x, dims=[3])


def views_multicrop(x, crop: int):
    """5 crop (4 góc + giữa) kích thước `crop` từ mỗi ảnh trong batch.

    Trả về list 5 batch (mỗi batch là một crop của toàn bộ batch). Dùng với tta_logits
    bằng cách bọc từng crop thành một view_fn.
    """
    H, W = x.shape[-2:]
    crop = min(crop, H, W)
    ys = [0, H - crop, (H - crop) // 2, 0, H - crop]
    xs = [0, 0, (W - crop) // 2, W - crop, W - crop]
    return [x[:, :, y:y + crop, xx:xx + crop] for y, xx in zip(ys, xs)]


def views_multiscale(x, sizes):
    """Resize batch về từng kích thước trong `sizes`, trả về list các batch.

    Lưu ý: CNN có global pooling thì nhận mọi kích thước; ViT/Swin cần cùng kích thước
    với lúc train (giữ nguyên vị trí/cửa sổ) nên chỉ dùng cho CNN.
    """
    out = []
    for s in sizes:
        if x.shape[-1] == s and x.shape[-2] == s:
            out.append(x)
        else:
            out.append(F.interpolate(x, size=(s, s), mode="bilinear", align_corners=False))
    return out


# --------------------------------------------------------------------------- #
# Gộp view / ensemble
# --------------------------------------------------------------------------- #
def aggregate_views(logits_per_view, space: str = "prob"):
    """Gộp K lượt chạy của TTA thành một dự đoán.

      - space="prob":  trung bình softmax của từng view
      - space="logit": trung bình logit rồi softmax
    Trả về xác suất (N, 9) đã chuẩn hoá.
    """
    if space == "logit":
        z = np.mean(np.stack(logits_per_view, 0), axis=0)
        z = z - z.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)
    probs = [F.softmax(torch.as_tensor(lg, dtype=torch.float64), dim=1).numpy()
             for lg in logits_per_view]
    p = np.mean(np.stack(probs, 0), axis=0)
    return p / p.sum(1, keepdims=True)


def ensemble_probs(list_of_probs):
    """Trung bình xác suất của nhiều mô hình (cùng tập ảnh, cùng thứ tự file)."""
    p = np.mean(np.stack(list_of_probs, 0), axis=0)
    return p / p.sum(1, keepdims=True)


# --------------------------------------------------------------------------- #
# Temperature scaling
# --------------------------------------------------------------------------- #
def _logits_to_tensor(logits):
    return torch.as_tensor(np.asarray(logits), dtype=torch.float64)


def fit_temperature(val_logits, val_labels) -> float:
    """Tìm nhiệt độ T > 0 cực tiểu NLL trên VAL: p = softmax(logit / T).

    Tối ưu log(T) bằng LBFGS. Accuracy không đổi. KHÔNG khớp T trên test.
    """
    logits = _logits_to_tensor(val_logits)
    labels = torch.as_tensor(np.asarray(val_labels), dtype=torch.long)
    log_T = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_T], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        T = log_T.exp().clamp(0.05, 20.0)
        loss = F.cross_entropy(logits / T, labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_T.exp().clamp(0.05, 20.0).item())


def apply_temperature(logits, T: float):
    """Trả về softmax(logits / T)."""
    z = _logits_to_tensor(logits) / float(T)
    return F.softmax(z, dim=1).numpy()


# --------------------------------------------------------------------------- #
# Gộp BatchNorm vào conv
# --------------------------------------------------------------------------- #
def _fuse_pair(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    gamma, beta = bn.weight, bn.bias
    mean, var, eps = bn.running_mean, bn.running_var, bn.eps
    std = (var + eps).sqrt()
    scale = (gamma / std).reshape(-1, 1, 1, 1)
    w = conv.weight * scale
    b = conv.bias if conv.bias is not None else torch.zeros_like(mean)
    b = beta + (b - mean) * gamma / std
    fused = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size,
                      conv.stride, conv.padding, conv.dilation, conv.groups, bias=True)
    fused.weight.data.copy_(w.detach())
    fused.bias.data.copy_(b.detach())
    return fused.to(conv.weight.device, conv.weight.dtype)


def fuse_conv_bn(model):
    """Gộp BatchNorm2d vào Conv2d liền trước (chính xác lúc suy luận).

        w' = gamma * w / sqrt(var + eps)
        b' = beta + gamma * (b - mean) / sqrt(var + eps)

    Dùng torch.fx để tìm cặp conv->bn; nếu không trace được thì trả về model gốc.
    Với kiến trúc không có BN (ViT/Swin/ConvNeXt dùng LayerNorm) thì không áp dụng.
    """
    model.eval()
    try:
        import torch.fx as fx
        gm = fx.symbolic_trace(model)
    except Exception as e:  # pragma: no cover
        print(f"fuse_conv_bn: không trace được ({e}); giữ nguyên model")
        return model

    modules = dict(gm.named_modules())
    n_fused = 0
    for node in list(gm.graph.nodes):
        if node.op != "call_module":
            continue
        conv = modules.get(node.target)
        if not isinstance(conv, nn.Conv2d):
            continue
        users = [u for u in node.users]
        if len(users) != 1:
            continue
        bn_node = users[0]
        if bn_node.op != "call_module":
            continue
        bn = modules.get(bn_node.target)
        if not isinstance(bn, nn.BatchNorm2d):
            continue
        fused = _fuse_pair(conv, bn)
        name = f"__fused_{node.target.replace('.', '_')}__"
        gm.add_submodule(name, fused)
        # fused(conv(x)) nên nhận ĐẦU VÀO của conv (node.args), không phải đầu ra của BN.
        with gm.graph.inserting_after(bn_node):
            new_node = gm.graph.call_module(name, args=node.args, kwargs=node.kwargs)
        bn_node.replace_all_uses_with(new_node)
        gm.graph.erase_node(bn_node)
        gm.graph.erase_node(node)
        n_fused += 1

    gm.graph.lint()
    gm.recompile()
    print(f"fuse_conv_bn: đã gộp {n_fused} cặp Conv+BN")
    return gm
