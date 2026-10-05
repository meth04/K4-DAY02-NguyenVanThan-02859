"""benchmark.py - đo độ trễ suy luận đúng cách (slide Day 2, trang 73 và 75; GUIDE.md mục 4.1).

Hoàn thiện từ bộ khung starter/.

Quy tắc đo (vi phạm bị trừ điểm, RUBRIC mục 3):
  - warmup: bỏ >= 10 lần chạy đầu
  - đồng bộ GPU: torch.cuda.synchronize() (hoặc CUDA event) TRƯỚC và SAU đoạn cần đo
  - >= 50 lần đo, báo cáo p50, p95, p99 (không chỉ trung bình)
  - ghi rõ GPU, dtype, batch, độ phân giải, có/không gộp BN, phiên bản torch
  - chọn và ghi rõ có tính tiền xử lý hay không (ở đây KHÔNG tính tiền xử lý)
"""
from __future__ import annotations

import time

import numpy as np
import torch
from torch.amp import autocast


def bench(fn, warmup: int = 10, iters: int = 100, sync=None) -> dict:
    """Đo thời gian một hàm `fn()` (không tham số), trả về mili-giây.

    `sync` là hàm đồng bộ (ví dụ torch.cuda.synchronize) hoặc None trên CPU.
    """
    def _sync():
        if sync is not None:
            sync()

    for _ in range(max(0, warmup)):
        fn()
    _sync()

    times = []
    for _ in range(iters):
        _sync()
        t0 = time.perf_counter()
        fn()
        _sync()
        times.append((time.perf_counter() - t0) * 1000.0)

    a = np.asarray(times, dtype=np.float64)
    return {
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "mean": float(a.mean()),
        "n": int(iters),
    }


def latency_report(model, batch_size: int, img_size: int, dtype: str = "fp32",
                   device: str = "cuda", warmup: int = 10, iters: int = 100) -> dict:
    """Đo độ trễ forward của `model` với đầu vào ngẫu nhiên (batch_size, 3, img_size, img_size).

    Trả về dict ghi thẳng vào sheet `Latency` của results.xlsx.
    dtype: "fp32" | "amp" (autocast) | "fp16" (model.half()).
    Lưu ý: ở batch 1, AMP có thể CHẬM hơn FP32 — đo thật, đừng giả định.
    """
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        dev = torch.device("cpu")
    sync = torch.cuda.synchronize if dev.type == "cuda" else None

    model = model.to(dev).eval()
    if dtype == "fp16":
        model = model.half()
    x = torch.randn(batch_size, 3, img_size, img_size, device=dev)
    if dtype == "fp16":
        x = x.half()

    use_amp = dtype in ("amp", "amp_bf16")

    def fn():
        with torch.inference_mode():
            if use_amp:
                with autocast(device_type=dev.type, enabled=(dev.type == "cuda"),
                              dtype=torch.bfloat16 if dtype == "amp_bf16" else torch.float16):
                    model(x)
            else:
                model(x)

    r = bench(fn, warmup=warmup, iters=iters, sync=sync)
    return {
        "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else "CPU",
        "dtype": dtype,
        "batch": batch_size,
        "img_size": img_size,
        "p50": r["p50"], "p95": r["p95"], "p99": r["p99"], "mean": r["mean"],
        "images_per_s": batch_size / (r["p50"] / 1000.0),
        "torch": torch.__version__,
        "n_iters": r["n"],
    }


def tta_latency(model, k_views: int, batch_size: int = 1, img_size: int = 224,
                dtype: str = "fp32", device: str = "cuda",
                warmup: int = 10, iters: int = 100) -> dict:
    """Độ trễ của TTA K view: đo thật K lượt forward liên tiếp, so với K * p50 một view."""
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        dev = torch.device("cpu")
    sync = torch.cuda.synchronize if dev.type == "cuda" else None

    model = model.to(dev).eval()
    if dtype == "fp16":
        model = model.half()
    x = torch.randn(batch_size, 3, img_size, img_size, device=dev)
    if dtype == "fp16":
        x = x.half()
    use_amp = dtype in ("amp", "amp_bf16")

    def one():
        with torch.inference_mode():
            if use_amp:
                with autocast(device_type=dev.type, enabled=(dev.type == "cuda"),
                              dtype=torch.bfloat16 if dtype == "amp_bf16" else torch.float16):
                    model(x)
            else:
                model(x)

    def fn():
        for _ in range(k_views):
            one()

    r = bench(fn, warmup=warmup, iters=iters, sync=sync)
    single = bench(one, warmup=warmup, iters=iters, sync=sync)
    r.update({"k_views": k_views, "batch": batch_size, "img_size": img_size, "dtype": dtype,
              "images_per_s": batch_size / (r["p50"] / 1000.0),
              "single_p50": single["p50"], "expected_p50": k_views * single["p50"],
              "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else "CPU"})
    return r
