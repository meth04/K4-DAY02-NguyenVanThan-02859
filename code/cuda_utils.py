"""Select native CUDA AMP precision rather than emulated bfloat16."""
import torch


def amp_dtype_for_device(device):
    native_bf16 = (device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8
                   and torch.cuda.is_bf16_supported())
    return "bfloat16" if native_bf16 else "float16"
