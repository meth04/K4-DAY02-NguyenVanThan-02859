"""backbone_meta.py - tính #params, GMAC và tag trọng số cho từng backbone -> runs/backbone_meta.json.

Chạy:  python code/backbone_meta.py --backbones resnet50 convnext_tiny deit_small_patch16_224 \
            efficientnet_b0 mobilenetv3_large_100 --out runs/backbone_meta.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS))
import model as md  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="+", default=list(md.SUGGESTED_BACKBONES.values()))
    ap.add_argument("--num-classes", type=int, default=9)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--out", default="runs/backbone_meta.json")
    args = ap.parse_args()

    meta = {}
    for bb in args.backbones:
        m = md.build_model(bb, pretrained=True, num_classes=args.num_classes)
        meta[bb] = {
            "params_m": round(md.count_params(m), 2),
            "gmac": round(md.count_gmacs(m, args.img_size), 2),
            "weight_tag": md.pretrained_tag(m),
        }
        print(f"{bb}: {meta[bb]}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Đã ghi {out}")


if __name__ == "__main__":
    main()
