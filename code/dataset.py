"""dataset.py - đọc DeepWeeds, kiểm tra chia dữ liệu, transform, DataLoader.

Hoàn thiện từ bộ khung starter/. Quy tắc chia dữ liệu bắt buộc (S1-S6): README.md mục 2.1.

Giao diện giữ nguyên:
    load_split(labels_dir, fold=0)            -> (train_df, val_df, test_df)
    check_split(train_df, val_df, test_df, images_dir) -> dict
    build_transforms(train, img_size, aug)    -> torchvision transform
    DeepWeedsDataset[i]                       -> (image_tensor, label:int, filename:str)
    make_loader(df, images_dir, transform, batch_size, train, sampler, num_workers)
"""
from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

NUM_CLASSES = 9
# Thứ tự lớp theo cột `Label` của labels.csv (0 = Chinee Apple ... 7 = Snake Weed, 8 = Negatives).
CLASS_NAMES = [
    "Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
    "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Ảnh gốc DeepWeeds là 256x256; ta resize/center-crop theo img_size.
NATIVE_SIZE = 256


def load_split(labels_dir: str | Path, fold: int = 0):
    """Đọc train_subset{fold}.csv, val_subset{fold}.csv, test_subset{fold}.csv (S1).

    Mỗi file có cột `Filename, Label, Species`. KHÔNG sửa, lọc hay chia lại dữ liệu.
    """
    labels_dir = Path(labels_dir)
    dfs = []
    for split in ("train", "val", "test"):
        path = labels_dir / f"{split}_subset{fold}.csv"
        if not path.exists():
            raise FileNotFoundError(f"không thấy {path}; hãy tải các CSV từ GitHub của tác giả")
        df = pd.read_csv(path)
        for col in ("Filename", "Label"):
            if col not in df.columns:
                raise ValueError(f"{path} thiếu cột {col}")
        df = df[["Filename", "Label"]].copy()
        df["Label"] = df["Label"].astype(int)
        dfs.append(df)
    return dfs[0], dfs[1], dfs[2]


def _counts_by_class(df: pd.DataFrame) -> dict[int, int]:
    return {int(k): int(v) for k, v in df["Label"].value_counts().sort_index().items()}


def check_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                images_dir: str | Path, strict: bool = True) -> dict:
    """Kiểm tra bắt buộc trước khi train (README.md mục 2.1). In ra và trả về dict số liệu.

    1. số ảnh mỗi tập và mỗi lớp (kỳ vọng xấp xỉ 60/20/20).
    2. giao của từng cặp tập theo Filename phải RỖNG.
    3. hợp ba tập phải bằng đúng 17.509 ảnh.
    4. mọi Filename đều tồn tại trong `images_dir`.

    `strict=True` (mặc định, dùng cho bài nộp thật) sẽ assert tổng = 17509.
    `strict=False` chỉ dùng cho smoke-test trên tập con.
    """
    images_dir = Path(images_dir)
    n = {"train": len(train_df), "val": len(val_df), "test": len(test_df)}
    total = sum(n.values())

    sets = {
        "train": set(train_df["Filename"]),
        "val": set(val_df["Filename"]),
        "test": set(test_df["Filename"]),
    }
    overlap = {
        "train∩val": len(sets["train"] & sets["val"]),
        "train∩test": len(sets["train"] & sets["test"]),
        "val∩test": len(sets["val"] & sets["test"]),
    }
    union = len(sets["train"] | sets["val"] | sets["test"])

    per_class = {
        "train": _counts_by_class(train_df),
        "val": _counts_by_class(val_df),
        "test": _counts_by_class(test_df),
    }

    print("=" * 72)
    print("KIỂM TRA CHIA DỮ LIỆU (fold 0)")
    print("=" * 72)
    print(f"Số ảnh: train={n['train']} val={n['val']} test={n['test']} tổng={total} "
          f"(tỉ lệ {n['train']/total:.3f}/{n['val']/total:.3f}/{n['test']/total:.3f})")
    print(f"Giao các tập: {overlap}")
    print(f"Hợp ba tập: {union} ảnh")
    print("\nSố ảnh mỗi lớp (Label: train/val/test):")
    for c in range(NUM_CLASSES):
        print(f"  {c} {CLASS_NAMES[c]:>14s}: "
              f"{per_class['train'].get(c, 0):5d} / {per_class['val'].get(c, 0):5d} / "
              f"{per_class['test'].get(c, 0):5d}")
    print("=" * 72)

    # --- Các assert bắt buộc ---
    for k, v in overlap.items():
        assert v == 0, f"giao {k} khác rỗng ({v} ảnh) - vi phạm quy tắc chia dữ liệu"
    if strict:
        assert union == 17509, f"hợp ba tập = {union}, kỳ vọng 17509 ảnh"

    missing = [f for f in (sets["train"] | sets["val"] | sets["test"])
               if not (images_dir / f).exists()]
    assert not missing, (f"{len(missing)} file trong CSV không tồn tại trong {images_dir}; "
                         f"ví dụ: {missing[:3]}")

    return {"n": n, "per_class": per_class, "overlap": overlap, "union": union,
            "images_dir": str(images_dir)}


def build_transforms(train: bool, img_size: int = 224, aug: str = "basic"):
    """Tạo transform. `aug` chọn mức augmentation (trục B).

    - "basic": RandomResizedCrop + lật ngang.
    - "color": basic + ColorJitter (đổi màu).
    - "trivial": TrivialAugmentWide + RandomResizedCrop + lật ngang.
    - "randaug": RandAugment(num_ops=2, magnitude=9) + RandomResizedCrop + lật ngang.
    Val/test: resize cạnh ngắn về 256 rồi CenterCrop(img_size) + Normalize. KHÔNG ngẫu nhiên.
    (Không dùng lật dọc: hướng sinh trưởng của cỏ là tín hiệu hợp lệ.)
    """
    aug = (aug or "basic").lower()
    if train:
        ops = [transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0))]
        if aug == "color":
            ops.append(transforms.ColorJitter(0.4, 0.4, 0.4, 0.1))
        elif aug == "trivial":
            ops.append(transforms.TrivialAugmentWide())
        elif aug == "randaug":
            ops.append(transforms.RandAugment(num_ops=2, magnitude=9))
        elif aug != "basic":
            raise ValueError(f"aug không hỗ trợ: {aug}")
        ops += [transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
        return transforms.Compose(ops)

    # eval: giữ đúng khung nhìn, không ngẫu nhiên
    resize = max(NATIVE_SIZE, img_size)
    ops = []
    if resize != img_size:
        ops.append(transforms.Resize(resize))
    if resize != img_size:
        ops.append(transforms.CenterCrop(img_size))
    ops += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(ops)


class DeepWeedsDataset(Dataset):
    """Dataset đọc ảnh từ `images_dir` theo DataFrame (Filename, Label).

    __getitem__(i) -> (ảnh đã transform, nhãn int, tên file str).
    """

    def __init__(self, df: pd.DataFrame, images_dir: str | Path, transform=None,
                 preload: bool = False):
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.filenames = self.df["Filename"].tolist()
        self.labels = self.df["Label"].astype(int).tolist()
        self.preload = preload
        self._cache = None
        if preload:
            self._cache = [Image.open(self.images_dir / f).convert("RGB").copy()
                           for f in self.df["Filename"]]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        filename = self.filenames[i]
        label = self.labels[i]
        if self._cache is not None:
            img = self._cache[i]
        else:
            with Image.open(self.images_dir / filename) as source:
                img = source.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label, filename


def _seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def make_loader(df: pd.DataFrame, images_dir: str | Path, transform, batch_size: int,
                train: bool, sampler: str | None = None, num_workers: int = 2,
                seed: int = 0, drop_last: bool | None = None):
    """Tạo DataLoader.

    - train=True: shuffle (hoặc sampler cân bằng); train=False: giữ thứ tự df.
    - sampler="balanced": WeightedRandomSampler trọng số 1/(số ảnh của lớp) (trục D).
    - seed cho worker (worker_init_fn) để tái lập.
    """
    ds = DeepWeedsDataset(df, images_dir, transform)
    if drop_last is None:
        drop_last = bool(train)

    if train and sampler == "balanced":
        counts = df["Label"].value_counts().to_dict()
        w = df["Label"].map(lambda c: 1.0 / counts[c]).to_numpy(dtype=np.float64)
        g = torch.Generator()
        g.manual_seed(seed)
        samp = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double),
                                     num_samples=len(df), replacement=True, generator=g)
        shuffle = False
    else:
        samp = None
        shuffle = bool(train)

    generator = torch.Generator().manual_seed(seed)
    worker_options = ({"persistent_workers": True, "prefetch_factor": 2}
                      if num_workers > 0 else {})

    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, sampler=samp,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(), drop_last=drop_last,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
        generator=generator, **worker_options,
    )
