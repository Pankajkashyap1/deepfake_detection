from pathlib import Path
import random
from collections import defaultdict
from PIL import Image
from torch.utils.data import Dataset


class DeepFakeFaceDataset(Dataset):
    """
    Generic loader for diffusion-face dataset:
    - one real folder (e.g. wiki)
    - multiple fake folders
    - each contains many subfolders with images

    Split is group-based using the first subfolder level under each source folder.
    """

    IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    def __init__(
        self,
        root,
        transform=None,
        split="train",
        real_dir="wiki",
        fake_dirs=None,
        train_ratio=0.8,
        val_ratio=0.1,
        seed=42,
        max_images_per_group=None,
    ):
        self.root = Path(root)
        self.transform = transform
        self.split = split
        self.real_dir = real_dir
        self.fake_dirs = fake_dirs
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1.0 - train_ratio - val_ratio
        self.seed = seed
        self.max_images_per_group = max_images_per_group
        self.samples = []

        if split not in ["train", "val", "test"]:
            raise ValueError("split must be train, val, or test")

        if self.test_ratio < 0:
            raise ValueError("train_ratio + val_ratio must be <= 1.0")

        if not self.root.exists():
            raise FileNotFoundError(f"Missing root: {self.root}")

        all_top_dirs = sorted([p.name for p in self.root.iterdir() if p.is_dir()])

        if fake_dirs is None:
            fake_dirs = [d for d in all_top_dirs if d != real_dir]

        self.fake_dirs = fake_dirs

        groups = []
        groups.extend(self._collect_groups(self.real_dir, label=0))
        for fake_dir in self.fake_dirs:
            groups.extend(self._collect_groups(fake_dir, label=1))

        # Fake dirs reuse the real identities under the same subfolder and
        # filename, so split by subfolder name across all top dirs to keep a
        # real image and its fakes in the same split.
        group_keys = sorted({item["rel_group"].split("/", 1)[1] for item in groups})

        rng = random.Random(seed)
        rng.shuffle(group_keys)

        n = len(group_keys)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        if split == "train":
            selected_keys = set(group_keys[:train_end])
        elif split == "val":
            selected_keys = set(group_keys[train_end:val_end])
        else:
            selected_keys = set(group_keys[val_end:])

        selected_groups = [g for g in groups if g["rel_group"].split("/", 1)[1] in selected_keys]

        for item in selected_groups:
            label = item["label"]
            rel_group = item["rel_group"]
            image_paths = item["images"]

            if self.max_images_per_group is not None and len(image_paths) > self.max_images_per_group:
                rng.shuffle(image_paths)
                image_paths = image_paths[:self.max_images_per_group]

            for img_path in image_paths:
                self.samples.append((str(img_path), label, rel_group))

        print(f"DeepFakeFace {split.upper()} samples: {len(self.samples)}")

    def _collect_groups(self, top_dir_name, label):
        top_dir = self.root / top_dir_name
        if not top_dir.exists():
            print(f"[WARN] Missing top-level dir: {top_dir}")
            return []

        # first-level subfolders are treated as groups
        subdirs = [p for p in sorted(top_dir.iterdir()) if p.is_dir()]

        groups = []

        # If images are directly inside top_dir, treat top_dir as one group
        direct_images = [p for p in top_dir.iterdir() if p.is_file() and p.suffix.lower() in self.IMG_EXTS]
        if len(direct_images) > 0:
            groups.append({
                "label": label,
                "rel_group": f"{top_dir_name}/__root__",
                "images": sorted(direct_images),
            })

        for subdir in subdirs:
            images = sorted([
                p for p in subdir.rglob("*")
                if p.is_file() and p.suffix.lower() in self.IMG_EXTS
            ])
            if len(images) == 0:
                continue

            groups.append({
                "label": label,
                "rel_group": f"{top_dir_name}/{subdir.name}",
                "images": images,
            })

        return groups

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, group_rel = self.samples[idx]

        with Image.open(img_path) as img:
            image = img.convert("RGB")

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "label": label,
            "path": img_path,
            "video": group_rel,   # keep trainer/eval interface consistent
        }