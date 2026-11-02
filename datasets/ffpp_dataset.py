import os
import random
import numpy as np
from torch.utils.data import Dataset
from PIL import Image


class FFPPDataset(Dataset):
    def __init__(
        self,
        root,
        transform=None,
        split="train",
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
        max_frames_per_video=None,
        balance=False,
    ):
        self.samples = []
        self.transform = transform
        self.root = root
        self.split = split
        self.seed = seed
        self.max_frames_per_video = max_frames_per_video
        self.balance = balance

        if split not in ["train", "val", "test"]:
            raise ValueError(f"Unsupported split: {split}")

        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(
                f"train_ratio + val_ratio + test_ratio must be 1.0, got {total_ratio}"
            )

        rng = random.Random(seed)

        # -----------------------------
        # Collect videos
        # -----------------------------
        self.video_list = []
        for label_name in ["real", "fake"]:
            label = 0 if label_name == "real" else 1
            label_path = os.path.join(root, label_name)

            if not os.path.isdir(label_path):
                raise FileNotFoundError(f"Missing folder: {label_path}")

            for video in sorted(os.listdir(label_path)):
                video_path = os.path.join(label_path, video)
                if os.path.isdir(video_path):
                    self.video_list.append((video, label_name, label))

        # -----------------------------
        # Split by IDENTITY GROUP
        # Fakes are named <method>_<target>_<source> and reuse the real
        # videos <target> and <source>. Keep each pair together with all
        # of its fakes so no identity appears in more than one split.
        # -----------------------------
        parent = {}

        def find(x):
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def video_ids(video, label_name):
            if label_name == "real":
                return [video]
            parts = video.rsplit("_", 2)
            return parts[1:] if len(parts) == 3 else [video]

        for video, label_name, _ in self.video_list:
            ids = video_ids(video, label_name)
            for other in ids[1:]:
                parent[find(other)] = find(ids[0])

        groups = {}
        for item in self.video_list:
            group_id = find(video_ids(item[0], item[1])[0])
            groups.setdefault(group_id, []).append(item)

        group_keys = sorted(groups)
        rng.shuffle(group_keys)

        n_total = len(group_keys)
        train_end = int(n_total * train_ratio)
        val_end = train_end + int(n_total * val_ratio)

        if split == "train":
            selected_keys = group_keys[:train_end]
        elif split == "val":
            selected_keys = group_keys[train_end:val_end]
        else:  # test
            selected_keys = group_keys[val_end:]

        selected_videos = [item for key in selected_keys for item in groups[key]]

        # -----------------------------
        # Expand videos to frames
        # -----------------------------
        for video, label_name, label in selected_videos:
            video_path = os.path.join(root, label_name, video)

            frames = [
                os.path.join(video_path, img)
                for img in sorted(os.listdir(video_path))
                if img.lower().endswith((".jpg", ".jpeg", ".png"))
            ]

            if len(frames) == 0:
                continue

            if self.max_frames_per_video is not None and len(frames) > self.max_frames_per_video:
                idxs = np.linspace(
                    0,
                    len(frames) - 1,
                    self.max_frames_per_video,
                    dtype=int
                )
                frames = [frames[i] for i in idxs]

            for img_path in frames:
                self.samples.append((img_path, label, video))

        # -----------------------------
        # Optional balancing
        # Recommended:
        # - train: balance=True or False
        # - val:   balance=False
        # - test:  balance=False
        # -----------------------------
        if self.balance:
            real_samples = [s for s in self.samples if s[1] == 0]
            fake_samples = [s for s in self.samples if s[1] == 1]

            print(f"Before balance [{split}] -> Real: {len(real_samples)} | Fake: {len(fake_samples)}")

            min_len = min(len(real_samples), len(fake_samples))
            rng.shuffle(real_samples)
            rng.shuffle(fake_samples)

            real_samples = real_samples[:min_len]
            fake_samples = fake_samples[:min_len]

            self.samples = real_samples + fake_samples
            rng.shuffle(self.samples)

            print(f"After balance [{split}] -> Total: {len(self.samples)}")

        print(f"{split.upper()} samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, video = self.samples[idx]

        with Image.open(img_path) as img:
            image = img.convert("RGB")

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "label": label,
            "path": img_path,
            "video": video,
        }