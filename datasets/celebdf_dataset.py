from pathlib import Path
import random
import numpy as np
from torch.utils.data import Dataset
from PIL import Image


class CelebDFDataset(Dataset):
    def __init__(
        self,
        root,
        transform=None,
        split="test",
        list_file="List_of_testing_videos.txt",
        train_ratio=0.85,
        seed=42,
        max_frames_per_video=None,
    ):
        self.root = Path(root)
        self.transform = transform
        self.split = split
        self.max_frames_per_video = max_frames_per_video
        self.samples = []

        if split not in ["train", "val", "test"]:
            raise ValueError("split must be one of: train, val, test")

        list_path = self.root / list_file
        if not list_path.exists():
            raise FileNotFoundError(f"Missing file: {list_path}")

        with open(list_path, "r") as f:
            raw_entries = [line.strip() for line in f if line.strip()]

        # Parse official test list entries
        official_test_videos = set()
        for line in raw_entries:
            parts = line.split(maxsplit=1)
            rel_video_path = parts[1] if len(parts) == 2 else parts[0]
            rel_video_path = rel_video_path.strip()
            rel_path_obj = Path(rel_video_path)
            if len(rel_path_obj.parts) < 2:
                continue
            label_name = rel_path_obj.parts[0]
            video_stem = rel_path_obj.stem
            official_test_videos.add((label_name, video_stem))

        # Collect all available videos from frames folders
        all_videos = []
        for label_name in ["Celeb-real", "Celeb-synthesis", "YouTube-real"]:
            label = 1 if label_name == "Celeb-synthesis" else 0
            frames_root = self.root / label_name / "frames"
            if not frames_root.exists():
                continue

            for video_dir in sorted([p for p in frames_root.iterdir() if p.is_dir()]):
                video_stem = video_dir.name
                all_videos.append((label_name, video_stem, label))

        # Split
        if split == "test":
            selected_videos = [
                (label_name, video_stem, label)
                for (label_name, video_stem, label) in all_videos
                if (label_name, video_stem) in official_test_videos
            ]
        else:
            trainval_videos = [
                (label_name, video_stem, label)
                for (label_name, video_stem, label) in all_videos
                if (label_name, video_stem) not in official_test_videos
            ]

            rng = random.Random(seed)
            rng.shuffle(trainval_videos)

            split_idx = int(len(trainval_videos) * train_ratio)
            if split == "train":
                selected_videos = trainval_videos[:split_idx]
            else:
                selected_videos = trainval_videos[split_idx:]

        # Expand videos to frames
        for label_name, video_stem, label in selected_videos:
            frames_dir = self.root / label_name / "frames" / video_stem
            if not frames_dir.exists():
                continue

            frames = sorted([
                p for p in frames_dir.iterdir()
                if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
            ])

            if len(frames) == 0:
                continue

            if self.max_frames_per_video is not None and len(frames) > self.max_frames_per_video:
                idxs = np.linspace(0, len(frames) - 1, self.max_frames_per_video, dtype=int)
                frames = [frames[i] for i in idxs]

            rel_video = f"{label_name}/{video_stem}"
            for frame_path in frames:
                self.samples.append((str(frame_path), label, rel_video))

        print(f"CELEB-DF {split.upper()} samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, video_rel = self.samples[idx]

        with Image.open(img_path) as img:
            image = img.convert("RGB")

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "label": label,
            "path": img_path,
            "video": video_rel,
        }