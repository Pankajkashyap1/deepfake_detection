import csv
import json
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
from PIL import Image


class DFDCDataset(Dataset):
    def __init__(
        self,
        frames_root,
        transform=None,
        labels_path=None,
        max_frames_per_video=None,
    ):
        self.frames_root = Path(frames_root)
        self.transform = transform
        self.labels_path = labels_path
        self.max_frames_per_video = max_frames_per_video
        self.samples = []
        self.video_label_map = {}

        if not self.frames_root.exists():
            raise FileNotFoundError(f"Missing frames root: {self.frames_root}")

        if labels_path is not None:
            self.video_label_map = self._load_labels(labels_path)

        video_dirs = sorted([p for p in self.frames_root.iterdir() if p.is_dir()])

        for video_dir in video_dirs:
            video_name = video_dir.name
            label = self.video_label_map.get(video_name, None)

            frames = sorted([
                p for p in video_dir.iterdir()
                if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
            ])

            if len(frames) == 0:
                continue

            if self.max_frames_per_video is not None and len(frames) > self.max_frames_per_video:
                idxs = np.linspace(0, len(frames) - 1, self.max_frames_per_video, dtype=int)
                frames = [frames[i] for i in idxs]

            for frame_path in frames:
                self.samples.append((str(frame_path), label, video_name))

        print(f"DFDC samples: {len(self.samples)}")
        print(f"DFDC videos : {len(video_dirs)}")
        if labels_path is None:
            print("Labels: not provided (prediction-only mode)")
        else:
            print(f"Labeled videos: {len(self.video_label_map)}")

    def _load_labels(self, labels_path):
        labels_path = Path(labels_path)
        if not labels_path.exists():
            raise FileNotFoundError(f"Missing labels file: {labels_path}")

        if labels_path.suffix.lower() == ".json":
            with open(labels_path, "r") as f:
                data = json.load(f)

            label_map = {}
            for k, v in data.items():
                video_stem = Path(k).stem

                # DFDC metadata.json style
                # {
                #   "aalscayrfi.mp4": {
                #       "augmentations": {...},
                #       "is_fake": 0
                #   }
                # }
                if isinstance(v, dict):
                    if "is_fake" in v:
                        raw = v["is_fake"]
                    elif "label" in v:
                        raw = v["label"]
                    elif "class" in v:
                        raw = v["class"]
                    else:
                        raw = None
                else:
                    raw = v

                if raw is None:
                    continue

                label_map[video_stem] = self._normalize_label(raw)

            return label_map

        elif labels_path.suffix.lower() == ".csv":
            label_map = {}
            with open(labels_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                cols = reader.fieldnames or []

                video_col = None
                label_col = None
                for c in cols:
                    cl = c.lower()
                    if cl in ["video", "filename", "file", "path", "name"]:
                        video_col = c
                    if cl in ["label", "class", "target", "is_fake"]:
                        label_col = c

                if video_col is None or label_col is None:
                    raise ValueError(f"Could not infer video/label columns from CSV columns: {cols}")

                for row in reader:
                    video_stem = Path(row[video_col]).stem
                    label_map[video_stem] = self._normalize_label(row[label_col])

            return label_map

        else:
            raise ValueError(f"Unsupported label file format: {labels_path}")

    def _normalize_label(self, raw):
        if raw is None:
            return None

        if isinstance(raw, (int, float)):
            return int(raw)

        s = str(raw).strip().lower()

        if s in ["fake", "1", "true"]:
            return 1
        if s in ["real", "0", "false"]:
            return 0

        raise ValueError(f"Unsupported label value: {raw}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, video = self.samples[idx]

        with Image.open(img_path) as img:
            image = img.convert("RGB")

        if self.transform:
            image = self.transform(image)

        item = {
            "image": image,
            "path": img_path,
            "video": video,
        }

        if label is not None:
            item["label"] = label

        return item