import os
import argparse
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from facenet_pytorch import MTCNN

# Celeb-DF videos -> face crops in the layout CelebDFDataset expects:
#   ROOT/<label>/frames/<video_stem>/<frame_idx>.png
# Crops match FF_fixed: 32 evenly spaced frames, square, 256x256 PNG.

ROOT = "/data/datasets/deepfake_datasets/Celeb-DF-v2"
LABEL_DIRS = ["Celeb-real", "Celeb-synthesis", "YouTube-real"]

FRAMES_PER_VIDEO = 32
OUT_SIZE = 256
MARGIN_RATIO = 0.35
MIN_FACE_SIZE = 40
MIN_FACE_PROB = 0.9
FALLBACK_CENTER_CROP = True
NUM_WORKERS = 2
# detect on downscaled frames (Celeb-DF faces are >100px), crop at full resolution
DETECT_SCALE = 0.5


def expand_box(x1, y1, x2, y2, w, h, margin_ratio=0.35):
    bw = x2 - x1
    bh = y2 - y1

    # square box around the face center, with margin
    side = max(bw, bh) * (1 + 2 * margin_ratio)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    nx1 = int(max(0, cx - side / 2))
    ny1 = int(max(0, cy - side / 2))
    nx2 = int(min(w, cx + side / 2))
    ny2 = int(min(h, cy + side / 2))
    return nx1, ny1, nx2, ny2


def center_crop_fallback(img):
    h, w = img.shape[:2]
    crop_w = int(w * 0.5)
    crop_h = int(h * 0.7)
    cx = w // 2
    cy = int(h * 0.42)
    x1 = max(0, cx - crop_w // 2)
    y1 = max(0, cy - crop_h // 2)
    x2 = min(w, x1 + crop_w)
    y2 = min(h, y1 + crop_h)
    return img[y1:y2, x1:x2]


class VideoFrames(Dataset):
    """Decodes FRAMES_PER_VIDEO evenly spaced RGB frames per video in worker processes."""

    def __init__(self, videos):
        self.videos = videos

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video_path, out_dir = self.videos[idx]
        cap = cv2.VideoCapture(video_path)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        wanted = set(np.linspace(0, max(n - 1, 0), FRAMES_PER_VIDEO, dtype=int).tolist())

        frames, frame_ids = [], []
        i = 0
        while len(frames) < len(wanted):
            ok, frame = cap.read()
            if not ok:
                break
            if i in wanted:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frame_ids.append(i)
            i += 1
        cap.release()
        return frames, frame_ids, out_dir


def keep_numpy(item):
    # default collate would turn frames into tensors, which MTCNN cannot batch
    return item


def collect_videos():
    videos = []
    for label in LABEL_DIRS:
        label_dir = os.path.join(ROOT, label)
        if not os.path.isdir(label_dir):
            print(f"Missing label folder: {label_dir}")
            continue
        for fname in sorted(os.listdir(label_dir)):
            if not fname.endswith(".mp4"):
                continue
            out_dir = os.path.join(label_dir, "frames", os.path.splitext(fname)[0])
            # resume: skip videos that already have all their frames
            if os.path.isdir(out_dir) and len(os.listdir(out_dir)) >= FRAMES_PER_VIDEO:
                continue
            videos.append((os.path.join(label_dir, fname), out_dir))
    return videos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = MTCNN(keep_all=True, device=device, min_face_size=int(MIN_FACE_SIZE * DETECT_SCALE))

    videos = collect_videos()[args.shard::args.num_shards]
    print(f"Videos to process (shard {args.shard}/{args.num_shards}): {len(videos)}")

    loader = DataLoader(
        VideoFrames(videos),
        batch_size=None,
        num_workers=NUM_WORKERS,
        collate_fn=keep_numpy,
    )

    saved = 0
    detected = 0
    fallback = 0
    failed_videos = 0

    for frames, frame_ids, out_dir in tqdm(loader, total=len(videos)):
        if len(frames) == 0:
            failed_videos += 1
            continue

        os.makedirs(out_dir, exist_ok=True)
        small = [
            cv2.resize(f, None, fx=DETECT_SCALE, fy=DETECT_SCALE, interpolation=cv2.INTER_AREA)
            for f in frames
        ]
        boxes, probs = detector.detect(small)

        for frame, frame_id, frame_boxes, frame_probs in zip(frames, frame_ids, boxes, probs):
            h, w = frame.shape[:2]
            best_box = None
            best_score = -1.0

            if frame_boxes is not None:
                for box, score in zip(frame_boxes, frame_probs):
                    x1, y1, x2, y2 = box / DETECT_SCALE
                    if score < MIN_FACE_PROB or x2 - x1 < MIN_FACE_SIZE or y2 - y1 < MIN_FACE_SIZE:
                        continue
                    if score > best_score:
                        best_score = score
                        best_box = (x1, y1, x2, y2)

            if best_box is not None:
                x1, y1, x2, y2 = expand_box(*best_box, w, h, MARGIN_RATIO)
                crop = frame[y1:y2, x1:x2]
                detected += 1
            elif FALLBACK_CENTER_CROP:
                crop = center_crop_fallback(frame)
                fallback += 1
            else:
                continue

            crop = cv2.resize(crop, (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_AREA)
            out_path = os.path.join(out_dir, f"{frame_id:03d}.png")
            if cv2.imwrite(out_path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)):
                saved += 1

    print("\n===== FACE EXTRACTION SUMMARY =====")
    print(f"Saved frames   : {saved}")
    print(f"Face detected  : {detected}")
    print(f"Center fallback: {fallback}")
    print(f"Failed videos  : {failed_videos}")


if __name__ == "__main__":
    main()
