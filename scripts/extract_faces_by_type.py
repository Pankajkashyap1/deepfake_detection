import os
import cv2
from tqdm import tqdm
import mediapipe as mp

INPUT_ROOT = "/data/datasets/FaceForensics++"
OUTPUT_ROOT = "/data/datasets/deepfake_datasets/FF_faces_by_type"

VIDEO_CATEGORIES = [
    "Deepfakes",
    "Face2Face",
    "FaceShifter",
    "FaceSwap",
    "NeuralTextures",
    "Original",
]

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")
MARGIN_RATIO = 0.35
MIN_FACE_SIZE = 40
FALLBACK_CENTER_CROP = True

mp_face = mp.solutions.face_detection
detector = mp_face.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.5
)


def expand_box(x1, y1, x2, y2, w, h, margin_ratio=0.35):
    bw = x2 - x1
    bh = y2 - y1

    mx = int(bw * margin_ratio)
    my = int(bh * margin_ratio)

    nx1 = max(0, x1 - mx)
    ny1 = max(0, y1 - my)
    nx2 = min(w, x2 + mx)
    ny2 = min(h, y2 + my)

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


def crop_face_from_bgr(img_bgr):
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    results = detector.process(img_rgb)

    if not results.detections:
        if FALLBACK_CENTER_CROP:
            return center_crop_fallback(img_bgr), "fallback"
        return None, "no_face"

    best_box = None
    best_score = -1.0

    for det in results.detections:
        score = float(det.score[0])
        bbox = det.location_data.relative_bounding_box

        x1 = int(bbox.xmin * w)
        y1 = int(bbox.ymin * h)
        bw = int(bbox.width * w)
        bh = int(bbox.height * h)
        x2 = x1 + bw
        y2 = y1 + bh

        if bw < MIN_FACE_SIZE or bh < MIN_FACE_SIZE:
            continue

        if score > best_score:
            best_score = score
            best_box = (x1, y1, x2, y2)

    if best_box is None:
        if FALLBACK_CENTER_CROP:
            return center_crop_fallback(img_bgr), "fallback"
        return None, "small_face"

    x1, y1, x2, y2 = expand_box(*best_box, w, h, MARGIN_RATIO)

    crop = img_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        if FALLBACK_CENTER_CROP:
            return center_crop_fallback(img_bgr), "fallback"
        return None, "empty_crop"

    return crop, "mediapipe"


def process_video(video_path, out_dir):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"saved": 0, "mediapipe": 0, "fallback": 0, "failed": 1}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return {"saved": 0, "mediapipe": 0, "fallback": 0, "failed": 1}

    middle_idx = total_frames // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_idx)

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        return {"saved": 0, "mediapipe": 0, "fallback": 0, "failed": 1}

    crop, status = crop_face_from_bgr(frame)
    if crop is None:
        return {"saved": 0, "mediapipe": 0, "fallback": 0, "failed": 1}

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "000.jpg")

    ok = cv2.imwrite(out_path, crop)
    if not ok:
        return {"saved": 0, "mediapipe": 0, "fallback": 0, "failed": 1}

    return {
        "saved": 1,
        "mediapipe": 1 if status == "mediapipe" else 0,
        "fallback": 1 if status == "fallback" else 0,
        "failed": 0,
    }


def process_dataset():
    total_videos = 0
    total_saved = 0
    total_mediapipe = 0
    total_fallback = 0
    total_failed = 0

    for category in VIDEO_CATEGORIES:
        in_dir = os.path.join(INPUT_ROOT, category)
        out_cat_dir = os.path.join(OUTPUT_ROOT, category)

        if not os.path.isdir(in_dir):
            print(f"Missing category: {in_dir}")
            continue

        video_files = sorted(
            [f for f in os.listdir(in_dir) if f.lower().endswith(VIDEO_EXTS)]
        )

        print(f"\nProcessing {category}: {len(video_files)} videos")

        for fname in tqdm(video_files, desc=f"{category}"):
            total_videos += 1

            video_path = os.path.join(in_dir, fname)
            video_id = os.path.splitext(fname)[0]
            out_dir = os.path.join(out_cat_dir, video_id)

            stats = process_video(video_path, out_dir)

            total_saved += stats["saved"]
            total_mediapipe += stats["mediapipe"]
            total_fallback += stats["fallback"]
            total_failed += stats["failed"]

    print("\n===== FACE EXTRACTION BY TYPE SUMMARY =====")
    print(f"Total videos      : {total_videos}")
    print(f"Saved face crops  : {total_saved}")
    print(f"MediaPipe crops   : {total_mediapipe}")
    print(f"Fallback crops    : {total_fallback}")
    print(f"Failed videos     : {total_failed}")


if __name__ == "__main__":
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    process_dataset()