import os
import cv2
from tqdm import tqdm
import mediapipe as mp

INPUT_ROOT = "/data/datasets/deepfake_datasets/FF_frames"
OUTPUT_ROOT = "/data/datasets/deepfake_datasets/FF_faces"

IMG_EXTS = (".jpg", ".jpeg", ".png")
MARGIN_RATIO = 0.35
FALLBACK_CENTER_CROP = True
MIN_FACE_SIZE = 40

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


def crop_face(image_path):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return None, "read_failed"

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


def process_dataset(input_root, output_root):
    total = 0
    saved = 0
    mp_used = 0
    fallback = 0
    failed = 0

    for cls in ["real", "fake"]:
        cls_in = os.path.join(input_root, cls)
        cls_out = os.path.join(output_root, cls)

        if not os.path.isdir(cls_in):
            print(f"Missing class folder: {cls_in}")
            continue

        for video in tqdm(sorted(os.listdir(cls_in)), desc=f"Processing {cls}"):
            in_dir = os.path.join(cls_in, video)
            out_dir = os.path.join(cls_out, video)

            if not os.path.isdir(in_dir):
                continue

            os.makedirs(out_dir, exist_ok=True)

            for fname in sorted(os.listdir(in_dir)):
                if not fname.lower().endswith(IMG_EXTS):
                    continue

                total += 1
                in_path = os.path.join(in_dir, fname)
                out_path = os.path.join(out_dir, fname)

                crop, status = crop_face(in_path)

                if crop is None:
                    failed += 1
                    continue

                ok = cv2.imwrite(out_path, crop)
                if ok:
                    saved += 1
                    if status == "mediapipe":
                        mp_used += 1
                    elif status == "fallback":
                        fallback += 1
                else:
                    failed += 1

    print("\n===== FACE EXTRACTION SUMMARY =====")
    print(f"Total images    : {total}")
    print(f"Saved crops     : {saved}")
    print(f"MediaPipe crops : {mp_used}")
    print(f"Fallback crops  : {fallback}")
    print(f"Failed          : {failed}")


if __name__ == "__main__":
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    process_dataset(INPUT_ROOT, OUTPUT_ROOT)