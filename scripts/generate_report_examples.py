import os
import sys
import json
import argparse
import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.cm as cm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.deepfake_detector import DeepfakeVLMDetector
from datasets.transforms import get_transforms


def build_model_config(cfg):
    return OmegaConf.create({
        **OmegaConf.to_container(cfg.model, resolve=True),
        **OmegaConf.to_container(cfg.fusion, resolve=True),
    })


def load_detector(ckpt_path: str, config_path: str, device: str):
    cfg = OmegaConf.load(config_path)
    model_cfg = build_model_config(cfg)

    model = DeepfakeVLMDetector(model_cfg)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    print("Missing keys count   :", len(missing))
    print("Unexpected keys count:", len(unexpected))

    model.set_stage(1)
    model.to(device)
    model.eval()
    return model


def load_llm(model_name: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
    )
    llm.to(device)
    llm.eval()
    return tokenizer, llm


def run_detector(model, image_tensor, threshold: float):
    with torch.no_grad():
        outputs = model(image_tensor)
        logits = outputs["logits"]
        probs = torch.softmax(logits, dim=1)

        fake_prob = probs[0, 1].item()
        real_prob = probs[0, 0].item()

        pred_idx = 1 if fake_prob >= threshold else 0
        pred_label = "FAKE" if pred_idx == 1 else "REAL"
        confidence = fake_prob if pred_idx == 1 else real_prob

        heatmap = outputs["localizer_heatmap"][0].detach().cpu().numpy()

    return pred_label, confidence, fake_prob, heatmap


def heatmap_to_region(heatmap: np.ndarray):
    h, w = heatmap.shape
    hs, ws = h // 3, w // 3

    region_names = {
        (0, 0): "left forehead / left eye",
        (0, 1): "forehead / eyes",
        (0, 2): "right forehead / right eye",
        (1, 0): "left cheek",
        (1, 1): "nose",
        (1, 2): "right cheek",
        (2, 0): "left jaw",
        (2, 1): "mouth / chin",
        (2, 2): "right jaw",
    }

    best_score = -1.0
    best_region = "central face"

    for i in range(3):
        for j in range(3):
            patch = heatmap[i * hs:(i + 1) * hs, j * ws:(j + 1) * ws]
            score = float(patch.mean())
            if score > best_score:
                best_score = score
                best_region = region_names[(i, j)]

    return best_region, best_score


def build_evidence(pred_label: str, fake_prob: float, region: str):
    if pred_label == "FAKE":
        confidence = "high" if fake_prob >= 0.85 else "moderate"
        evidence_type = "suspicious local inconsistency"
    else:
        confidence = "high" if fake_prob <= 0.15 else "moderate"
        evidence_type = "visually consistent appearance"

    return {
        "prediction": pred_label,
        "fake_prob": round(fake_prob, 4),
        "confidence": confidence,
        "region": region,
        "evidence_type": evidence_type,
    }


SYSTEM_PROMPT = """You are a forensic face analysis assistant.
Write a short explanation that must be consistent with the detector prediction.
If prediction is REAL, explain visual consistency and the absence of strong manipulation evidence.
If prediction is FAKE, explain the suspicious facial region and possible manipulation evidence.
Do not contradict the prediction.
Use cautious, evidence-based language.
Keep the answer to 2 sentences maximum.
"""


def build_user_prompt(evidence: dict):
    return f"""Prediction: {evidence['prediction']}
Fake probability: {evidence['fake_prob']}
Confidence: {evidence['confidence']}
Most suspicious region: {evidence['region']}
Evidence type: {evidence['evidence_type']}

Instruction:
- REAL -> explain why the face appears consistent.
- FAKE -> explain why the highlighted region may indicate manipulation.
Write the final explanation only.
"""


def generate_llm_explanation(tokenizer, llm, evidence: dict, device: str):
    user_prompt = build_user_prompt(evidence)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        prompt_text = SYSTEM_PROMPT + "\n\n" + user_prompt + "\nExplanation:"

    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    ).to(device)

    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = llm.generate(
            **inputs,
            max_new_tokens=60,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    gen_ids = outputs[0][input_len:]
    decoded = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return decoded


def safe_fallback(pred_label: str, confidence: str, region: str):
    if pred_label == "FAKE":
        return (
            f"The image is predicted as fake because the highlighted region around the {region} "
            f"shows suspicious local inconsistencies that may indicate manipulation. "
            f"The model confidence is {confidence}."
        )
    else:
        return (
            f"The image is predicted as real because the highlighted region around the {region} "
            f"appears visually consistent and does not show strong manipulation evidence. "
            f"The model confidence is {confidence}."
        )


def consistency_guard(pred_label: str, text: str, confidence: str, region: str):
    text_low = text.lower().strip()

    banned_markers = [
        "prediction:",
        "fake probability:",
        "real probability:",
        "confidence:",
        "most suspicious region:",
        "evidence type:",
        "instruction:",
        "write the final",
    ]
    if any(marker in text_low for marker in banned_markers):
        return safe_fallback(pred_label, confidence, region)

    bad_for_real = any(
        phrase in text_low for phrase in [
            "predicted as fake",
            "appears fake",
            "is fake",
            "forged",
            "manipulation artifact",
            "suspicious inconsisten"
        ]
    )

    bad_for_fake = any(
        phrase in text_low for phrase in [
            "predicted as real",
            "appears real",
            "is real",
            "healthy individual",
            "strong visual consistency",
            "no manipulation evidence",
            "visually consistent"
        ]
    )

    if pred_label == "REAL" and bad_for_real:
        return safe_fallback(pred_label, confidence, region)

    if pred_label == "FAKE" and bad_for_fake:
        return safe_fallback(pred_label, confidence, region)

    if len(text.strip()) < 12:
        return safe_fallback(pred_label, confidence, region)

    return text.strip()


def save_overlay(image_pil: Image.Image, heatmap: np.ndarray, out_path: str, alpha: float = 0.45):
    image = np.array(image_pil).astype(np.float32) / 255.0

    h, w = image.shape[:2]
    heatmap = np.clip(heatmap, 0, 1)

    if heatmap.shape[0] != h or heatmap.shape[1] != w:
        heatmap = np.array(Image.fromarray((heatmap * 255).astype(np.uint8)).resize((w, h))) / 255.0

    colored = cm.jet(heatmap)[..., :3]
    overlay = (1 - alpha) * image + alpha * colored
    overlay = np.clip(overlay, 0, 1)

    Image.fromarray((overlay * 255).astype(np.uint8)).save(out_path)


def process_one(model, tokenizer, llm, image_path, config, threshold, output_dir, device):
    os.makedirs(output_dir, exist_ok=True)

    image = Image.open(image_path).convert("RGB")
    transform = get_transforms(train=False)
    image_tensor = transform(image).unsqueeze(0).to(device)

    pred_label, confidence_score, fake_prob, heatmap = run_detector(model, image_tensor, threshold)
    region, region_score = heatmap_to_region(heatmap)
    evidence = build_evidence(pred_label, fake_prob, region)

    raw_explanation = generate_llm_explanation(tokenizer, llm, evidence, device)
    final_explanation = consistency_guard(
        pred_label=pred_label,
        text=raw_explanation,
        confidence=evidence["confidence"],
        region=region
    )

    original_out = os.path.join(output_dir, "original.png")
    overlay_out = os.path.join(output_dir, "overlay.png")
    json_out = os.path.join(output_dir, "result.json")
    txt_out = os.path.join(output_dir, "explanation.txt")

    image.save(original_out)
    save_overlay(image, heatmap, overlay_out, alpha=0.45)

    result = {
        "image_path": image_path,
        "prediction": pred_label,
        "confidence_percent": round(confidence_score * 100, 2),
        "fake_prob": round(fake_prob, 4),
        "most_active_region": region,
        "region_score": round(region_score, 4),
        "explanation": final_explanation,
    }

    with open(json_out, "w") as f:
        json.dump(result, f, indent=2)

    with open(txt_out, "w") as f:
        f.write(final_explanation + "\n")

    print("\n===== FINAL RESULT =====")
    print(f"Image              : {image_path}")
    print(f"Prediction         : {pred_label}")
    print(f"Confidence         : {confidence_score:.2%}")
    print(f"Fake prob          : {fake_prob:.4f}")
    print(f"Most active region : {region}")
    print(f"Region score       : {region_score:.4f}")
    print(f"Explanation        : {final_explanation}")
    print(f"Saved original     : {original_out}")
    print(f"Saved overlay      : {overlay_out}")
    print(f"Saved result json  : {json_out}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Generate final report examples from deepfake_vlm detector")
    parser.add_argument("--config", type=str, default="configs/train_stage1.yaml")
    parser.add_argument("--ckpt", type=str, default="checkpoints/stage1_final_detector_no_reasoning.pth")
    parser.add_argument("--threshold", type=float, default=0.29)
    parser.add_argument("--llm_model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--output_root", type=str, default="results/report_examples")
    parser.add_argument("--images", nargs="+", required=True)
    parser.add_argument("--names", nargs="+", required=True)
    args = parser.parse_args()

    if len(args.images) != len(args.names):
        raise ValueError("--images and --names must have the same length")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = load_detector(args.ckpt, args.config, device)
    tokenizer, llm = load_llm(args.llm_model, device)

    os.makedirs(args.output_root, exist_ok=True)

    manifest = {}
    for image_path, name in zip(args.images, args.names):
        out_dir = os.path.join(args.output_root, name)
        manifest[name] = process_one(
            model=model,
            tokenizer=tokenizer,
            llm=llm,
            image_path=image_path,
            config=args.config,
            threshold=args.threshold,
            output_dir=out_dir,
            device=device
        )

    with open(os.path.join(args.output_root, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()