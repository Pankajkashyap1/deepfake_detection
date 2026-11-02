
import torch
from torch.utils.data import DataLoader
from models import DeepfakeVLMDetector

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"\n🚀 SANITY CHECK on {DEVICE}\n")

# -----------------------------
# Dummy Dataset
# -----------------------------
class DummyDataset:
    def __len__(self):
        return 8

    def __getitem__(self, idx):
        return {
            "image": torch.randn(3, 224, 224),
            "label": torch.tensor(idx % 2),
            "explanation": "This is a fake face with artifacts." if idx % 2 else "This is a real face."
        }

dataset = DummyDataset()
loader = DataLoader(dataset, batch_size=2)

# -----------------------------
# Model
# -----------------------------
model = DeepfakeVLMDetector({
    "embed_dim": 768,
    "use_llm": True,
    "llm_model": "gpt2"
}).to(DEVICE)

model.set_stage(2)

# -----------------------------
# Batch
# -----------------------------
batch = next(iter(loader))

images = batch["image"].to(DEVICE)
labels = batch["label"].to(DEVICE)
explanations = batch["explanation"]

print("📦 Input:", images.shape)

# -----------------------------
# Forward
# -----------------------------
outputs = model(images, labels=labels, explanations=explanations)


# -----------------------------
# Prepare image
# -----------------------------
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Prepare image
# -----------------------------
img = images[0].detach().cpu().permute(1, 2, 0).numpy()
img = (img - img.min()) / (img.max() - img.min() + 1e-6)

# -----------------------------
# Prepare heatmap
# -----------------------------
heatmap = outputs["heatmap"][0].detach().cpu().numpy()
heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-6)

# -----------------------------
# Overlay and SAVE
# -----------------------------
plt.figure(figsize=(6,6))
plt.imshow(img)
plt.imshow(heatmap, cmap='jet', alpha=0.5)
plt.axis("off")

plt.savefig("heatmap_overlay.png", bbox_inches='tight')
print("✅ Saved heatmap as heatmap_overlay.png")