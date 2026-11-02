import numpy as np
import matplotlib.pyplot as plt
import cv2


def normalize_heatmap(heatmap):
    """Normalize heatmap to [0,1]."""
    heatmap = heatmap.astype(np.float32)
    heatmap = heatmap - heatmap.min()
    heatmap = heatmap / (heatmap.max() + 1e-6)
    return heatmap


def overlay_heatmap(image, heatmap, alpha=0.5):
    """Overlay heatmap on image safely."""
    image = np.array(image)

    # ensure RGB image
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    heatmap = normalize_heatmap(heatmap)

    # resize heatmap to image HxW if needed
    h, w = image.shape[:2]
    if heatmap.shape[:2] != (h, w):
        heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)

    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

    overlay = cv2.addWeighted(image, 1 - alpha, heatmap_color, alpha, 0)
    return overlay


def visualize_result(image, heatmap, pred, save_path=None):
    """Full visualization pipeline."""
    image_np = np.array(image)
    heatmap = normalize_heatmap(heatmap)

    # resize heatmap for standalone display too
    h, w = image_np.shape[:2]
    if heatmap.shape[:2] != (h, w):
        heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)

    overlay = overlay_heatmap(image, heatmap)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(image_np)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(heatmap, cmap="jet")
    plt.title("Heatmap")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title(f"Prediction: {pred}")
    plt.axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()