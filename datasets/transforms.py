import io
import random
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T


class RandomJPEGCompression:
    def __init__(self, p=0.5, quality_range=(35, 95)):
        self.p = p
        self.quality_range = quality_range

    def __call__(self, img):
        if random.random() > self.p:
            return img

        quality = random.randint(*self.quality_range)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")


class RandomDownUpSample:
    def __init__(self, p=0.4, scale_range=(0.5, 0.9)):
        self.p = p
        self.scale_range = scale_range

    def __call__(self, img):
        if random.random() > self.p:
            return img

        w, h = img.size
        scale = random.uniform(*self.scale_range)
        new_w = max(32, int(w * scale))
        new_h = max(32, int(h * scale))

        img_small = img.resize((new_w, new_h), Image.BILINEAR)
        img_back = img_small.resize((w, h), Image.BILINEAR)
        return img_back


class AddGaussianNoise:
    def __init__(self, p=0.3, std_range=(0.0, 0.03)):
        self.p = p
        self.std_range = std_range

    def __call__(self, tensor):
        if random.random() > self.p:
            return tensor

        std = random.uniform(*self.std_range)
        noise = torch.randn_like(tensor) * std
        tensor = tensor + noise
        return tensor.clamp(0.0, 1.0)


def get_transforms(train=True, domain_generalization=False):
    clip_mean = [0.48145466, 0.4578275, 0.40821073]
    clip_std = [0.26862954, 0.26130258, 0.27577711]

    if train and domain_generalization:
        return T.Compose([
            T.Resize((224, 224)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomResizedCrop(
                224,
                scale=(0.85, 1.0),
                ratio=(0.95, 1.05)
            ),
            T.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.10,
                hue=0.03
            ),
            RandomJPEGCompression(p=0.5, quality_range=(35, 95)),
            RandomDownUpSample(p=0.4, scale_range=(0.5, 0.9)),
            T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.2),
            T.ToTensor(),
            T.Lambda(lambda x: x + torch.randn_like(x) * random.uniform(0.0, 0.02) if random.random() < 0.3 else x),
            T.Lambda(lambda x: x.clamp(0.0, 1.0)),
            T.Normalize(mean=clip_mean, std=clip_std),
        ])

    if train:
        return T.Compose([
            T.Resize((224, 224)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1,
                hue=0.02
            ),
            T.ToTensor(),
            T.Normalize(mean=clip_mean, std=clip_std),
        ])

    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=clip_mean, std=clip_std),
    ])