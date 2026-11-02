from datasets.ffpp_dataset import FFPPDataset
from datasets.transforms import get_transforms

root = "/data/datasets/deepfake_datasets/FF_fixed"   # change if needed

train_ds = FFPPDataset(
    root=root,
    split="train",
    transform=get_transforms(train=True),
    split_ratio=0.8,
    seed=42,
    max_frames_per_video=None,
    balance=False,
)

val_ds = FFPPDataset(
    root=root,
    split="val",
    transform=get_transforms(train=False),
    split_ratio=0.8,
    seed=42,
    max_frames_per_video=None,
    balance=False,
)

print("Train size:", len(train_ds))
print("Val size:", len(val_ds))