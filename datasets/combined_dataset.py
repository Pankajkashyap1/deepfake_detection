import random
from torch.utils.data import Dataset


class CombinedDataset(Dataset):
    """
    Combine multiple datasets that each return:
    {
        "image": ...,
        "label": ...,
        "path": ...,
        "video": ...
    }

    Exposes:
    - self.samples -> list of (global_id, label)
      so existing trainer code for class weights still works.
    """

    def __init__(self, datasets_dict, balance_sources=False, seed=42):
        """
        datasets_dict: dict like {"ffpp": ffpp_dataset, "celebdf": celebdf_dataset}
        balance_sources: if True, downsample each source to the smallest source size
        """
        self.datasets_dict = datasets_dict
        self.source_names = list(datasets_dict.keys())
        self.seed = seed

        rng = random.Random(seed)

        source_indices = {}
        for source_name, ds in datasets_dict.items():
            source_indices[source_name] = list(range(len(ds)))

        if balance_sources:
            min_len = min(len(v) for v in source_indices.values())
            for source_name in self.source_names:
                rng.shuffle(source_indices[source_name])
                source_indices[source_name] = source_indices[source_name][:min_len]

        self.index_map = []
        self.samples = []

        for source_name in self.source_names:
            ds = datasets_dict[source_name]
            for local_idx in source_indices[source_name]:
                # We assume each underlying dataset has .samples with label at position 1
                label = ds.samples[local_idx][1]
                self.index_map.append((source_name, local_idx))
                self.samples.append((f"{source_name}:{local_idx}", label))

        rng.shuffle(self.index_map)
        rng.shuffle(self.samples)

        print("CombinedDataset summary:")
        for source_name in self.source_names:
            used = sum(1 for s, _ in self.index_map if s == source_name)
            print(f"  {source_name}: {used}")

        print(f"  total: {len(self.index_map)}")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        source_name, local_idx = self.index_map[idx]
        item = self.datasets_dict[source_name][local_idx]

        out = dict(item)
        out["source"] = source_name
        return out