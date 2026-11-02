import os
import json
from PIL import Image
from torch.utils.data import Dataset


class DDVQADataset(Dataset):
    """
    DD-VQA dataset loader using FF_faces_by_type.

    sample_id format:
        manipulationid_src_tgt
    examples:
        0_166_167
        1_166_167
        5_166_167

    Mapping:
        0 -> Deepfakes
        1 -> Face2Face
        2 -> FaceShifter
        3 -> FaceSwap
        5 -> Original
        6 -> NeuralTextures
    """

    MANIP_TO_FOLDER = {
        0: "Deepfakes",
        1: "Face2Face",
        2: "FaceShifter",
        3: "FaceSwap",
        5: "Original",
        6: "NeuralTextures"
    }

    def __init__(
        self,
        annotation_path,
        image_root,
        transform=None,
        max_samples=None,
        use_first_answer_only=True,
    ):
        self.annotation_path = annotation_path
        self.image_root = image_root
        self.transform = transform
        self.max_samples = max_samples
        self.use_first_answer_only = use_first_answer_only

        self.raw_data = {}

        if os.path.isdir(annotation_path):
            json_files = []
            for root, _, files in os.walk(annotation_path):
                for fname in files:
                    if fname.endswith(".json"):
                        json_files.append(os.path.join(root, fname))

            json_files = sorted(json_files)

            print(f"Loading DD-VQA annotations from directory: {annotation_path}")
            print(f"Found JSON files: {len(json_files)}")

            for jf in json_files:
                with open(jf, "r") as f:
                    data = json.load(f)

                if not isinstance(data, dict):
                    continue

                self.raw_data.update(data)

        else:
            with open(annotation_path, "r") as f:
                self.raw_data = json.load(f)

        self.samples = []
        self._build_samples()

        if self.max_samples is not None:
            self.samples = self.samples[:self.max_samples]

        print(f"DDVQADataset loaded samples: {len(self.samples)}")

    def _parse_sample_id(self, sample_id):
        parts = sample_id.split("_")
        manipulation_id = int(parts[0])

        if manipulation_id == 5:
            if len(parts) == 2:
                src_id = parts[1]
                tgt_id = None
            elif len(parts) >= 3:
                src_id = parts[1]
                tgt_id = parts[2]
            else:
                raise ValueError(f"Unexpected original sample_id format: {sample_id}")

            label = 0
            return manipulation_id, src_id, tgt_id, label
        
        if len(parts) < 3:
            raise ValueError(f"Unexpected manipulated sample_id format: {sample_id}")

        src_id = parts[1]
        tgt_id = parts[2]
        label = 1
        return manipulation_id, src_id, tgt_id, label
        

    def _resolve_image_path(self, manipulation_id, src_id, tgt_id):
        folder = self.MANIP_TO_FOLDER[manipulation_id]

        if manipulation_id == 5:
            video_dir = os.path.join(self.image_root, folder, src_id)
        else:
            pair_id = f"{src_id}_{tgt_id}"
            video_dir = os.path.join(self.image_root, folder, pair_id)

        for fname in ("000.jpg", "000.png"):
            image_path = os.path.join(video_dir, fname)
            if os.path.exists(image_path):
                return image_path

        return None

    def _build_samples(self):
        missing_images = 0

        for sample_id, qa_dict in self.raw_data.items():
            manipulation_id, src_id, tgt_id, label = self._parse_sample_id(sample_id)
            image_path = self._resolve_image_path(manipulation_id, src_id, tgt_id)

            if image_path is None:
                missing_images += 1
                continue

            for question_id, qa_item in qa_dict.items():
                question = qa_item.get("question", "").strip()
                answers = qa_item.get("answer", [])

                if not question or not isinstance(answers, list) or len(answers) == 0:
                    continue

                if self.use_first_answer_only:
                    answers = [answers[0]]

                for ans in answers:
                    ans = ans.strip()
                    if not ans:
                        continue

                    self.samples.append({
                        "sample_id": sample_id,
                        "question_id": question_id,
                        "manipulation_id": manipulation_id,
                        "label": label,
                        "image_path": image_path,
                        "question": question,
                        "answer": ans
                    })

        print(f"DDVQA missing image matches: {missing_images}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        image = Image.open(item["image_path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "label": item["label"],
            "question": item["question"],
            "answer": item["answer"],
            "sample_id": item["sample_id"],
            "question_id": item["question_id"],
            "manipulation_id": item["manipulation_id"],
            "image_path": item["image_path"]
        }