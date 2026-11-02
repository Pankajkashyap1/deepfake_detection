import os
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import random
import numpy as np

from datasets.ffpp_dataset import FFPPDataset
from datasets.celebdf_dataset import CelebDFDataset
from datasets.combined_dataset import CombinedDataset
from datasets.deepfakeface_dataset import DeepFakeFaceDataset
from datasets.dd_vqa_dataset import DDVQADataset
from datasets.transforms import get_transforms
from models.deepfake_detector import DeepfakeVLMDetector
from training.stage1_trainer import Stage1Trainer
from training.stage2_trainer import Stage2Trainer


def set_seed(seed: int = 42):
    """Ensure reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@hydra.main(version_base=None, config_path="configs", config_name="train_stage1")
def main(cfg: DictConfig):

    print("\n===== CONFIG =====")
    print(OmegaConf.to_yaml(cfg))

    # seed
    seed = cfg.get("seed", 42)
    set_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # save config
    os.makedirs("configs_saved", exist_ok=True)
    OmegaConf.save(cfg, f"configs_saved/config_stage{cfg.stage}.yaml")

    # ============================================================
    # STAGE 1
    # ============================================================
    if cfg.stage == 1:
        print("\nLoading Stage 1 datasets...")
        
        if cfg.data.type == "ff++":

            train_ds = FFPPDataset(
                root=cfg.data.root,
                split="train",
                transform=get_transforms(train=True),
                train_ratio=cfg.data.get("train_ratio", 0.7),
                val_ratio=cfg.data.get("val_ratio", 0.15),
                test_ratio=cfg.data.get("test_ratio", 0.15),
                seed=cfg.get("seed", 42),
                max_frames_per_video=cfg.data.get("max_frames_per_video", None),
                balance=cfg.data.get("train_balance", True),
            )

            val_ds = FFPPDataset(
                root=cfg.data.root,
                split="val",
                transform=get_transforms(train=False),
                train_ratio=cfg.data.get("train_ratio", 0.7),
                val_ratio=cfg.data.get("val_ratio", 0.15),
                test_ratio=cfg.data.get("test_ratio", 0.15),
                seed=cfg.get("seed", 42),
                max_frames_per_video=cfg.data.get("max_frames_per_video", None),
                balance=cfg.data.get("val_balance", False),
            )

            test_ds = FFPPDataset(
                root=cfg.data.root,
                split="test",
                transform=get_transforms(train=False),
                train_ratio=cfg.data.get("train_ratio", 0.7),
                val_ratio=cfg.data.get("val_ratio", 0.15),
                test_ratio=cfg.data.get("test_ratio", 0.15),
                seed=cfg.get("seed", 42),
                max_frames_per_video=cfg.data.get("max_frames_per_video", None),
                balance=cfg.data.get("test_balance", False),
            )
        
        elif cfg.data.type == "celebdf":
            
            train_ds = CelebDFDataset(
                root=cfg.data.root,
                split="train",
                transform=get_transforms(train=True),
                train_ratio=cfg.data.get("train_ratio", 0.85),
                seed=cfg.get("seed", 42),
                max_frames_per_video=cfg.data.get("max_frames_per_video", None),
            )

            val_ds = CelebDFDataset(
                root=cfg.data.root,
                split="val",
                transform=get_transforms(train=False),
                train_ratio=cfg.data.get("train_ratio", 0.85),
                seed=cfg.get("seed", 42),
                max_frames_per_video=cfg.data.get("max_frames_per_video", None),
            )

            test_ds = CelebDFDataset(
                root=cfg.data.root,
                split="test",
                transform=get_transforms(train=False),
                train_ratio=cfg.data.get("train_ratio", 0.85),
                seed=cfg.get("seed", 42),
                max_frames_per_video=cfg.data.get("max_frames_per_video", None),
            )

        elif cfg.data.type == "multisource_ffpp_celebdf":
            
            ffpp_train = FFPPDataset(
                root=cfg.data.ffpp_root,
                split="train",
                transform=get_transforms(train=True, domain_generalization=True),
                train_ratio=cfg.data.get("ffpp_train_ratio", 0.7),
                val_ratio=cfg.data.get("ffpp_val_ratio", 0.15),
                test_ratio=cfg.data.get("ffpp_test_ratio", 0.15),
                seed=cfg.get("seed", 42),
                max_frames_per_video=cfg.data.get("max_frames_per_video", None),
                balance=False,
            )

            ffpp_val = FFPPDataset(
                root=cfg.data.ffpp_root,
                split="val",
                transform=get_transforms(train=False),
                train_ratio=cfg.data.get("ffpp_train_ratio", 0.7),
                val_ratio=cfg.data.get("ffpp_val_ratio", 0.15),
                test_ratio=cfg.data.get("ffpp_test_ratio", 0.15),
                seed=cfg.get("seed", 42),
                max_frames_per_video=cfg.data.get("max_frames_per_video", None),
                balance=False,
            )

            ffpp_test = FFPPDataset(
                root=cfg.data.ffpp_root,
                split="test",
                transform=get_transforms(train=False),
                train_ratio=cfg.data.get("ffpp_train_ratio", 0.7),
                val_ratio=cfg.data.get("ffpp_val_ratio", 0.15),
                test_ratio=cfg.data.get("ffpp_test_ratio", 0.15),
                seed=cfg.get("seed", 42),
                max_frames_per_video=cfg.data.get("max_frames_per_video", None),
                balance=False,
            )

            celeb_train = CelebDFDataset(
                root=cfg.data.celebdf_root,
                split="train",
                transform=get_transforms(train=True, domain_generalization=True),
                train_ratio=cfg.data.get("celebdf_train_ratio", 0.85),
                seed=cfg.get("seed", 42),
                max_frames_per_video=cfg.data.get("max_frames_per_video", None),
            )

            celeb_val = CelebDFDataset(
                root=cfg.data.celebdf_root,
                split="val",
                transform=get_transforms(train=False),
                train_ratio=cfg.data.get("celebdf_train_ratio", 0.85),
                seed=cfg.get("seed", 42),
                max_frames_per_video=cfg.data.get("max_frames_per_video", None),
            )

            celeb_test = CelebDFDataset(
                root=cfg.data.celebdf_root,
                split="test",
                transform=get_transforms(train=False),
                train_ratio=cfg.data.get("celebdf_train_ratio", 0.85),
                seed=cfg.get("seed", 42),
                max_frames_per_video=cfg.data.get("max_frames_per_video", None),
            )

            train_ds = CombinedDataset(
                {
                    "ffpp": ffpp_train,
                    "celebdf": celeb_train,
                },
                balance_sources=cfg.data.get("balance_sources", True),
                seed=cfg.get("seed", 42),
            )

            val_ds = CombinedDataset(
                {
                    "ffpp": ffpp_val,
                    "celebdf": celeb_val,
                },
                balance_sources=False,
                seed=cfg.get("seed", 42),
            )

            test_ds = CombinedDataset(
                {
                    "ffpp": ffpp_test,
                    "celebdf": celeb_test,
                },
                balance_sources=False,
                seed=cfg.get("seed", 42),
            )

        elif cfg.data.type == "multisource_ffpp_celebdf_dff":
            ffpp_train = FFPPDataset(
                root=cfg.data.ffpp_root,
                split="train",
                transform=get_transforms(train=True),
                train_ratio=cfg.data.get("ffpp_train_ratio", 0.7),
                val_ratio=cfg.data.get("ffpp_val_ratio", 0.15),
                test_ratio=cfg.data.get("ffpp_test_ratio", 0.15),
                seed=cfg.get("seed", 42),
                max_frames_per_video=cfg.data.get("max_frames_per_video", None),
                balance=False,
            )

            ffpp_val = FFPPDataset(
                root=cfg.data.ffpp_root,
                split="val",
                transform=get_transforms(train=False),
                train_ratio=cfg.data.get("ffpp_train_ratio", 0.7),
                val_ratio=cfg.data.get("ffpp_val_ratio", 0.15),
                test_ratio=cfg.data.get("ffpp_test_ratio", 0.15),
                seed=cfg.get("seed", 42),
                max_frames_per_video=cfg.data.get("max_frames_per_video", None),
                balance=False,
            )

            ffpp_test = FFPPDataset(
                root=cfg.data.ffpp_root,
                split="test",
                transform=get_transforms(train=False),
                train_ratio=cfg.data.get("ffpp_train_ratio", 0.7),
                val_ratio=cfg.data.get("ffpp_val_ratio", 0.15),
                test_ratio=cfg.data.get("ffpp_test_ratio", 0.15),
                seed=cfg.get("seed", 42),
                max_frames_per_video=cfg.data.get("max_frames_per_video", None),
                balance=False,
            )

            celeb_train = CelebDFDataset(
                root=cfg.data.celebdf_root,
                split="train",
                transform=get_transforms(train=True),
                train_ratio=cfg.data.get("celebdf_train_ratio", 0.85),
                seed=cfg.get("seed", 42),
                max_frames_per_video=cfg.data.get("max_frames_per_video", None),
            )

            celeb_val = CelebDFDataset(
                root=cfg.data.celebdf_root,
                split="val",
                transform=get_transforms(train=False),
                train_ratio=cfg.data.get("celebdf_train_ratio", 0.85),
                seed=cfg.get("seed", 42),
                max_frames_per_video=cfg.data.get("max_frames_per_video", None),
            )

            celeb_test = CelebDFDataset(
                root=cfg.data.celebdf_root,
                split="test",
                transform=get_transforms(train=False),
                train_ratio=cfg.data.get("celebdf_train_ratio", 0.85),
                seed=cfg.get("seed", 42),
                max_frames_per_video=cfg.data.get("max_frames_per_video", None),
            )

            dff_train = DeepFakeFaceDataset(
                root=cfg.data.dff_root,
                split="train",
                transform=get_transforms(train=True),
                real_dir=cfg.data.get("dff_real_dir", "wiki"),
                fake_dirs=cfg.data.get("dff_fake_dirs", None),
                train_ratio=cfg.data.get("dff_train_ratio", 0.8),
                val_ratio=cfg.data.get("dff_val_ratio", 0.1),
                seed=cfg.get("seed", 42),
                max_images_per_group=cfg.data.get("dff_max_images_per_group", None),
            )

            dff_val = DeepFakeFaceDataset(
                root=cfg.data.dff_root,
                split="val",
                transform=get_transforms(train=False),
                real_dir=cfg.data.get("dff_real_dir", "wiki"),
                fake_dirs=cfg.data.get("dff_fake_dirs", None),
                train_ratio=cfg.data.get("dff_train_ratio", 0.8),
                val_ratio=cfg.data.get("dff_val_ratio", 0.1),
                seed=cfg.get("seed", 42),
                max_images_per_group=cfg.data.get("dff_max_images_per_group", None),
            )

            dff_test = DeepFakeFaceDataset(
                root=cfg.data.dff_root,
                split="test",
                transform=get_transforms(train=False),
                real_dir=cfg.data.get("dff_real_dir", "wiki"),
                fake_dirs=cfg.data.get("dff_fake_dirs", None),
                train_ratio=cfg.data.get("dff_train_ratio", 0.8),
                val_ratio=cfg.data.get("dff_val_ratio", 0.1),
                seed=cfg.get("seed", 42),
                max_images_per_group=cfg.data.get("dff_max_images_per_group", None),
            )

            train_ds = CombinedDataset(
                {
                    "ffpp": ffpp_train,
                    "celebdf": celeb_train,
                    "dff": dff_train,
                },
                balance_sources=cfg.data.get("balance_sources", True),
                seed=cfg.get("seed", 42),
            )

            val_ds = CombinedDataset(
                {
                    "ffpp": ffpp_val,
                    "celebdf": celeb_val,
                    "dff": dff_val,
                },
                balance_sources=False,
                seed=cfg.get("seed", 42),
            )

            test_ds = CombinedDataset(
                {
                    "ffpp": ffpp_test,
                    "celebdf": celeb_test,
                    "dff": dff_test,
                },
                balance_sources=False,
                seed=cfg.get("seed", 42),
            )

        else:
            raise ValueError(f"Unsupported dataset type: {cfg.data.type}")

        print(f"Train samples: {len(train_ds)}")
        print(f"Val samples: {len(val_ds)}")
        print(f"Test samples: {len(test_ds)}")

        print("\nBuilding model...")
        model_cfg = OmegaConf.create({
            **OmegaConf.to_container(cfg.model, resolve=True),
            **OmegaConf.to_container(cfg.fusion, resolve=True),
        })

        model = DeepfakeVLMDetector(model_cfg)
        model = model.to(device)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"Total params: {total_params/1e6:.2f}M")
        print(f"Trainable params: {trainable_params/1e6:.2f}M")

        print("\n===== Stage 1: Detector Training =====")

        stage1_cfg = OmegaConf.create({
            **OmegaConf.to_container(cfg.training.stage1, resolve=True),
            **OmegaConf.to_container(cfg.losses, resolve=True),
            "use_wandb": cfg.get("use_wandb", False),
        })

        trainer = Stage1Trainer(
            stage1_cfg,
            model,
            train_ds,
            val_ds,
            test_ds
        )

        trainer.train()

        # ------------------------------------------------
        # Final held-out test evaluation using best checkpoint
        # ------------------------------------------------
        best_ckpt = torch.load("checkpoints/stage1_best.pth", map_location="cpu")
        trainer.accelerator.unwrap_model(trainer.model).load_state_dict(best_ckpt["model"], strict=False)

        test_loss, test_metrics = trainer.test()

        if trainer.accelerator.is_main_process:
            print("\n===== Final Test Results =====")
            print(f"Test Loss : {test_loss:.4f}")
            print(f"Test Acc  : {test_metrics['accuracy']:.4f}")
            print(f"Test AUC  : {test_metrics['auc']:.4f}")
            print(f"Test EER  : {test_metrics['eer']:.4f}")
            print(f"Test F1   : {test_metrics['f1']:.4f}")
            print(f"Test Prec : {test_metrics['precision']:.4f}")
            print(f"Test Rec  : {test_metrics['recall']:.4f}")

            torch.save({
                "test_loss": test_loss,
                "test_metrics": test_metrics,
            }, "checkpoints/stage1_test_results.pth")

    # ============================================================
    # STAGE 2
    # ============================================================
    elif cfg.stage == 2:
        print("\n===== Stage 2: Explanation Training =====")
        print("\nLoading Stage 2 dataset...")

        stage2_ds = DDVQADataset(
            annotation_path=cfg.data.annotation_path,
            image_root=cfg.data.image_root,
            transform=get_transforms(train=True),
            max_samples=cfg.data.get("max_samples", None),
            use_first_answer_only=cfg.data.get("use_first_answer_only", True)
        )

        print(f"Stage 2 samples: {len(stage2_ds)}")

        print("\nBuilding model...")
        model_cfg = OmegaConf.create({
            **OmegaConf.to_container(cfg.model, resolve=True),
            **OmegaConf.to_container(cfg.fusion, resolve=True),
            "use_llm": True,
        })

        model = DeepfakeVLMDetector(model_cfg)

        detector_ckpt_path = cfg.training.stage2.detector_ckpt
        detector_ckpt = torch.load(detector_ckpt_path, map_location="cpu")

        missing, unexpected = model.load_state_dict(detector_ckpt["model"], strict=False)
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)

        model.set_stage(2)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"Total params: {total_params/1e6:.2f}M")
        print(f"Trainable params: {trainable_params/1e6:.2f}M")

        trainer = Stage2Trainer(
            cfg.training.stage2,
            model,
            stage2_ds
        )

        trainer.train()

    else:
        raise ValueError(f"Invalid stage: {cfg.stage}")

    print("\n✅ Training completed!")



if __name__ == "__main__":
    main()