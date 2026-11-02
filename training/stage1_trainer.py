import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from accelerate import Accelerator
import wandb
import os

from omegaconf import OmegaConf
from transformers import get_cosine_schedule_with_warmup
from losses.deepfake_losses import DeepfakeLoss
from utils.metrics import compute_metrics


class Stage1Trainer:
    def __init__(self, config: dict, model: nn.Module, train_dataset, val_dataset, test_dataset=None):
        self.config = config
        self.accelerator = Accelerator(mixed_precision="fp16")

        self.model = model
        self.model.set_stage(1)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.get("num_workers", 4),
            pin_memory=True
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.get("num_workers", 4),
            pin_memory=True
        )

        self.test_loader = None
        if test_dataset is not None:
            self.test_loader = DataLoader(
                test_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=config.get("num_workers", 4),
                pin_memory=True
            )

        train_labels = [s[1] for s in train_dataset.samples]
        num_real = sum(1 for x in train_labels if x == 0)
        num_fake = sum(1 for x in train_labels if x == 1)
        total = num_real + num_fake

        w_real = total / (2.0 * num_real)
        w_fake = total / (2.0 * num_fake)
        class_weights = torch.tensor([w_real, w_fake], dtype=torch.float32)

        loss_cfg = {
            "lambda_cls": config.get("lambda_cls", 1.0),
            "lambda_contrast": config.get("lambda_contrast", 0.1),
            "lambda_proto": config.get("lambda_proto", 0.01),
            "temperature": config.get("temperature", 0.07),
            "label_smoothing": config.get("label_smoothing", 0.0),
        }

        self.loss_fn = DeepfakeLoss(loss_cfg, class_weights=class_weights)

        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=config.lr,
            weight_decay=config.weight_decay
        )

        accum_steps = config.get("grad_accum_steps", 1)
        num_update_steps_per_epoch = (len(self.train_loader) + accum_steps - 1) // accum_steps
        num_training_steps = num_update_steps_per_epoch * config.epochs
        num_warmup_steps = int(0.1 * num_training_steps)

        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

        prepare_items = [
            self.model,
            self.optimizer,
            self.scheduler,
            self.train_loader,
            self.val_loader,
        ]
        if self.test_loader is not None:
            prepare_items.append(self.test_loader)

        prepared = self.accelerator.prepare(*prepare_items)

        if self.test_loader is not None:
            self.model, self.optimizer, self.scheduler, self.train_loader, self.val_loader, self.test_loader = prepared
        else:
            self.model, self.optimizer, self.scheduler, self.train_loader, self.val_loader = prepared

        self.loss_fn = self.loss_fn.to(self.accelerator.device)

        self.use_wandb = config.get("use_wandb", False)
        if self.use_wandb and self.accelerator.is_main_process:
            wandb.init(
                project="deepfake-vlm-stage1",
                config=OmegaConf.to_container(config, resolve=True)
            )

    def train_epoch(self, epoch: int):
        accum_steps = self.config.get("grad_accum_steps", 1)
        max_grad_norm = self.config.get("max_grad_norm", 1.0)

        self.model.train()
        total_loss = 0.0
        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(self.train_loader):
            images = batch["image"]
            labels = batch["label"]

            with self.accelerator.autocast():
                outputs = self.model(images)
                losses = self.loss_fn(outputs, labels, stage=1)
                total_batch_loss = losses["total"]
                loss = total_batch_loss / accum_steps

            self.accelerator.backward(loss)

            if (batch_idx + 1) % accum_steps == 0:
                self.accelerator.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, self.model.parameters()),
                    max_grad_norm
                )
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step()

            total_loss += total_batch_loss.item()

            if batch_idx % 50 == 0 and self.accelerator.is_main_process:
                print(
                    f"Epoch {epoch} | Step {batch_idx} | "
                    f"Loss: {total_batch_loss.item():.4f} | "
                    f"LR: {self.scheduler.get_last_lr()[0]:.6e}"
                )

                if self.use_wandb:
                    wandb.log({
                        "epoch": epoch,
                        "train/loss": total_batch_loss.item(),
                        "train/lr": self.scheduler.get_last_lr()[0]
                    })

        if len(self.train_loader) % accum_steps != 0:
            self.accelerator.clip_grad_norm_(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                max_grad_norm
            )
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()

        return total_loss / len(self.train_loader)

    def evaluate(self, loader, epoch: int, split_name: str = "val"):
        self.model.eval()
        total_loss = 0.0

        all_logits = []
        all_labels = []
        all_probs = []

        threshold = self.config.get("eval_threshold", 0.5)

        with torch.no_grad():
            for batch in loader:
                images = batch["image"]
                labels = batch["label"]

                outputs = self.model(images)
                losses = self.loss_fn(outputs, labels, stage=1)

                total_loss += losses["total"].item()

                logits = outputs["logits"]
                probs = torch.softmax(logits, dim=1)[:, 1]

                gathered_logits = self.accelerator.gather_for_metrics(logits)
                gathered_labels = self.accelerator.gather_for_metrics(labels)
                gathered_probs = self.accelerator.gather_for_metrics(probs)

                all_logits.append(gathered_logits.detach().cpu())
                all_labels.append(gathered_labels.detach().cpu())
                all_probs.append(gathered_probs.detach().cpu())

        logits = torch.cat(all_logits, dim=0)
        labels = torch.cat(all_labels, dim=0)
        probs = torch.cat(all_probs, dim=0)

        metrics = compute_metrics(logits, labels, threshold=threshold)
        avg_loss = total_loss / len(loader)

        if self.accelerator.is_main_process:
            print(f"\n{split_name.upper()} Results (Epoch {epoch}):")
            print(f"Loss     : {avg_loss:.4f}")
            print(f"Accuracy : {metrics['accuracy']:.4f}")
            print(f"AUC      : {metrics['auc']:.4f}")
            print(f"EER      : {metrics['eer']:.4f}")
            print(f"F1       : {metrics['f1']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall   : {metrics['recall']:.4f}")

        if self.use_wandb and self.accelerator.is_main_process and split_name == "val":
            wandb.log({
                "epoch": epoch,
                "val/loss": avg_loss,
                "val/acc": metrics["accuracy"],
                "val/auc": metrics["auc"],
                "val/eer": metrics["eer"],
                "val/f1": metrics["f1"],
                "val/precision": metrics["precision"],
                "val/recall": metrics["recall"],
            })

        metrics["labels_list"] = labels.numpy().tolist()
        metrics["probs_list"] = probs.numpy().tolist()

        return avg_loss, metrics

    def validate(self, epoch: int):
        return self.evaluate(self.val_loader, epoch=epoch, split_name="val")

    def test(self):
        if self.test_loader is None:
            raise ValueError("test_loader is not available.")
        return self.evaluate(self.test_loader, epoch=-1, split_name="test")

    def train(self):
        best_auc = 0.0
        best_epoch = -1
        epochs_no_improve = 0
        patience = self.config.get("early_stopping_patience", 5)

        os.makedirs("checkpoints", exist_ok=True)

        for epoch in range(self.config.epochs):
            train_loss = self.train_epoch(epoch)
            val_loss, metrics = self.validate(epoch)

            current_auc = metrics["auc"]

            if current_auc > best_auc:
                best_auc = current_auc
                best_epoch = epoch
                epochs_no_improve = 0

                if self.accelerator.is_main_process:
                    torch.save({
                        "model": self.accelerator.unwrap_model(self.model).state_dict(),
                        "epoch": epoch,
                        "auc": best_auc,
                        "config": OmegaConf.to_container(self.config, resolve=True),
                        "val_labels": metrics["labels_list"],
                        "val_probs": metrics["probs_list"],
                    }, "checkpoints/stage1_best.pth")

                print(f"New best model at epoch {epoch} | AUC: {best_auc:.4f}")
            else:
                epochs_no_improve += 1
                print(
                    f"No AUC improvement for {epochs_no_improve} epoch(s). "
                    f"Best AUC: {best_auc:.4f} at epoch {best_epoch}"
                )

            if self.accelerator.is_main_process:
                torch.save({
                    "model": self.accelerator.unwrap_model(self.model).state_dict(),
                    "epoch": epoch,
                    "auc": current_auc,
                    "config": OmegaConf.to_container(self.config, resolve=True),
                }, "checkpoints/stage1_last.pth")

            if epochs_no_improve >= patience:
                print(
                    f"Early stopping at epoch {epoch}. "
                    f"Best AUC: {best_auc:.4f} at epoch {best_epoch}"
                )
                break

        print("Stage 1 Training Complete")