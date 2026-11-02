import os
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from transformers import get_cosine_schedule_with_warmup
from omegaconf import OmegaConf


class Stage2Trainer:
    def __init__(self, config, model, dataset, val_dataset=None):
        self.config = config
        self.model = model
        self.dataset = dataset
        self.val_dataset = val_dataset

        self.accelerator = Accelerator(mixed_precision="no")

        # IMPORTANT: switch model to stage 2 before optimizer creation
        self.model.set_stage(2)

        self.loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.get("num_workers", 4),
            pin_memory=True,
        )

        self.val_loader = None
        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=config.get("num_workers", 4),
                pin_memory=True,
            )

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if len(trainable_params) == 0:
            raise ValueError("No trainable parameters found for Stage 2. Did set_stage(2) run correctly?")
     
        trainable_params_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Trainable params: {trainable_params_count/1e6:.2f}M")

        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.lr,
            weight_decay=config.weight_decay
        )

        num_training_steps = len(self.loader) * config.epochs
        num_warmup_steps = int(0.1 * num_training_steps)

        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

        items_to_prepare = [self.model, self.optimizer, self.loader, self.scheduler]
        if self.val_loader is not None:
            items_to_prepare.append(self.val_loader)

        prepared = self.accelerator.prepare(*items_to_prepare)

        if self.val_loader is not None:
            self.model, self.optimizer, self.loader, self.scheduler, self.val_loader = prepared
        else:
            self.model, self.optimizer, self.loader, self.scheduler = prepared

    def _set_stage2_modes(self):
        # frozen detector should stay deterministic
        self.model.eval()

        # explainer should still train
        model_ref = self.accelerator.unwrap_model(self.model)
        if hasattr(model_ref, "explainer") and model_ref.explainer is not None:
            model_ref.explainer.train()

    def train_epoch(self, epoch):
        max_grad_norm = self.config.get("max_grad_norm", 1.0)

        self._set_stage2_modes()
        total_loss = 0.0

        for batch_idx, batch in enumerate(self.loader):
            images = batch["image"]
            answers = batch["answer"]
            questions = batch["question"]

            explanation_inputs = {
                "questions": list(questions),
                "answers": list(answers)
            }

            self.optimizer.zero_grad()

            with self.accelerator.autocast():
                outputs = self.model(images, explanations=explanation_inputs)

                if "exp_loss" not in outputs:
                    raise ValueError("Stage 2 model output does not contain exp_loss")

                loss = outputs["exp_loss"]

            self.accelerator.backward(loss)

            self.accelerator.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                max_grad_norm
            )

            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()

            if batch_idx % 20 == 0 and self.accelerator.is_main_process:
                print(f"Epoch {epoch} | Step {batch_idx} | Exp Loss: {loss.item():.4f}")

        return total_loss / len(self.loader)

    def validate(self, epoch):
        if self.val_loader is None:
            return None

        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in self.val_loader:
                images = batch["image"]
                answers = batch["answer"]
                questions = batch["question"]

                explanation_inputs = {
                    "questions": list(questions),
                    "answers": list(answers)
                }

                outputs = self.model(images, explanations=explanation_inputs)

                if "exp_loss" not in outputs:
                    raise ValueError("Stage 2 model output does not contain exp_loss")

                loss = outputs["exp_loss"]
                total_loss += loss.item()

        avg_loss = total_loss / len(self.val_loader)

        if self.accelerator.is_main_process:
            print(f"Epoch {epoch} | Val Exp Loss: {avg_loss:.4f}")

        return avg_loss

    def train(self):
        best_loss = float("inf")
        best_epoch = -1
        epochs_no_improve = 0
        patience = self.config.get("early_stopping_patience", 3)

        os.makedirs("checkpoints", exist_ok=True)

        for epoch in range(self.config.epochs):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate(epoch)

            current_loss = val_loss if val_loss is not None else train_loss

            if self.accelerator.is_main_process:
                print(f"Epoch {epoch} | Avg Train Exp Loss: {train_loss:.4f}")

            if current_loss < best_loss:
                best_loss = current_loss
                best_epoch = epoch
                epochs_no_improve = 0

                if self.accelerator.is_main_process:
                    torch.save({
                        "model": self.accelerator.unwrap_model(self.model).state_dict(),
                        "epoch": epoch,
                        "exp_loss": best_loss,
                        "config": OmegaConf.to_container(self.config, resolve=True),
                    }, "checkpoints/stage2_best.pth")

                print(f"New best Stage 2 model at epoch {epoch} | Exp Loss: {best_loss:.4f}")

            else:
                epochs_no_improve += 1
                print(
                    f"No improvement for {epochs_no_improve} epoch(s). "
                    f"Best loss: {best_loss:.4f} at epoch {best_epoch}"
                )

            if self.accelerator.is_main_process:
                torch.save({
                    "model": self.accelerator.unwrap_model(self.model).state_dict(),
                    "epoch": epoch,
                    "exp_loss": current_loss,
                    "config": OmegaConf.to_container(self.config, resolve=True),
                }, "checkpoints/stage2_last.pth")

            if epochs_no_improve >= patience:
                print(
                    f"Early stopping triggered at epoch {epoch}. "
                    f"Best loss: {best_loss:.4f} at epoch {best_epoch}"
                )
                break

        print("Stage 2 Training Complete")