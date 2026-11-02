import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType


class ExplanationGenerator(nn.Module):
    """LLM-based explanation generation with heatmap-guided visual tokens."""

    def __init__(self, llm_name: str, embed_dim: int = 768):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(llm_name)
        self.tokenizer.padding_side = "left"

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        if "gpt2" in llm_name.lower():
            target_modules = ["c_attn", "c_proj"]
        else:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )

        self.llm = get_peft_model(self.llm, lora_config)
        self.llm.enable_input_require_grads()

        self.visual_proj = nn.Linear(embed_dim, self.llm.config.hidden_size)

        # Start the visual tokens at the scale of the LLM's word embeddings. The default
        # init makes them ~30x larger, which drowns out the text prompt (loss ~10 vs ~3).
        # Detector tokens are LayerNorm-ed, so their norm is about sqrt(embed_dim).
        with torch.no_grad():
            text_norm = self.llm.get_input_embeddings().weight.float().norm(dim=-1).mean().item()
            std = text_norm / ((embed_dim * self.llm.config.hidden_size) ** 0.5)
            nn.init.normal_(self.visual_proj.weight, std=std)
            nn.init.zeros_(self.visual_proj.bias)

        # Freeze everything first
        for param in self.llm.parameters():
            param.requires_grad = False

        # Unfreeze only LoRA params
        for name, param in self.llm.named_parameters():
            if "lora_" in name:
                param.requires_grad = True

        # Train visual projection
        for param in self.visual_proj.parameters():
            param.requires_grad = True
            
        self.instruction = (
            "You are an expert in deepfake face analysis. "
            "Your explanation must be consistent with the detector prediction. "
            "If the detector prediction is REAL, explain why the face appears natural, "
            "consistent, and free of strong manipulation artifacts. "
            "If the detector prediction is FAKE, explain which one or two suspicious facial "
            "regions or visual inconsistencies suggest manipulation. "
            "Use cautious, evidence-based language. "
            "Focus on the highlighted facial region. "
            "Do not contradict the detector prediction. "
            "Do not describe the image as cartoon, CGI, animation, or computer-generated "
            "unless that is unmistakably visible. "
            "Keep the answer concise, in 1-2 sentences."
        )


    def _build_prompt(self, question: str, label: str) -> str:
        if label == "real":
            task_line = (
                "Explain why the face appears natural and why strong manipulation artifacts "
                "are not evident."
            )
        else:
            task_line = (
                "Explain which one or two suspicious facial regions suggest manipulation."
            )

        return (
            f"{self.instruction}\n"
            f"Question: {question}\n"
            f"Detector prediction: {label}\n"
            f"Task: {task_line}\n"
            f"Answer:"
        )


    def _format_training_target(self, label: str, answer: str) -> str:
        raw = " ".join(answer.strip().split())
        raw_lower = raw.lower()

        if label == "real":
            if any(k in raw_lower for k in ["natural", "consistent", "authentic", "real", "no strong"]):
                return raw if raw.endswith(".") else raw + "."
            return (
                "The image appears real because facial structure, skin texture, and lighting "
                "look consistent, and no strong manipulation artifacts are evident."
            )

        # fake label
        prefixes = [
            "the image looks fake because",
            "the image appears fake because",
            "the image looks fake.",
            "the image appears fake.",
            "fake because",
            "because",
        ]

        cleaned = raw
        cleaned_lower = raw_lower
        for p in prefixes:
            if cleaned_lower.startswith(p):
                cleaned = cleaned[len(p):].strip(" .,:;-")
                break

        if len(cleaned) == 0:
            cleaned = (
                "some facial regions show blending inconsistencies, asymmetry, "
                "or unnatural texture"
            )

        cleaned = cleaned.rstrip(".")
        cleaned = cleaned[0].lower() + cleaned[1:]
        return f"The image appears fake because {cleaned}."
    

    def apply_heatmap(self, tokens: torch.Tensor, heatmap: torch.Tensor) -> torch.Tensor:
        heatmap = F.interpolate(
            heatmap.unsqueeze(1), size=(14, 14), mode="bilinear", align_corners=False
        )

        heatmap = heatmap.flatten(2)
        heatmap = heatmap / (heatmap.max(dim=-1, keepdim=True)[0] + 1e-6)

        global_token = tokens[:, :1]
        patch_tokens = tokens[:, 1:197]
        pixel_tokens = tokens[:, 197:]

        patch_tokens = patch_tokens * heatmap.transpose(1, 2)
        tokens = torch.cat([global_token, patch_tokens, pixel_tokens], dim=1)
        return tokens

    def forward(
        self,
        tokens: torch.Tensor,
        heatmap: torch.Tensor,
        labels_text: List[str],
        questions_text: List[str],
        answers_text: List[str],
    ) -> Tuple[torch.Tensor, List[str]]:

        device = tokens.device
        B = tokens.size(0)

        visual_embed = self.visual_proj(self.apply_heatmap(tokens, heatmap)).to(self.llm.dtype)

        prompts = [
            self._build_prompt(q, label)
            for q, label in zip(questions_text, labels_text)
        ]

        target_texts = [
            self._format_training_target(label, ans)
            for label, ans in zip(labels_text, answers_text)
        ]

        prompt_ids = self.tokenizer(prompts, truncation=True, max_length=256)["input_ids"]
        target_ids = self.tokenizer(
            target_texts, add_special_tokens=False, truncation=True, max_length=127
        )["input_ids"]

        # [prompt | target | eos], right-padded: the target directly follows the
        # prompt, the model learns to stop, and padding is never in the loss
        seqs, label_seqs = [], []
        for p, t in zip(prompt_ids, target_ids):
            t = t + [self.tokenizer.eos_token_id]
            seqs.append(p + t)
            label_seqs.append([-100] * len(p) + t)

        max_len = max(len(s) for s in seqs)
        input_ids = torch.full((B, max_len), self.tokenizer.pad_token_id, dtype=torch.long)
        text_mask = torch.zeros((B, max_len), dtype=torch.long)
        text_labels = torch.full((B, max_len), -100, dtype=torch.long)
        for i, (s, l) in enumerate(zip(seqs, label_seqs)):
            input_ids[i, :len(s)] = torch.tensor(s)
            text_mask[i, :len(s)] = 1
            text_labels[i, :len(l)] = torch.tensor(l)

        input_ids = input_ids.to(device)
        text_mask = text_mask.to(device)
        text_labels = text_labels.to(device)

        text_embeds = self.llm.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([visual_embed, text_embeds], dim=1).to(self.llm.dtype)

        visual_mask = torch.ones(B, visual_embed.size(1), device=device, dtype=text_mask.dtype)
        attention_mask = torch.cat([visual_mask, text_mask], dim=1)

        visual_labels = torch.full((B, visual_embed.size(1)), -100, device=device, dtype=torch.long)
        labels = torch.cat([visual_labels, text_labels], dim=1)

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels
        )

        # generation is only needed at inference; running it every training step is wasted time
        generated_text = []
        if not self.training:
            generated_text = self.generate_explanations(
                tokens=tokens,
                heatmap=heatmap,
                labels_text=labels_text,
                questions_text=questions_text
            )

        return outputs.loss, generated_text

    def generate_explanations(
        self,
        tokens: torch.Tensor,
        heatmap: torch.Tensor,
        labels_text: List[str],
        questions_text: List[str]
    ) -> List[str]:

        device = tokens.device
        tokens = self.apply_heatmap(tokens, heatmap)

        embed_fn = self.llm.get_input_embeddings()
        embed_device = embed_fn.weight.device
        embed_dtype = embed_fn.weight.dtype

        visual_embed = self.visual_proj(tokens).to(device=embed_device, dtype=embed_dtype)

        # same prompt as training
        prompts = [
            self._build_prompt(q, label)
            for q, label in zip(questions_text, labels_text)
        ]

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        ).to(embed_device)

        prompt_embeds = embed_fn(inputs.input_ids)

        inputs_embeds = torch.cat([visual_embed, prompt_embeds], dim=1)

        visual_mask = torch.ones(
            tokens.size(0), visual_embed.size(1),
            device=embed_device,
            dtype=inputs.attention_mask.dtype
        )

        attention_mask = torch.cat([visual_mask, inputs.attention_mask], dim=1)

        outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=64,
            do_sample=False,
            num_beams=1,
            pad_token_id=self.tokenizer.eos_token_id
        )

        texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        texts = [t.split("Answer:")[-1].strip() for t in texts]
        return texts