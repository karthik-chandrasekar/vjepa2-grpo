"""OpenVLA-OFT policy wrapper.

OpenVLA-OFT (Kim et al. 2025, arXiv 2502.19645) is the base VLA policy.
Key properties:
  - Parallel decoding (predicts full 8-step action chunk in one forward pass)
  - Continuous action representation (L1 regression, not tokenized)
  - 7B Llama backbone with vision tokens
  - Action chunk size = 8 by default

For GRPO we need:
  - sample_action_chunk(obs, lang, n_samples=G): G stochastic samples
  - log_prob_action_chunks(obs, lang, actions): for the surrogate gradient
  - LoRA-only updates (frozen backbone)

IMPORTANT: OpenVLA-OFT in HF currently exposes deterministic action prediction.
For RL we add a small Gaussian noise head over the continuous actions; the
log-prob is then closed-form. This matches the SimpleVLA-RL recipe.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple
from pathlib import Path


class OpenVLAOFTPolicy(nn.Module):
    """Wrapper that exposes the GRPO-friendly API around an OpenVLA-OFT HF
    checkpoint.

    Note: the OpenVLA-OFT HF integration may need a custom loader. The
    moojink/openvla-7b-oft-* checkpoints ship a `processor_config.json` and
    custom code. Verify the exact load path on first run with:

        from transformers import AutoModelForVision2Seq, AutoProcessor
        m = AutoModelForVision2Seq.from_pretrained(
            "moojink/openvla-7b-oft-finetuned-libero-spatial",
            trust_remote_code=True, torch_dtype=torch.bfloat16,
        )
        p = AutoProcessor.from_pretrained(
            "moojink/openvla-7b-oft-finetuned-libero-spatial",
            trust_remote_code=True,
        )

    If the OFT-specific action head isn't directly exposed, you may need to
    fork the OpenVLA-OFT codebase (github.com/moojink/openvla-oft) instead of
    using HF AutoModel.
    """

    def __init__(
        self,
        model_id: str,
        action_dim: int = 7,
        action_chunk: int = 8,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[List[str]] = None,
        action_log_std_init: float = -1.0,
    ):
        super().__init__()
        from transformers import AutoModelForVision2Seq, AutoProcessor
        self.action_dim = action_dim
        self.action_chunk = action_chunk

        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_id, trust_remote_code=True, torch_dtype=dtype,
        ).to(device)

        # Freeze base
        for p in self.model.parameters():
            p.requires_grad = False

        # Apply LoRA
        self._apply_lora(lora_r, lora_alpha, lora_dropout, lora_target_modules)

        # Action stochasticity: state-independent log-std vector (per dim)
        # Small init so early policy ~ deterministic OFT predictions
        self.log_std = nn.Parameter(
            torch.full((action_dim,), action_log_std_init, dtype=torch.float32)
        )

    def _apply_lora(self, r, alpha, dropout, target_modules):
        from peft import LoraConfig, get_peft_model
        if target_modules is None:
            # Llama-style defaults; verify against actual module names!
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        cfg = LoraConfig(
            r=r, lora_alpha=alpha, lora_dropout=dropout,
            target_modules=target_modules, bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, cfg)
        self.model.print_trainable_parameters()

    # ----- inference API -----------------------------------------------------

    def _predict_mean_action(
        self,
        pixel_obs: torch.Tensor,
        instruction: str,
    ) -> torch.Tensor:
        """Returns the OFT deterministic action chunk: [action_chunk, action_dim]."""
        # The exact call signature here depends on OpenVLA-OFT's processor.
        # In the canonical openvla-oft codebase:
        #   inputs = processor(images=pixel_obs, text=prompt, return_tensors="pt")
        #   action = model.predict_action(**inputs, unnorm_key=...)
        # This is a placeholder; replace with the actual OFT call after verifying
        # the loaded model interface.
        prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
        inputs = self.processor(images=pixel_obs, text=prompt, return_tensors="pt")
        inputs = {k: v.to(next(self.model.parameters()).device) for k, v in inputs.items()}
        # predict_action is OFT-specific
        action = self.model.predict_action(**inputs, unnorm_key="bridge_orig")
        # action: [action_chunk, action_dim]  (or [action_dim] if not chunked)
        if action.ndim == 1:
            action = action.unsqueeze(0).repeat(self.action_chunk, 1)
        return action

    @torch.no_grad()
    def act(self, obs_pixels, instruction: str) -> torch.Tensor:
        """Deterministic action for eval. Returns [action_chunk, action_dim]."""
        return self._predict_mean_action(obs_pixels, instruction)

    def sample_action_chunk(
        self,
        obs_pixels,
        instruction: str,
        n_samples: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action chunks under Gaussian noise around the OFT mean.

        Returns:
            actions:  [n_samples, action_chunk, action_dim]
            log_prob: [n_samples] sum-log-prob of each chunk
        """
        mean = self._predict_mean_action(obs_pixels, instruction)  # [chunk, A]
        std = self.log_std.exp().to(mean.device).to(mean.dtype)
        eps = torch.randn(n_samples, *mean.shape, device=mean.device, dtype=mean.dtype)
        actions = mean.unsqueeze(0) + eps * std.view(1, 1, -1)
        # log p(action) = sum over chunk*A of -0.5 ((a-mu)/std)^2 - log(std) - 0.5 log(2pi)
        lp = -0.5 * ((actions - mean.unsqueeze(0)) / std.view(1, 1, -1)) ** 2
        lp = lp - self.log_std.view(1, 1, -1) - 0.5 * torch.log(torch.tensor(2 * 3.14159265))
        log_prob = lp.sum(dim=(1, 2))   # [n_samples]
        return actions, log_prob

    def log_prob_action_chunks(
        self,
        obs_pixels,
        instruction: str,
        actions: torch.Tensor,   # [G, chunk, A]
    ) -> torch.Tensor:
        """Compute log-prob of given actions under the current policy.

        Recomputes the OFT mean for grads to flow through LoRA params.
        """
        mean = self._predict_mean_action(obs_pixels, instruction)  # [chunk, A]
        std = self.log_std.exp().to(mean.device).to(mean.dtype)
        lp = -0.5 * ((actions - mean.unsqueeze(0)) / std.view(1, 1, -1)) ** 2
        lp = lp - self.log_std.view(1, 1, -1) - 0.5 * torch.log(torch.tensor(2 * 3.14159265))
        return lp.sum(dim=(1, 2))       # [G]

    # ----- ckpt API ----------------------------------------------------------

    def save_lora(self, path: str):
        Path(path).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        torch.save({"log_std": self.log_std.detach().cpu()},
                   Path(path) / "policy_extras.pt")

    def load_lora(self, path: str):
        from peft import PeftModel
        self.model = PeftModel.from_pretrained(self.model, path)
        extras = torch.load(Path(path) / "policy_extras.pt", map_location="cpu")
        with torch.no_grad():
            self.log_std.copy_(extras["log_std"].to(self.log_std.device))
