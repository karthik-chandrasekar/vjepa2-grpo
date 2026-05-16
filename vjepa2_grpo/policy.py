"""OpenVLA-OFT policy wrapper for GRPO post-training.

IMPORTANT: this module runs in **venv-oft** (Python 3.11, transformers 4.40.1
from moojink's fork, torch 2.2). It is NOT importable from venv1 — V-JEPA-2
loading would crash on 4.40.1. The GRPO process imports this; the encoder
process does not.

Architecture (per openvla-oft/experiments/robot/openvla_utils.py:get_vla):
  1. Auto-class registration for the custom OpenVLA classes (one-time).
  2. Load the VLA backbone (DinoSigLIP vision + Llama-7B LLM) via HF AutoModel.
  3. Load the L1RegressionActionHead from a separate checkpoint shipped in the
     same HF snapshot (action_head--150000_checkpoint.pt for libero-spatial).
  4. Load the ProprioProjector from another snapshot file.
  5. Call vla.vision_backbone.set_num_images_in_input(2) for dual cam.
  6. Wrap the Llama LM with LoRA on attention projections. Vision backbone,
     projector, action head, and proprio projector remain frozen.

GRPO needs three operating modes, with different gradient behavior:

  act(obs, instr)
    Deterministic OFT mean, no gradients. For eval rollouts.

  sample(obs, instr, n_samples)
    No gradients. Computes the OFT mean once, draws G Gaussian-noise samples
    around it, returns (actions, log_prob) where log_prob is the snapshot
    log-prob under the policy *as of now*. Used to populate the replay batch.

  recompute_log_prob(obs, instr, actions)
    WITH gradients. Recomputes the OFT mean for the same (obs, instr), then
    evaluates Gaussian log_prob of the given actions. Gradient flows through
    the Llama LM (LoRA targets) and the log_std parameter. This is what
    makes the GRPO surrogate update actually train LoRA.

Subtlety on gradient flow:
  The reference `get_vla_action` always wraps `predict_action` in
  `with torch.inference_mode():`. We CANNOT do that for `recompute_log_prob`
  or no gradient flows. The OFT model's `predict_action` itself does not
  appear to internally force inference_mode (checked against modeling_prismatic).
  If a future model rev does, the fallback is to bypass `predict_action`
  and call the LM forward + action_head manually — see `_compute_action_mean`.
"""
from __future__ import annotations
import glob
import json
import math
import sys
from pathlib import Path
from typing import Optional, List, Tuple, Union

import numpy as np
from PIL import Image
import torch
import torch.nn as nn

# venv-oft imports — these fail loudly in venv1, which is correct.
sys.path.insert(0, "/workspace/repos/openvla-oft")

from transformers import AutoConfig, AutoModelForVision2Seq, AutoProcessor, AutoImageProcessor
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.action_heads import L1RegressionActionHead
from prismatic.models.projectors import ProprioProjector


# Register once at import time. Idempotent if already registered.
def _ensure_oft_autoclass_registered():
    try:
        AutoConfig.register("openvla", OpenVLAConfig)
    except ValueError:
        pass  # already registered
    try:
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    except ValueError:
        pass
    try:
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    except ValueError:
        pass
    try:
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
    except ValueError:
        pass


_ensure_oft_autoclass_registered()


def _resolve_snapshot(model_id: str) -> str:
    """Resolve an HF model id to its locally cached snapshot path under
    /workspace/.hf_cache (avoids re-download)."""
    safe = model_id.replace("/", "--")
    pat = f"/workspace/.hf_cache/models--{safe}/snapshots/*/"
    snaps = glob.glob(pat)
    if not snaps:
        raise FileNotFoundError(
            f"No locally cached snapshot for {model_id} at {pat}. "
            f"Pre-download with: huggingface-cli download {model_id}"
        )
    if len(snaps) > 1:
        # newest one — refs/main symlink resolves there
        snaps.sort(key=lambda p: Path(p).stat().st_mtime, reverse=True)
    return snaps[0]


def _find_component_ckpt(snap_dir: str, pattern: str) -> str:
    """Locate the proprio_projector / action_head .pt file in the snapshot.

    The reference uses hf_hub_download against the HF Hub model id; that
    redirects to the same cached file. We look directly to avoid the network
    call (and the requirement that the user have HF auth set up correctly)."""
    matches = [p for p in Path(snap_dir).iterdir() if pattern in p.name and p.suffix == ".pt"]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly 1 {pattern}*.pt file in {snap_dir}, "
            f"found {len(matches)}: {[m.name for m in matches]}"
        )
    return str(matches[0])


def _load_component_state_dict(path: str) -> dict:
    """Strip 'module.' DDP prefix from a saved component state dict."""
    sd = torch.load(path, weights_only=True, map_location="cpu")
    return {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

class OpenVLAOFTPolicy(nn.Module):
    """OpenVLA-OFT-based stochastic policy for GRPO post-training.

    Public API consumed by GRPO (vjepa2_grpo/grpo.py):
        act(obs, instr) -> torch.Tensor [chunk, action_dim]    eval
        sample(obs, instr, n_samples) -> (actions, log_prob)   on-policy rollout
        recompute_log_prob(obs, instr, actions) -> log_prob    surrogate update
        save_lora(path) / load_lora(path)                      checkpointing

    obs is a dict matching the LIBERO observation keys:
        {
          "agentview_image":         np.ndarray [H,W,3] uint8  (primary cam)
          "robot0_eye_in_hand_image": np.ndarray [H,W,3] uint8 (wrist cam)
          "state":                   np.ndarray [8]   float    (8-dim proprio)
        }
    instr is a plain string ("turn on the stove ...").
    """

    # libero-spatial OFT uses libero_spatial_no_noops as the unnorm key
    DEFAULT_UNNORM_KEY = "libero_spatial_no_noops"

    def __init__(
        self,
        model_id: str = "moojink/openvla-7b-oft-finetuned-libero-spatial",
        action_dim: int = 7,
        action_chunk: int = 8,
        proprio_dim: int = 8,
        num_images_in_input: int = 2,
        unnorm_key: Optional[str] = None,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda:0",
        # LoRA scope (Llama backbone only)
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[List[str]] = None,
        # Gaussian noise around the OFT mean
        action_log_std_init: float = -1.0,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.action_chunk = action_chunk
        self.proprio_dim = proprio_dim
        self.num_images_in_input = num_images_in_input
        self.unnorm_key = unnorm_key or self.DEFAULT_UNNORM_KEY
        self.dtype = dtype
        self.device = device

        snap = _resolve_snapshot(model_id)
        print(f"[policy] loading OFT from local snapshot: {snap}")

        # 1. Backbone + processor
        self.processor = AutoProcessor.from_pretrained(snap, trust_remote_code=True)
        self.vla: OpenVLAForActionPrediction = AutoModelForVision2Seq.from_pretrained(
            snap, trust_remote_code=True,
            torch_dtype=dtype, low_cpu_mem_usage=True,
        )

        # Configure dual-image input (must happen before .to(device))
        self.vla.vision_backbone.set_num_images_in_input(num_images_in_input)

        # 2. Freeze the entire backbone before applying LoRA
        for p in self.vla.parameters():
            p.requires_grad = False

        # 3. LoRA on Llama attention projections
        self._apply_lora(lora_r, lora_alpha, lora_dropout, lora_target_modules)

        # 4. Action head (L1 regression) — frozen, but in the compute graph
        llm_dim = self.vla.llm_dim
        self.action_head = L1RegressionActionHead(
            input_dim=llm_dim, hidden_dim=llm_dim, action_dim=action_dim,
        ).to(dtype).to(device).eval()
        ah_ckpt = _find_component_ckpt(snap, "action_head")
        self.action_head.load_state_dict(_load_component_state_dict(ah_ckpt))
        for p in self.action_head.parameters():
            p.requires_grad = False
        print(f"[policy] loaded action_head from {Path(ah_ckpt).name}")

        # 5. Proprio projector — frozen
        self.proprio_projector = ProprioProjector(
            llm_dim=llm_dim, proprio_dim=proprio_dim,
        ).to(dtype).to(device).eval()
        pp_ckpt = _find_component_ckpt(snap, "proprio_projector")
        self.proprio_projector.load_state_dict(_load_component_state_dict(pp_ckpt))
        for p in self.proprio_projector.parameters():
            p.requires_grad = False
        print(f"[policy] loaded proprio_projector from {Path(pp_ckpt).name}")

        # 6. Move VLA to device (LoRA params follow)
        self.vla = self.vla.to(device)

        # 7. Norm stats: dataset_statistics.json is the source of truth for the
        #    libero fine-tune. Reference's `_load_dataset_stats` puts it on
        #    `vla.norm_stats`, but only with the fine-tune's keys.
        # Load the libero fine-tune stats and merge into the base-VLA stats
        # ALREADY loaded by from_pretrained. After PEFT wrapping, the model
        # we need to modify is the inner base model (PEFT exposes it via
        # base_model.model), not the wrapper.
        stats_path = Path(snap) / "dataset_statistics.json"
        inner = self.vla.base_model.model if hasattr(self.vla, "base_model") else self.vla
        # Ensure norm_stats exists on inner (it should — loaded from checkpoint).
        if not hasattr(inner, "norm_stats") or inner.norm_stats is None:
            inner.norm_stats = {}
        if stats_path.exists():
            with open(stats_path) as f:
                libero_stats = json.load(f)
            # Merge: keep base-VLA's keys (the 25 datasets it was trained on),
            # add/overwrite with the libero fine-tune keys. predict_action reads
            # from inner.norm_stats.
            inner.norm_stats.update(libero_stats)
        # Mirror to the wrapper so external callers see it consistently
        self.vla.norm_stats = inner.norm_stats
        if self.unnorm_key not in inner.norm_stats:
            raise RuntimeError(
                f"unnorm_key={self.unnorm_key!r} not in norm_stats. "
                f"Available keys: {list(inner.norm_stats.keys())}"
            )
        print(f"[policy] norm_stats key: {self.unnorm_key} "
              f"(total keys: {len(inner.norm_stats)})")

        # 8. Learned per-dim log-std for the Gaussian noise wrapper
        self.log_std = nn.Parameter(
            torch.full((action_dim,), action_log_std_init,
                       dtype=torch.float32, device=device)
        )

        self._report_param_counts()

    def _apply_lora(self, r, alpha, dropout, target_modules):
        from peft import LoraConfig, get_peft_model
        if target_modules is None:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        cfg = LoraConfig(
            r=r, lora_alpha=alpha, lora_dropout=dropout,
            target_modules=target_modules, bias="none",
            task_type="CAUSAL_LM",
        )
        self.vla = get_peft_model(self.vla, cfg)
        # PEFT injects LoRA on top — print to log
        self.vla.print_trainable_parameters()

    def _report_param_counts(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[policy] total params: {total/1e9:.2f}B   "
              f"trainable: {trainable/1e6:.2f}M ({100*trainable/total:.3f}%)")

    # ----- observation prep --------------------------------------------------

    def _build_vla_inputs(self, obs: dict, instr: str) -> dict:
        """Replicates the reference's `get_vla_action` input prep, returning
        the dict that `predict_action(**inputs, ...)` consumes.

        - Concatenates primary image + wrist image along the channel axis
          (the path that `set_num_images_in_input(2)` enables).
        - Builds the OFT prompt string.
        - Normalizes proprio in-place using the loaded norm_stats.
        """
        primary = self._asarray_uint8_3d(obs["agentview_image"])
        primary_pil = Image.fromarray(primary)
        prompt = f"In: What action should the robot take to {instr.lower()}?\nOut:"
        inputs = self.processor(prompt, primary_pil).to(self.device, dtype=self.dtype)

        if self.num_images_in_input > 1:
            # wrist image stored under robot0_eye_in_hand_image in raw LIBERO
            if "robot0_eye_in_hand_image" in obs:
                wrist = obs["robot0_eye_in_hand_image"]
            elif "eye_in_hand_rgb" in obs:
                wrist = obs["eye_in_hand_rgb"]
            else:
                wrist = None
            if wrist is None:
                raise KeyError(
                    "num_images_in_input>1 but no wrist image found in obs "
                    "(looked for 'robot0_eye_in_hand_image', 'eye_in_hand_rgb')"
                )
            wrist = self._asarray_uint8_3d(wrist)
            wrist_pil = Image.fromarray(wrist)
            wrist_inputs = self.processor(prompt, wrist_pil).to(self.device, dtype=self.dtype)
            inputs["pixel_values"] = torch.cat(
                [inputs["pixel_values"], wrist_inputs["pixel_values"]], dim=1,
            )

        # 8-dim proprio, normalized
        proprio_raw = np.asarray(obs["state"], dtype=np.float32)
        if proprio_raw.shape != (self.proprio_dim,):
            raise ValueError(
                f"expected proprio shape ({self.proprio_dim},), got {proprio_raw.shape}"
            )
        norm_p = self._normalize_proprio(proprio_raw)
        inputs["proprio"] = norm_p

        return inputs

    def _normalize_proprio(self, proprio: np.ndarray) -> np.ndarray:
        """Mirrors openvla_utils.normalize_proprio for BOUNDS_Q99 stats."""
        stats = self.vla.norm_stats[self.unnorm_key]["proprio"]
        # libero-spatial uses BOUNDS_Q99: clip to [q01, q99], then scale to [-1,1]
        # Stats keys: q01, q99, min, max, mean, std (we use q01/q99 per OFT default).
        q01 = np.asarray(stats["q01"], dtype=np.float32)
        q99 = np.asarray(stats["q99"], dtype=np.float32)
        # Guard against zero spread (defensive — shouldn't happen for libero)
        span = np.maximum(q99 - q01, 1e-8)
        p = np.clip(proprio, q01, q99)
        p = 2.0 * (p - q01) / span - 1.0
        return p.astype(np.float32)

    @staticmethod
    def _asarray_uint8_3d(x) -> np.ndarray:
        if hasattr(x, "detach"):
            x = x.detach().cpu().numpy()
        x = np.asarray(x)
        if x.dtype != np.uint8:
            x = x.clip(0, 255).astype(np.uint8) if x.max() > 1.0001 \
                else (x * 255).clip(0, 255).astype(np.uint8)
        assert x.ndim == 3 and x.shape[-1] == 3, f"bad image shape {x.shape}"
        return x

    # ----- core: action mean -------------------------------------------------

    def _compute_action_mean_grad_safe(self, obs: dict, instr: str) -> torch.Tensor:
        """Manual forward replicating predict_action -> _regression_or_discrete_prediction
        -> _unnormalize_actions, but keeping the entire path in torch with gradient
        flow preserved. The upstream OFT implementation does .detach().numpy() at
        the action-head exit (modeling_prismatic.py:_regression_or_discrete_prediction),
        which severs grad to LoRA. This method skips that detach and replaces the
        numpy unnormalize with a torch.where so backward() reaches LoRA params.

        Returns: [chunk, action_dim] float32 tensor with grad, on device.
        """
        IGNORE_INDEX = -100  # HF convention; used by _process_action_masks

        inputs = self._build_vla_inputs(obs, instr)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        pixel_values = inputs["pixel_values"]
        proprio = inputs["proprio"]

        # PEFT-wrapped model: predict_action lives on the inner, and its private
        # helpers (_prepare_input_for_action_prediction etc) are inner methods.
        inner = self.vla.base_model.model if hasattr(self.vla, "base_model") else self.vla

        # 1. Match training-time empty-token insertion (29871 = empty token after ':')
        if not torch.all(input_ids[:, -1] == 29871):
            empty_tok = torch.tensor([[29871]], device=input_ids.device, dtype=input_ids.dtype)
            input_ids = torch.cat((input_ids, empty_tok), dim=1)

        # 2. Fake labels (used to build the action token mask)
        labels = input_ids.clone()
        labels[:] = IGNORE_INDEX

        NUM_PROMPT_TOKENS = input_ids.shape[-1] - 1

        # 3. Add action-token slots (these are the positions whose hidden states
        # we read out for the action head)
        input_ids, attention_mask = inner._prepare_input_for_action_prediction(input_ids, attention_mask)
        labels = inner._prepare_labels_for_action_prediction(labels, input_ids)

        # 4. Embeddings + action mask
        input_embeddings = inner.get_input_embeddings()(input_ids)
        all_actions_mask = inner._process_action_masks(labels)

        # 5. Language embeddings extracted before zeroing (mirrors predict_action)
        language_embeddings = input_embeddings[~all_actions_mask].reshape(
            input_embeddings.shape[0], -1, input_embeddings.shape[2]
        )

        # 6. Vision features (no FiLM for libero-spatial)
        projected_patch_embeddings = inner._process_vision_features(
            pixel_values, language_embeddings, use_film=False
        )

        # 7. Fuse proprio into the patch embedding sequence
        proprio_t = torch.as_tensor(
            proprio,
            device=projected_patch_embeddings.device,
            dtype=projected_patch_embeddings.dtype,
        )
        if proprio_t.dim() == 1:
            proprio_t = proprio_t.unsqueeze(0)  # [B=1, proprio_dim]
        projected_patch_embeddings = inner._process_proprio_features(
            projected_patch_embeddings, proprio_t, self.proprio_projector
        )

        NUM_PATCHES = inner.vision_backbone.get_num_patches() * inner.vision_backbone.get_num_images_in_input()
        NUM_PATCHES += 1  # +1 for proprio token (we always use proprio)

        # 8. Zero out action-token positions in the embedding sequence
        all_actions_mask_3d = all_actions_mask.unsqueeze(-1)
        input_embeddings = input_embeddings * ~all_actions_mask_3d

        # 9. Build the multimodal sequence (vision patches + language + action slots)
        multimodal_embeddings, multimodal_attention_mask = inner._build_multimodal_attention(
            input_embeddings, projected_patch_embeddings, attention_mask
        )

        # 10. LLM forward — grad flows through LoRA-injected attention projections
        lm_out = inner.language_model(
            input_ids=None,
            attention_mask=multimodal_attention_mask,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=multimodal_embeddings,
            labels=None,
            use_cache=None,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )

        # 11. Extract action-token hidden states
        last_hidden_states = lm_out.hidden_states[-1]
        action_hidden = last_hidden_states[
            :,
            NUM_PATCHES + NUM_PROMPT_TOKENS : NUM_PATCHES + NUM_PROMPT_TOKENS + self.action_dim * self.action_chunk,
            :,
        ]  # [B=1, chunk*action_dim, D]

        # 12. Action head — KEEP graph alive (do NOT detach/numpy)
        normalized_actions = self.action_head.predict_action(action_hidden)
        normalized_actions = normalized_actions.reshape(self.action_chunk, self.action_dim)

        # 13. Torch unnormalize (replaces the numpy version in _unnormalize_actions)
        stats = inner.norm_stats[self.unnorm_key]["action"]
        q01 = torch.as_tensor(stats["q01"], device=normalized_actions.device, dtype=normalized_actions.dtype)
        q99 = torch.as_tensor(stats["q99"], device=normalized_actions.device, dtype=normalized_actions.dtype)
        mask_list = stats.get("mask", [True] * self.action_dim)
        mask = torch.as_tensor(mask_list, device=normalized_actions.device, dtype=torch.bool)
        actions = torch.where(
            mask,
            0.5 * (normalized_actions + 1.0) * (q99 - q01 + 1e-8) + q01,
            normalized_actions,
        )

        return actions.to(torch.float32)

    def _compute_action_mean(self, obs: dict, instr: str, *, grad: bool) -> torch.Tensor:
        """Run the VLA + action head, return action mean [chunk, action_dim].

        grad=True dispatches to the manual forward (`_compute_action_mean_grad_safe`)
        which preserves gradient flow into LoRA. Upstream `predict_action` does an
        explicit .detach().numpy() at the action-head exit, severing grad — fine
        for inference, broken for RL training.

        grad=False uses upstream predict_action (faster, can use any future kv-cache
        optimizations the OFT authors add).
        """
        if grad:
            return self._compute_action_mean_grad_safe(obs, instr)

        inputs = self._build_vla_inputs(obs, instr)
        ctx = torch.no_grad()
        with ctx:
            # predict_action returns (action, _) where action is
            # [chunk, action_dim] un-normalized
            action, _ = self.vla.predict_action(
                **{k: v for k, v in inputs.items() if k != "proprio"},
                unnorm_key=self.unnorm_key,
                do_sample=False,
                proprio=inputs["proprio"],
                proprio_projector=self.proprio_projector,
                action_head=self.action_head,
                use_film=False,
            )
        # `predict_action` returns numpy by default in some OFT revs.
        # Coerce to tensor on device.
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).to(self.device, dtype=torch.float32)
        elif action.dtype != torch.float32:
            action = action.to(torch.float32)
        # Ensure 2D [chunk, action_dim]
        if action.ndim == 1:
            action = action.unsqueeze(0).expand(self.action_chunk, -1)
        assert action.shape == (self.action_chunk, self.action_dim), (
            f"unexpected action shape {action.shape}, want "
            f"({self.action_chunk}, {self.action_dim})"
        )
        return action

    # ----- public API consumed by GRPO ---------------------------------------

    @torch.no_grad()
    def act(self, obs: dict, instr: str) -> torch.Tensor:
        """Deterministic action for eval. [action_chunk, action_dim] on device."""
        return self._compute_action_mean(obs, instr, grad=False)

    @torch.no_grad()
    def sample(
        self, obs: dict, instr: str, n_samples: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample n_samples action chunks via Gaussian noise around the OFT mean.

        Returns:
            actions:  [n_samples, action_chunk, action_dim]   float32 on device
            log_prob: [n_samples]                              float32 on device
        """
        mean = self._compute_action_mean(obs, instr, grad=False)  # [C, A]
        std = self.log_std.exp()                                  # [A]
        eps = torch.randn(n_samples, *mean.shape,
                          device=mean.device, dtype=mean.dtype)
        actions = mean.unsqueeze(0) + eps * std.view(1, 1, -1)
        log_prob = self._gaussian_log_prob(actions, mean, std)    # [n_samples]
        return actions, log_prob

    def recompute_log_prob(
        self, obs: dict, instr: str, actions: torch.Tensor,
    ) -> torch.Tensor:
        """Re-evaluate log_prob of given actions under the CURRENT policy.

        actions: [G, action_chunk, action_dim]
        Returns: [G]   — differentiable through LoRA params + log_std.
        """
        mean = self._compute_action_mean(obs, instr, grad=True)   # [C, A]
        std = self.log_std.exp()
        return self._gaussian_log_prob(actions, mean, std)

    def _gaussian_log_prob(
        self, actions: torch.Tensor, mean: torch.Tensor, std: torch.Tensor,
    ) -> torch.Tensor:
        """Closed-form independent Gaussian log-prob, summed over (chunk, dim).

        actions: [G, C, A]  mean: [C, A]  std: [A]
        Returns: [G]
        """
        # Broadcast: [G,C,A] - [1,C,A] = [G,C,A]
        diff = actions - mean.unsqueeze(0)
        # log p(a) = -0.5*((a-mu)/sigma)^2 - log(sigma) - 0.5*log(2*pi)
        log_two_pi = math.log(2.0 * math.pi)
        lp = -0.5 * (diff / std.view(1, 1, -1)) ** 2
        lp = lp - torch.log(std).view(1, 1, -1) - 0.5 * log_two_pi
        return lp.sum(dim=(1, 2))   # [G]

    # ----- LoRA save / load --------------------------------------------------

    def save_lora(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        # save_pretrained on the PEFT-wrapped model writes only the adapter
        self.vla.save_pretrained(str(path))
        torch.save(
            {"log_std": self.log_std.detach().cpu(),
             "unnorm_key": self.unnorm_key},
            path / "policy_extras.pt",
        )
        print(f"[policy] saved LoRA adapter + log_std to {path}")

    def load_lora(self, path: Union[str, Path]) -> None:
        from peft import PeftModel
        path = Path(path)
        # NOTE: this reloads the adapter ON TOP of the existing LoRA. For a
        # fresh load it's simpler to construct a new OpenVLAOFTPolicy and only
        # call load_lora on it.
        self.vla = PeftModel.from_pretrained(self.vla, str(path))
        extras = torch.load(path / "policy_extras.pt", map_location="cpu")
        with torch.no_grad():
            self.log_std.copy_(extras["log_std"].to(self.log_std.device))
        print(f"[policy] loaded LoRA adapter + log_std from {path}")
