"""vjepa2-grpo: Imagined-trajectory GRPO on V-JEPA-2 latents."""

__version__ = "0.1.0"

from .encoder import VJepa2Encoder, load_encoder, EMBED_DIM
from .predictor import BlockCausalACPredictor
from .critic import ProgressCritic
from .faiss_anchor import FaissAnchorBuffer

__all__ = [
    "VJepa2Encoder", "load_encoder", "EMBED_DIM",
    "BlockCausalACPredictor",
    "ProgressCritic",
    "FaissAnchorBuffer",
]
