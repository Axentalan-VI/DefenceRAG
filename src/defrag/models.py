"""Single source of truth for Gemma 4 model loading.

Every module loads models through here so the dtype constraint is applied in
exactly one place.

**The dtype constraint**, which is the reverse of the usual advice for these
cards. Gemma is trained in bf16 and carries activation magnitudes above fp16's
~65504 ceiling, so **Gemma in fp16 produces NaNs** - silently. We hit this with
EmbeddingGemma: every vector in a 1,913-chunk index came out NaN, and because
NaN normalises and sorts without complaint, the broken index still passed most
retrieval smoke probes on BM25 alone.

Kaggle's free tier gives T4 x2 or P100. The T4 is Turing (SM 7.5) with no
bf16 tensor cores, so bf16 there falls back to fp32 compute: slower, but
numerically correct. That is the right trade for a value you cannot eyeball.
Everything below therefore pins **bf16**, and never float16.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Kaggle Model handles, attached to the notebook as Model inputs so the demo
# runs with internet disabled.
REASONER = "google/gemma-4/transformers/gemma-4-12b-it-qat-q4_0-unquantized"
VERIFIER = "google/gemma-4/transformers/gemma-4-e4b-it"
EMBEDDER = "google/embeddinggemma/transformers/embeddinggemma-300m"

# For the CPU-backed HF Space, via llama.cpp / Ollama.
REASONER_GGUF = "google/gemma-4/gguf/gemma-4-12b-it-qat-q4_0-gguf"

# Where Kaggle mounts attached Model inputs.
KAGGLE_INPUT = Path("/kaggle/input")


@dataclass(frozen=True)
class LoadSpec:
    """How one role's model gets onto the GPU."""

    handle: str
    four_bit: bool
    approx_vram_gb: float   # once loaded, for budgeting against 16GB
    note: str = ""


SPECS: dict[str, LoadSpec] = {
    # QAT weights shipped unquantized: the model was *trained* aware of 4-bit
    # rounding, so re-quantizing to NF4 costs far less quality than quantizing
    # a normally-trained checkpoint would.
    "reasoner": LoadSpec(REASONER, four_bit=True, approx_vram_gb=8.0,
                         note="12B Unified, planner role, thinking mode on"),
    "verifier": LoadSpec(VERIFIER, four_bit=False, approx_vram_gb=5.0,
                         note="E4B, groundedness check, runs per-answer"),
}


def on_kaggle() -> bool:
    return KAGGLE_INPUT.exists() or "KAGGLE_KERNEL_RUN_TYPE" in os.environ


def resolve_path(handle: str, local_override: str | None = None) -> str:
    """Turn a Kaggle Model handle into a loadable path.

    On Kaggle, attached Model inputs mount under /kaggle/input/<slug>/<...>.
    Off Kaggle, kagglehub caches the download. An explicit override always wins,
    which is what lets the test suite and CI point at a fixture.
    """
    if local_override:
        return local_override

    if on_kaggle():
        slug = handle.split("/")[-1]
        for candidate in KAGGLE_INPUT.glob(f"**/{slug}"):
            return str(candidate)
        # Fall through to kagglehub rather than failing: the notebook may not
        # have the model attached yet.

    import kagglehub

    return kagglehub.model_download(handle)


def _dtype():
    import torch

    return torch.bfloat16   # never float16 - Gemma overflows it; see docstring


def quantization_config():
    """NF4 config for the reasoner, with bf16 compute."""
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=_dtype(),
    )


def load_causal_lm(role: str, local_override: str | None = None, device_map: str = "auto"):
    """Load the Gemma 4 causal LM for ``role`` ("reasoner" or "verifier").

    Returns ``(model, tokenizer)``.
    """
    try:
        spec = SPECS[role]
    except KeyError:
        raise ValueError(f"unknown role {role!r}; have {sorted(SPECS)}") from None

    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = resolve_path(spec.handle, local_override)

    kwargs: dict = {"device_map": device_map, "dtype": _dtype()}
    if spec.four_bit:
        kwargs["quantization_config"] = quantization_config()

    try:
        model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
    except TypeError:
        # transformers < 5 spells it torch_dtype.
        kwargs["torch_dtype"] = kwargs.pop("dtype")
        model = AutoModelForCausalLM.from_pretrained(path, **kwargs)

    tokenizer = AutoTokenizer.from_pretrained(path)
    return model, tokenizer


def load_embedder(local_override: str | None = None, device: str | None = None):
    """Load EmbeddingGemma wrapped for the retrieval index."""
    from .index import EmbeddingGemmaEmbedder

    return EmbeddingGemmaEmbedder(resolve_path(EMBEDDER, local_override), device=device)
