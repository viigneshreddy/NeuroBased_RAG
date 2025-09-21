# llm/providers/hf_local.py
# Lightweight Hugging Face LLM wrapper for local/Streamlit Cloud usage.

from __future__ import annotations

import os
from typing import Tuple, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ----------------------------
# Configuration (tweak here)
# ----------------------------
# Small, deploy-friendly default model. You can switch to Qwen/Qwen2.5-3B-Instruct if you have more RAM.
_DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Safe attention backend (no flash-attention requirements).
_ATTENTION_IMPL = "eager"

# For safety, avoid remote code unless the model explicitly requires it.
_TRUST_REMOTE_CODE = False

# Pin a specific commit for reproducibility (optional). Example: "a1b2c3d4e5..."
_REVISION: Optional[str] = None

# Read optional Hugging Face token from env (useful on Streamlit Cloud → App secrets).
_HF_TOKEN: Optional[str] = os.getenv("HF_TOKEN")


# ---------------------------------
# Optional Streamlit resource cache
# ---------------------------------
def _identity(x):
    return x

try:
    import streamlit as st
    cache_resource = st.cache_resource  # persists across reruns on Streamlit Cloud
except Exception:
    cache_resource = _identity  # no-op fallback when not in Streamlit


# ----------------------------
# Device & dtype selection
# ----------------------------
def _pick_device_and_dtype() -> Tuple[dict, torch.dtype]:
    """
    Chooses a device map and dtype based on availability.
    - Apple Silicon: MPS + float16
    - CUDA: auto + bfloat16
    - CPU: cpu + float32
    """
    if torch.backends.mps.is_available():
        return {"device_map": {"": "mps"}}, torch.float16
    if torch.cuda.is_available():
        return {"device_map": "auto"}, torch.bfloat16
    return {"device_map": "cpu"}, torch.float32


# ----------------------------
# Lazy, cached loader
# ----------------------------
@cache_resource
def _load(model_name: str = _DEFAULT_MODEL):
    """
    Loads and returns (tokenizer, model). Cached so it only loads once per session.
    """
    device_map_cfg, dtype = _pick_device_and_dtype()

    tok = AutoTokenizer.from_pretrained(
        model_name,
        revision=_REVISION,
        trust_remote_code=_TRUST_REMOTE_CODE,
        token=_HF_TOKEN,
    )
    # Ensure pad token exists for generation
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    mdl = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=_REVISION,
        trust_remote_code=_TRUST_REMOTE_CODE,
        low_cpu_mem_usage=True,
        attn_implementation=_ATTENTION_IMPL,
        dtype=dtype,                 # Use `dtype=` (newer Transformers) instead of torch_dtype
        device_map=device_map_cfg["device_map"],
        token=_HF_TOKEN,
    )

    # If you hit cache API mismatches with some models, uncomment:
    # mdl.config.use_cache = False

    return tok, mdl


# ----------------------------
# Text generation API
# ----------------------------
def generate(
    prompt: str,
    max_new_tokens: int = 220,
    temperature: float = 0.4,
    top_p: float = 0.95,
    repetition_penalty: float = 1.05,
    model_name: str = _DEFAULT_MODEL,
) -> str:
    """
    Simple text generation interface.
    """
    tok, mdl = _load(model_name)

    inputs = tok(prompt, return_tensors="pt")
    # Send tensors to the model's device
    inputs = {k: v.to(mdl.device) for k, v in inputs.items()}

    output_ids = mdl.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0),
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
        # use_cache=False,  # uncomment if you hit cache-related errors on some models
    )

    text = tok.decode(output_ids[0], skip_special_tokens=True)
    return text


# ----------------------------
# Optional: deterministic runs
# ----------------------------
def set_seed(seed: int = 42):
    """
    Set a global seed for reproducibility (best-effort).
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # MPS doesn't support full determinism yet, but we still set the CPU seed.
        pass
