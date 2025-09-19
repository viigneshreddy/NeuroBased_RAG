# llm/providers/hf_local.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

_DEFAULT_MODEL = "microsoft/Phi-3-mini-4k-instruct"
_ATTENTION_IMPL = "eager"     # Mac-friendly (no flash-attn)
_TRUST_REMOTE_CODE = False     # Safer default
_REVISION = None               # Optional: pin a specific commit SHA

_tokenizer = None
_model = None

def _pick_device_and_dtype():
    if torch.backends.mps.is_available():           # Apple Silicon GPU
        return {"device_map": {"": "mps"}, "dtype": torch.float16}
    elif torch.cuda.is_available():                 # NVIDIA GPU
        return {"device_map": "auto", "dtype": torch.bfloat16}
    else:                                           # CPU
        return {"device_map": "cpu", "dtype": torch.float32}

def _load(model_name: str = _DEFAULT_MODEL):
    global _tokenizer, _model
    if _tokenizer is not None and _model is not None:
        return _tokenizer, _model

    cfg = _pick_device_and_dtype()
    _tokenizer = AutoTokenizer.from_pretrained(
        model_name, revision=_REVISION, trust_remote_code=_TRUST_REMOTE_CODE
    )
    if _tokenizer.pad_token_id is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    _model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=cfg["dtype"],                 # <- use dtype (not torch_dtype)
        device_map=cfg["device_map"],
        attn_implementation=_ATTENTION_IMPL,
        low_cpu_mem_usage=True,
        revision=_REVISION,
        trust_remote_code=_TRUST_REMOTE_CODE,
    )
    # If you still hit a cache mismatch, uncomment:
    # _model.config.use_cache = False
    return _tokenizer, _model

def generate(prompt: str, max_new_tokens: int = 220, temperature: float = 0.4, model_name: str = _DEFAULT_MODEL) -> str:
    tok, mdl = _load(model_name)
    inputs = tok(prompt, return_tensors="pt").to(mdl.device)
    output = mdl.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature,
        repetition_penalty=1.05,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
        # use_cache=False,  # <- enable if you still hit the DynamicCache/seen_tokens error
    )
    return tok.decode(output[0], skip_special_tokens=True)
