import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

class _CustomGenerate:
    def generate(self, *args, **kwargs):
        print("generating")
        return 42

class SSDecodingModel:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        # Remove custom kwargs before calling AutoModelForCausalLM.from_pretrained
        custom_kwargs = {}
        if "layer_skip" in kwargs:
            custom_kwargs["layer_skip"] = kwargs.pop("layer_skip")
        if "layer_proj_weights" in kwargs:
            custom_kwargs["layer_proj_weights"] = kwargs.pop("layer_proj_weights")
        base = AutoModelForCausalLM.from_pretrained(*args, **kwargs)

        if not isinstance(base, _CustomGenerate):
            Patched = type(
                "SSDecodingModel",
                (_CustomGenerate, base.__class__),
                {},
            )
            base.__class__ = Patched

        base.layer_skip = custom_kwargs.get("layer_skip", -1)
        layer_proj_weights_path = custom_kwargs.get("layer_proj_weights")
        device_map = kwargs.get("device_map", "cuda")
        print(layer_proj_weights_path, device_map, base.layer_skip)
        base.layer_proj_weights = torch.load(layer_proj_weights_path).to(base.device) if layer_proj_weights_path else None

        return base

MODEL_ID = "meta-llama/Llama-4-Scout-17B-16E"
CACHE_DIR = "/mnt/ss-decoding/models/meta-llama/Llama-4-Scout-17B-16E"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = SSDecodingModel.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",
    trust_remote_code=True,
    cache_dir=CACHE_DIR,
    layer_skip=12,
    layer_proj_weights="/mnt/ss-decoding/vk/ss-decoding/full_sweep/heads/h12.pt"

)

out = model.generate()
print(out)