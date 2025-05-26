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
        base = AutoModelForCausalLM.from_pretrained(*args, **kwargs)

        if not isinstance(base, _CustomGenerate):
            Patched = type(
                "SSDecodingModel",
                (_CustomGenerate, base.__class__),
                {},
            )
            base.__class__ = Patched

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
    cache_dir=CACHE_DIR
)

out = model.generate()
print(out)