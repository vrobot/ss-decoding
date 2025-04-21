import os, torch, transformers as tx
m   = os.environ["MODEL_NAME"]
tok = tx.AutoTokenizer.from_pretrained(m)
_   = tx.AutoModelForCausalLM.from_pretrained(m, device_map="auto")
print("✅ smoke-test OK - model loaded; VRAM:",
      round(torch.cuda.memory_allocated()/1e9, 2), "GB")
