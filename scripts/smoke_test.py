import os, torch, transformers as tx
model_id = os.environ["MODEL_NAME"]

model = tx.AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    load_in_4bit=True,
    torch_dtype=torch.float16,
    trust_remote_code=True,
)
tokenizer = tx.AutoTokenizer.from_pretrained(model_id)
print("✅ Transformer + 4‑bit model loaded OK")
