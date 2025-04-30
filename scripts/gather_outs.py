from transformers import AutoProcessor, AutoModelForImageTextToText   
import torch
import torch.nn as nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
activations_47 = torch.load("/mnt/ss-decoding/activations/post_attention_layernorm_47.pt").to(device)

processor = AutoProcessor.from_pretrained("meta-llama/Llama-4-Scout-17B-16E-Instruct", cache_dir="/mnt/ss-decoding/models/Llama-4-Scout-17B-16E-Instruct/")
model = AutoModelForImageTextToText.from_pretrained(
    "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    device_map="auto",
    torch_dtype=torch.bfloat16,
    cache_dir="/mnt/ss-decoding/models/Llama-4-Scout-17B-16E-Instruct/"
)
model.eval()

BATCH_SIZE
out = 

def run_forward(start_idx: int, num_samples: int):
    with torch.no_grad():
        for i in range(num_samples):
            h = activations_47[start_idx+i].view(1, 1, -1)
            h_norm = model.language_model.model.norm(h)    
            logits = model.language_model.lm_head(h_norm)
            next_id = logits[:, -1, :].argmax(-1)
            print(processor.decode(next_id))


out = torch.tensor([]).to(device)

for i in range(activations_47.shape[0]):
    h = activations_47[i].view(1, -1, 5120)
    h_norm = model.language_model.model.norm(h)
    logits = model.language_model.lm_head(h_norm)
    next_ids = logits[0].argmax(-1).to(device)
    out = torch.cat([out, next_ids])
