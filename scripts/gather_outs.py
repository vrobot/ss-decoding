from transformers import AutoProcessor, AutoModelForImageTextToText   
import torch
import torch.nn as nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
activations_47 = torch.load("/mnt/ss-decoding/clean_activations/activations_nopadding_256.pt")['layer_47'].to(device)

processor = AutoProcessor.from_pretrained("meta-llama/Llama-4-Scout-17B-16E-Instruct", cache_dir="/mnt/ss-decoding/models/Llama-4-Scout-17B-16E-Instruct/")
model = AutoModelForImageTextToText.from_pretrained(
    "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    device_map="auto",
    torch_dtype=torch.bfloat16,
    cache_dir="/mnt/ss-decoding/models/Llama-4-Scout-17B-16E-Instruct/"
)
model.eval()

out = torch.tensor([]).to(device)

for i in range(activations_47.shape[0]):
    h = activations_47[i].view(1, -1, 5120)
    h_norm = model.language_model.model.norm(h)
    logits = model.language_model.lm_head(h_norm)
    next_ids = logits[0].argmax(-1).to(device)
    out = torch.cat([out, next_ids])
