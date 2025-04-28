from transformers import AutoProcessor, AutoModelForImageTextToText   
import torch
from datasets import load_dataset

BATCH_SIZE = 2
NUM_PROMPTS = 16

dataset_name = "cnn_dailymail"
config_name  = "3.0.0"
train_ds = load_dataset(dataset_name, config_name, split="train")

# mmlu = load_dataset("cais/mmlu", "all")
processor = AutoProcessor.from_pretrained("meta-llama/Llama-4-Scout-17B-16E-Instruct", cache_dir="/mnt/ss-decoding/models/Llama-4-Scout-17B-16E-Instruct/")
model = AutoModelForImageTextToText.from_pretrained(
    "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    device_map="auto",
    torch_dtype=torch.bfloat16,
    cache_dir="/mnt/ss-decoding/models/Llama-4-Scout-17B-16E-Instruct/"
)
model.eval()

def make_hook(layer_idx):
    def hook_fn(module, inp, out):
        h = out[0].detach().cpu().view(-1, out[0].shape[-1])
        key = f'post_attention_layernorm_{layer_idx}'
        if key not in activations:
            activations[key] = h
        else:
            activations[key] = torch.cat([activations[key], h], dim=0)
    return hook_fn


PROMPT = (
    "Summarize the highlights of the following article text:\n"
    "{text}\n"
    "Summary: "
)

prompts = [
    PROMPT.format(text=train_ds[i]["article"])
    for i in range(NUM_PROMPTS)
]

input_ids = processor(text=prompts, return_tensors="pt", padding=True, padding_side="left").input_ids
input_ids = input_ids.view(-1, BATCH_SIZE, input_ids.shape[-1])

activations = {}

for i in range(len(model.language_model.model.layers)):
    submod = model.language_model.model.layers[i]
    hook = submod.register_forward_hook(make_hook(i))

outs = []
for i, batch in enumerate(input_ids):
    print(i)
    out = model.generate(batch, max_new_tokens=128, do_sample=False)
    outs.append(out)

out = torch.stack(outs)
torch.save(out, f"out_{NUM_PROMPTS}.pt")
torch.save(activations, f"activations_{NUM_PROMPTS}.pt")
