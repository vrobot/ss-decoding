from transformers import AutoProcessor, AutoModelForImageTextToText   
import torch
from datasets import load_dataset
import time

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

BATCH_SIZE = 1
NUM_PROMPTS = 256

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
        h = out[0].detach().view(-1, out[0].shape[-1])
        key = f'layer_{layer_idx}'
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

# input_ids = processor(text=prompts, return_tensors="pt").input_ids.to("cuda")
# input_ids = input_ids.view(-1, 1, input_ids.shape[-1])

activations = {}

for i in [11, 23, 35, 41, 47]:
    submod = model.language_model.model.layers[i]
    hook = submod.register_forward_hook(make_hook(i))

outs = []
for i, prompt in enumerate(prompts):
    start = time.time()
    print(i)
    input_ids = processor(text=prompt, return_tensors="pt").input_ids.to("cuda")
    with torch.inference_mode():
        out = model.generate(input_ids, max_new_tokens=64, do_sample=False)
    end = time.time()
    print(f"Time taken: {end - start} seconds")
    outs.append(out)

torch.save(outs, f"out_nopadding_{NUM_PROMPTS}.pt")
torch.save(activations, f"activations_nopadding_{NUM_PROMPTS}.pt")
