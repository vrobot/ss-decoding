from transformers import AutoProcessor, AutoModelForImageTextToText   
import torch
from datasets import load_dataset

NUM_LAYERS = 48
BATCH_SIZE = 4
NUM_PROMPTS = 1024

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
        print(out.shape)
        chunk = out.view(-1, out.shape[-1]).detach().cpu()
        key = f'post_attention_layernorm_{layer_idx}'
        if key not in activations:
            # first time seeing this key, just store the tensor
            activations[key] = chunk
        else:
            # already exists, append along dim=0
            activations[key] = torch.cat([activations[key], chunk], dim=0)
    return hook_fn


# PROMPT = (
#     "Answer the following question by picking one of the choices given.\n"
#     "Only output the content of the choice and nothing else. Give the actual content of the choice, not the index.\n"
#     "Question: {question}\n"
#     "Choices: {choices}\n"
#     "Answer: "
# )

PROMPT = (
    "Summarize the highlights of the following article text:\n"
    "{text}\n"
    "Summary: "
)

prompts = [
    PROMPT.format(text=train_ds[i]["article"])
    for i in range(NUM_PROMPTS)
]

input_ids = processor(text=prompts, return_tensors="pt", padding=True, padding_side="right").input_ids
input_ids = input_ids.view(-1, BATCH_SIZE, input_ids.shape[-1])
test_input_ids = input_ids[:4]

activations = {}

for i in [46, 47]:
    submod = model.language_model.model.layers[i].post_attention_layernorm
    hook = submod.register_forward_hook(make_hook(i))

out = torch.stack([model.generate(batch, max_new_tokens=64).cpu() for batch in test_input_ids])
torch.save(out, "out.pt")
torch.save(activations, "activations.pt")