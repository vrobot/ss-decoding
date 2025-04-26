from transformers import AutoProcessor, AutoModelForImageTextToText   
import torch
from datasets import load_dataset


NUM_LAYERS = 48
BATCH_SIZE = 32
NUM_PROMPTS = 8192

mmlu = load_dataset("cais/mmlu", "all")
processor = AutoProcessor.from_pretrained("meta-llama/Llama-4-Scout-17B-16E-Instruct")
model = AutoModelForImageTextToText.from_pretrained(
    "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    device_map="auto",
    torch_dtype=torch.bfloat16
)
model.eval()

def make_hook(layer_idx):
    def hook_fn(module, inp, out):
        key = f'post_attention_layernorm_{layer_idx}'
        activations[key] = torch.concat([activations[key], out.detach().cpu()], dim=0)
    return hook_fn


PROMPT = (
    "Answer the following question by picking one of the choices given.\n"
    "Only output the content of the choice and nothing else. Give the actual content of the choice, not the index.\n"
    "Question: {question}\n"
    "Choices: {choices}\n"
    "Answer: "
)

prompts = [
    PROMPT.format(question=mmlu["test"][i]["question"], choices=mmlu["test"][i]["choices"])
    for i in range(8192)
]


input_ids = processor(text=prompts, return_tensors="pt", padding=True, padding_side="right").input_ids
input_ids = input_ids.view(-1, BATCH_SIZE, input_ids.shape[-1])

activations = {}

for i in range(NUM_LAYERS):
    submod = model.language_model.model.layers[i].post_attention_layernorm
    hook = submod.register_forward_hook(make_hook(i))

submod = model.language_model.model.layers[i].post_attention_layernorm
hook = submod.register_forward_hook(make_hook(i))


_ = model(input_ids)

def step_forward(input_ids):
    output = model(input_ids)
    next_id = output.logits[0][-1].argmax()
    if next_id == processor.tokenizer.eos_token_id: return
    print(processor.decode(next_id))
    input_ids = torch.cat([input_ids, torch.tensor([next_id]).to("cuda").unsqueeze(0)], dim=1)
    return input_ids

def step_forward(input_ids):
    output = model(input_ids=input_ids, use_cache=False)  # explicit
    next_id = output.logits[:, -1, :].argmax(dim=-1)  # batch safe
    if next_id.item() == processor.tokenizer.eos_token_id:
        return None
    print(processor.decode(next_id.item()))
    next_token = next_id.unsqueeze(0)  # (1, 1) shape
    input_ids = torch.cat([input_ids, next_token], dim=1)  # concat on sequence dim
    return input_ids

def manual_greedy(prompt, model, tokenizer, max_new_tokens=100):
    # tokenize once
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]
    past = None
    out_ids = []
    for _ in range(max_new_tokens):
        # 1) forward with caching
        out = model(input_ids=input_ids, past_key_values=past, use_cache=True)
        logits = out.logits[:, -1, :]
        past   = out.past_key_values
        # 2) pick next token
        next_id = logits.argmax(dim=-1)  # shape (1,)
        if next_id.item() == tokenizer.eos_token_id:
            break
        out_ids.append(next_id.item())
        # 3) feed only the new token in the next step
        input_ids = next_id.unsqueeze(0)
    return tokenizer.decode(out_ids, skip_special_tokens=True)

# usage:
print(manual_greedy(prompts[0], model, processor.tokenizer, max_new_tokens=100))


def manual_greedy(prompt, model, tokenizer, max_new_tokens=100):
    inputs    = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]        # (1, seq_len)
    past      = None
    out_ids   = []
    for _ in range(max_new_tokens):
        out    = model(input_ids=input_ids,
                       past_key_values=past,
                       use_cache=True)
        logits = out.logits[:, -1, :]       # (1, vocab)
        past   = out.past_key_values
        # get a Python int
        next_id = int(logits.argmax(dim=-1).item())
        if next_id == tokenizer.eos_token_id:
            break
        out_ids.append(next_id)
        # build a fresh tensor of shape (1,1) from the int
        input_ids = torch.tensor([[next_id]],
                                 device=model.device,
                                 dtype=torch.long)
    return tokenizer.decode(out_ids, skip_special_tokens=True)