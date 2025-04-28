from transformers import AutoProcessor, AutoModelForImageTextToText   
import torch

processor = AutoProcessor.from_pretrained("meta-llama/Llama-4-Scout-17B-16E-Instruct", cache_dir="/mnt/ss-decoding/models/Llama-4-Scout-17B-16E-Instruct/")
model = AutoModelForImageTextToText.from_pretrained(
    "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    device_map="auto",
    torch_dtype=torch.bfloat16,
    cache_dir="/mnt/ss-decoding/models/Llama-4-Scout-17B-16E-Instruct/"
)
model.eval()

activation = {}

def hook(module, inp, out):
    activation["h48"] = out[0].detach().cpu()

handle = model.language_model.model.layers[47].register_forward_hook(
    hook
)

prompt = "The meaning of life is "
def compare(prompt):
    input_ids = processor(text=prompt, return_tensors="pt").input_ids.to(model.device)
    output = model.generate(input_ids, max_new_tokens=1, do_sample=False)
    print("auto generated:", processor.decode(output[0]))
    h48 = activation["h48"][0][-1][:].view(1, 1, 5120)
    with torch.no_grad():
        h_norm = model.language_model.model.norm(h48)    
        logits = model.language_model.lm_head(h_norm)
        next_id = logits[:, -1, :].argmax(-1)
    print("manual generated:", processor.decode(next_id))
    prompt += processor.decode(next_id)
    return prompt


for _ in range(10):
    prompt = compare(prompt)
