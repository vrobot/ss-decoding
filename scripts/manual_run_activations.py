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

# activation = {}

# def hook(module, inp, out):
#     activation["h48"] = out[0].detach().cpu()

# handle = model.language_model.model.layers[47].register_forward_hook(
#     hook
# )

# prompt = "The meaning of life is "
# def compare(prompt):
#     input_ids = processor(text=prompt, return_tensors="pt").input_ids.to(model.device)
#     output = model.generate(input_ids, max_new_tokens=1, do_sample=False)
#     print("auto generated:", processor.decode(output[0]))
#     h48 = activation["h48"][0][-1][:].view(1, 1, 5120)
#     with torch.no_grad():
#         h_norm = lm_model.language_model.model.norm(h48)    
#         logits = lm_model.language_model.lm_head(h_norm)
#         next_id = logits[:, -1, :].argmax(-1)
#     print("manual generated:", processor.decode(next_id))
#     prompt += processor.decode(next_id)
#     return prompt


# for _ in range(10):
#     prompt = compare(prompt)


activations_47 = torch.load("activations/post_attention_layernorm_47.pt")
# activations_46 = torch.load("activations/post_attention_layernorm_46.pt")
# activations_45 = torch.load("activations/post_attention_layernorm_45.pt")
activations_40 = torch.load("activations/post_attention_layernorm_40.pt")
# activations_35 = torch.load("activations/post_attention_layernorm_35.pt")
out = torch.load("activations/outs_256.pt")

def compare(idx):
    with torch.no_grad():
        h48 = activations['post_attention_layernorm_47'][idx*2+1].view(1, 1, -1)
        h_norm = model.language_model.model.norm(h48)    
        logits = model.language_model.lm_head(h_norm)
        next_id_48 = logits[:, -1, :].argmax(-1)
        h47 = activations['post_attention_layernorm_40'][idx*2+1].view(1, 1, -1)
        h_norm = model.language_model.model.norm(h47)    
        logits = model.language_model.lm_head(h_norm)
        next_id_47 = logits[:, -1, :].argmax(-1)
        print(idx, processor.decode(next_id_48), processor.decode(next_id_47), processor.decode(out[-1][-1][idx]))

class ActivationMapping(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x):
        return self.proj(x)

act_map = ActivationMapping(dim=5120)
act_map.load_state_dict(torch.load("activations/activation_mapping40to47.pth"))
act_map.eval()
act_map = act_map.to(torch.bfloat16)

orig_count_correct = 0
count_correct = 0
def compare(idx):
    global count_correct
    global orig_count_correct
    with torch.no_grad():
        h47 = activations_47[idx*2+1].view(1, 1, -1)
        h_norm = model.language_model.model.norm(h47)    
        logits = model.language_model.lm_head(h_norm)
        next_id_47 = logits[:, -1, :].argmax(-1)
        h40 = activations_40[idx*2+1].view(1, 1, -1)
        h_norm = model.language_model.model.norm(h40)    
        logits = model.language_model.lm_head(h_norm)
        next_id_40 = logits[:, -1, :].argmax(-1)
        proj_out = act_map(activations_40[idx*2+1])
        p_norm = model.language_model.model.norm(proj_out)
        logits = model.language_model.lm_head(p_norm)
        p_next_id = logits.argmax(-1)
        if next_id_47 == next_id_40:
            orig_count_correct += 1
        if next_id_47 == p_next_id:
            count_correct += 1
        print(idx, processor.decode(next_id_47), processor.decode(p_next_id), processor.decode(next_id_40))


for i in range(100):
    compare(-i)