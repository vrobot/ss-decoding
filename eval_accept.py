import torch, argparse, json
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
layer  = 12
MODEL_ID = "meta-llama/Llama-4-Scout-17B-16E"
W      = torch.load("ee_head_l12_l4.pt", map_location = "cuda")
W = W.to(torch.bfloat16)
W_repl = {torch.device(f"cuda:{d}"): W.to(f"cuda:{d}") for d in range(torch.cuda.device_count())}
tok    = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
model  = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16,
            device_map="auto", attn_implementation = "flash_attention_2", trust_remote_code=True).eval()
DATA_FILE = "/mnt/ss-decoding/datasets/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json"
TEST_ROWS = "train[50000:55000]"
test_ds = load_dataset("json", data_files=DATA_FILE, split=TEST_ROWS)
hits=tot=0
with torch.no_grad():
    for row in tqdm(test_ds, total=5_000):
        if not row["conversations"]:
            continue
        ids = tok(row["conversations"][0]["value"],
                  return_tensors="pt",
                  truncation=True, max_length=256).to("cuda")
        out = model(**ids, use_cache=False, output_hidden_states=True)
        h12 = out.hidden_states[layer+1][:,-1,:]
        dev = h12.device
        draft = (h12 @ W_repl[dev].T).float() 
        gold = out.logits[:,-1,:].argmax(-1)
        pred = draft.argmax(-1)
        hits += (pred == gold).sum().item()
        tot += 1
print(f"L{layer} accept-rate: {hits/tot:6.2%}")
