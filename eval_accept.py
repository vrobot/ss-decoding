import torch, argparse, json
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--layer", type=int, nargs="+", default=[12])
parser.add_argument("--n", type=int, default=5_000)
args = parser.parse_args()

n_gpus = torch.cuda.device_count()
devices = [f"cuda:{i}" for i in range(n_gpus)]

Ws = []
for idx, layer in enumerate(args.layer):
    dev = devices[idx % n_gpus]
    W = torch.load(f"lsq_data/weights/ee_head_l{layer}.pt",
                   map_location=dev)
    Ws.append(W.to(torch.bfloat16))

MODEL_ID = "meta-llama/Llama-4-Scout-17B-16E"
tok    = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
model  = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation = "flash_attention_2",
            trust_remote_code=True,
            cache_dir="/mnt/ss-decoding/models/meta-llama/Llama-4-Scout-17B-16E"
)
model.eval()

DATA_FILE = "/mnt/ss-decoding/datasets/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json"
TRAIN_ROWS = f"train[50000:{50000+args.n}]"
ds = load_dataset("json", data_files=DATA_FILE, split=TRAIN_ROWS)

hits=[0]*len(args.layer)
tot=0
with torch.no_grad():
    for row in tqdm(ds, total=args.n):
        conv = row["conversations"]
        if not conv:
            continue
        prompt = conv[0]["value"]
        ids = tok(prompt,
                  return_tensors="pt",
                  truncation=True, max_length=256).to("cuda")
        out = model(**ids, use_cache=False, output_hidden_states=True)
        for i, layer in enumerate(args.layer):
            h = out.hidden_states[i+1][:,-1,:]
            dev = h.device
            draft = (h @ Ws[i].to(dev).T).float() 
            gold = out.logits[:,-1,:].argmax(-1)
            pred = draft.argmax(-1)
            hits[i] += (pred == gold).sum().item()
        tot += 1
acc = [hits[i]/tot for i in range(len(args.layer))]

for i, layer in enumerate(args.layer):
    print(f"L{layer} accept-rate: {acc[i]:6.2%}")
