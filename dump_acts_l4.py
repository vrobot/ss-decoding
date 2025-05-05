import torch, argparse, json
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--layer', type=int, default=12)        # drafter layer
parser.add_argument('--n',     type=int, default=50_000)
args = parser.parse_args()

MODEL_ID = "meta-llama/Llama-4-Scout-17B-16E"      # or the BF16 Maverick repo 
tok = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")

model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,      # still fine – weights dequantised on the fly
            device_map="auto",
            attn_implementation="flash_attention_2",
            trust_remote_code=True)          # Unsloth models need this
model.eval()
DATA_FILE = "/mnt/ss-decoding/datasets/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json"
TRAIN_ROWS = "train[:50000]"
ds = load_dataset("json", data_files=DATA_FILE, split=TRAIN_ROWS)

HIDDEN, LOGITS = [], []
with torch.no_grad():
    for row in tqdm(ds, total=args.n):
        conv = row["conversations"]
        if not conv:
            continue
        prompt = conv[0]["value"]
        ids = tok(prompt, return_tensors="pt",
                  truncation=True, max_length=256).to("cuda")

        out = model(**ids, use_cache=False, output_hidden_states=True)
        h    = out.hidden_states[args.layer+1][:,-1,:].float().cpu()  # (1,5120)
        log  = out.logits[:, -1, :].float().cpu()                     # (1,V)
        HIDDEN.append(h.squeeze(0));  LOGITS.append(log.squeeze(0))
        if len(HIDDEN) == args.n: break

torch.save({"X": torch.stack(HIDDEN), "Y": torch.stack(LOGITS)},
           f"lsq_data_l{args.layer}.pt")
