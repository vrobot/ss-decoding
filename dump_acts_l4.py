import torch, argparse, json
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument(
    "--layer",
    type=int,
    nargs="+",
    default=[12],
)     # drafter layer
parser.add_argument('--n',     type=int, default=50_000)
args = parser.parse_args()

MODEL_ID = "meta-llama/Llama-4-Scout-17B-16E"      # or the BF16 Maverick repo 
tok = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")

model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,      # still fine – weights dequantised on the fly
            device_map="auto",
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
            cache_dir="/mnt/ss-decoding/models/meta-llama/Llama-4-Scout-17B-16E"
)          # Unsloth models need this
model.eval()

dataset_name = "cnn_dailymail"
config_name  = "3.0.0"
ds = load_dataset(dataset_name, config_name, split=f"train[:{args.n}]")

PROMPT = (
    "Summarize the following article text\n"
    "{text}\n"
    "Summary: "
)

HIDDEN, LOGITS = [], []
with torch.no_grad():
    for row in tqdm(ds, total=args.n):
        prompt = PROMPT.format(text=row['article'])
        ids = tok(prompt, return_tensors="pt",
                  truncation=True, max_length=256).to("cuda")
        out = model(**ids, use_cache=False, output_hidden_states=True)
        hs = torch.stack(out.hidden_states, dim=0)
        layers = torch.tensor(args.layer)
        h    = hs[layers, 0, :, :].float().cpu()  # (L, 256, 5120)
        log  = out.logits[0, :, :].float().cpu()                     # (512, V)
        HIDDEN.append(h);  LOGITS.append(log)

print(len(HIDDEN), len(LOGITS))
X = torch.cat(HIDDEN, dim=1)
Y = torch.cat(LOGITS, dim=0)

# numpy storage is more efficient
X_np = X.cpu().numpy()
Y_np = Y.cpu().numpy()
for i, layer in enumerate(args.layer):
    torch.save(X_np[i, :, :], f"lsq_data/X_l{layer}.bin")
torch.save(Y_np, f"lsq_data/Y.bin")
