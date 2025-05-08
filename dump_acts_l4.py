import torch, argparse, json
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import h5py
import numpy as np

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

logits_size = 202048
hidden_size = 5120
L = len(args.layer)

with h5py.File("lsq_data.h5", "w") as f:
    X = f.create_dataset("X",
                         shape=(L, 0, hidden_size),
                         maxshape=(L, None, hidden_size),
                         dtype="float32",
                         chunks=(L,1,hidden_size))
    Y = f.create_dataset("Y",
                         shape=(0, logits_size),
                         maxshape=(None, logits_size),
                         dtype="float32",
                         chunks=(1,logits_size))
    with torch.no_grad():
        for row in tqdm(ds, total=args.n):
            prompt = PROMPT.format(text=row['article'])
            ids = tok(prompt, return_tensors="pt",
                    truncation=True, max_length=256).to("cuda")
            out = model(**ids, use_cache=False, output_hidden_states=True)
            hs = torch.stack(out.hidden_states, dim=0)
            layers = torch.tensor(args.layer)
            h    = hs[layers, 0, :, :].float().cpu()  # (L, 256, 5120)
            log  = out.logits[0, :, :].float().cpu()  # (512, V)
            h_np   = h.numpy()
            log_np = log.numpy()
            X.resize((L, X.shape[1] + h_np.shape[1], hidden_size))
            X[:, -h_np.shape[1]:, :] = h_np
            Y.resize((Y.shape[0] + log_np.shape[0], logits_size))
            Y[-log_np.shape[0]:, :]  = log_np

    for i, layer in enumerate(args.layer):
        np.save(f"lsq_data/X_l{layer}.npy", X[i,:,:])
    np.save("lsq_data/Y.npy", Y)
