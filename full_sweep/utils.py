import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

def load_model_and_tokenizer(model_id, cache_dir=None):
    """Load model and tokenizer with sensible defaults."""
    tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    tok.pad_token = tok.eos_token if tok.pad_token is None else tok.pad_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
        cache_dir=cache_dir
    ).eval()
    
    return model, tok

def load_prompts(data_file, n_prompts, split="train"):
    """Load prompts from ShareGPT format."""
    ds = load_dataset("json", data_files=data_file)
    ds_split = ds['train'].train_test_split(test_size=0.2, seed=42)
    prompts = []
    for i, row in enumerate(ds_split[split]):
        if len(prompts) >= n_prompts:
            break
        if row.get("conversations") and row["conversations"][0]["from"] == "human":
            prompts.append(row["conversations"][0]["value"])
    return prompts