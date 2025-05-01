# ── constants you already have ─────────────────────────────────────────────
hidden12   = layer12_h                    # (B, T, hidden_size)  ← capture from layer-12
B, T, _    = hidden12.shape
# device     = hidden12.device
mask4d     = torch.ones(B, 1, T, T, device=device, dtype=torch.bool)
pos_ids    = torch.arange(T, device=device).expand(B, -1)         # (B, T) long

# ── build RoPE tensor **once** from the global module ─────────────────────
rope       = model.language_model.model.rotary_emb                # shared instance
inv_freq   = rope.inv_freq                                        # (rot_dim,)
angles     = torch.outer(torch.arange(T, device=device, dtype=inv_freq.dtype),
                         inv_freq)                                # (T, rot_dim)
freqs_cis  = torch.polar(torch.ones_like(angles), angles)         # (T, rot_dim)
freqs_cis  = freqs_cis.unsqueeze(0)                               # (1, T, rot_dim)

# ── run layers 12 … N-1 with the SAME tensor each time ────────────────────
h = hidden12
for i, lyr in enumerate(model.language_model.model.layers[12:]):
    print(i)
    with autocast(dtype=torch.bfloat16):
        h = lyr(
            h,
            attention_mask      = None,
            position_ids        = pos_ids,
            position_embeddings = freqs_cis,   # shared RoPE tensor
            use_cache           = False,
            cache_position      = cache_pos,
            output_attentions   = False,
        )[0]

# ── final projection ──────────────────────────────────────────────────────
with autocast(dtype=torch.bfloat16):
    logits_partial = model.language_model.lm_head(
        model.language_model.model.norm(h)
    )

# (optional) compare with a clean forward
with torch.no_grad():
    logits_full = model(input_ids, use_cache=False).logits
print(torch.allclose(logits_partial, logits_full, atol=1e-4))  # → True


with autocast(dtype=torch.bfloat16):
    h = lyr(
        h,
        attention_mask      = mask4d,
        position_ids        = pos_ids,
        position_embeddings = freqs_cis,   # shared RoPE tensor
        cache_position      = cache_pos,
        use_cache           = False,
        output_attentions   = False,
    )[0]


with autocast(dtype=torch.bfloat16):
    for lyr in model.language_model.model.layers[12:]:
        h = lyr(
            h,
            attention_mask=None,           # ⚠︎ let Flash-Attn build it
            position_ids=pos_ids,
            position_embeddings=freqs,     # constant across layers
            cache_position=cache,          # ⚠︎ required by some layers
            use_cache=False,
            output_attentions=False,
        )[0]

with autocast(dtype=torch.bfloat16):
    logits_replay = model.language_model.lm_head(
        model.language_model.model.norm(h)
    )


import torch

def max_err(a, b):                      # (helper)
    return (a - b).abs().max().item()

with torch.no_grad():
    out = model(input_ids,
                output_hidden_states=True,
                use_cache=False)
    ref = out.hidden_states

    # layer-12 output you captured
    h = ref[13]                                       # ← check this index!
    B, T, _ = h.shape
    pos = torch.arange(T, device=h.device).expand(B, -1)
    cache = pos

    # RoPE once
    inv = model.language_model.model.rotary_emb.inv_freq
    angles = torch.outer(torch.arange(T, device=inv.device, dtype=inv.dtype), inv)
    freqs = torch.polar(torch.ones_like(angles), angles).unsqueeze(0)   # (1,T,d/2)

    # walk through layers 12…N-1 and compare
    for i, lyr in enumerate(model.language_model.model.layers[12:], start=12):
        h = lyr(h,
                attention_mask=None,           # Flash-Attn builds its own
                position_ids=pos,
                position_embeddings=freqs,
                cache_position=cache,          # needed by layers w/ attn_scale
                use_cache=False)[0]

        err = max_err(h, ref[i + 1])           # ref hidden after this layer
        print(f"layer {i:02d}  max err {err:.2e}")
        if err > 1e-7:                         # ← first bad layer
            break

h = lyr(h,
        attention_mask=None,           # Flash-Attn builds its own
        position_ids=pos,
        position_embeddings=freqs,
        cache_position=cache,          # needed by layers w/ attn_scale
        use_cache=False)[0]

h = lyr(h,
        attention_mask=None,           # Flash-Attn builds its own
        position_ids=pos,
        position_embeddings=freqs,
        cache_position=cache,          # needed by layers w/ attn_scale
        use_cache=False)[0]


with torch.no_grad():
    out = model(ids,
                output_hidden_states=True,
                use_cache=False)


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

import torch

prompt = "how do i make pasta?"
ids = processor(text=prompt, return_tensors="pt").input_ids.to(model.device)

idx = 12                                  # layer number you want to start from
layer = model.language_model.model.layers[idx]

# ── 1. grab the exact tensors this layer receives and produces ─────────────
grab = {}

def pre(_, args, kwargs):                 # hidden_states is the only positional
    grab["inp"] = args[0]
    grab["kw"]  = kwargs                 # attention_mask, position_ids, etc.

def post(_, __, out):                     # out is a tuple; [0] = hidden_states
    grab["ref_out"] = out[0]

h1 = layer.register_forward_pre_hook(pre,  with_kwargs=True)
h2 = layer.register_forward_hook(post)    # after-forward hook

with torch.no_grad():
    auto_out = model(ids, output_hidden_states=True, use_cache=False)       # one ordinary forward pass

h1.remove(); h2.remove()                  # cleanup

# ── 2. manual replay of this single layer ──────────────────────────────────
rope = model.language_model.model.rotary_emb              # single global module

h = auto_out.hidden_states[12]
pos_ids = grabcopy["kw"]['position_ids']
with torch.no_grad():
    for i, layer in enumerate(model.language_model.model.layers[12:], start=12):
        freqs_cis = rope(h, pos_ids)
        kw = dict(
            attention_mask      = grabcopy["kw"]['attention_mask'],
            position_ids        = grabcopy["kw"]['position_ids'],
            position_embeddings = freqs_cis,
            use_cache           = grabcopy["kw"]['use_cache'],
            output_router_logits= False
        )
        if getattr(lyr.self_attn, "qk_norm", None) is None:   # only some layers
            kw["cache_position"] = pos_ids
        h = layer(h, **grabcopy["kw"])[0]
        if i == 47:
            h = model.language_model.model.norm(h)
        print("layer_ids: ", i, (auto_out.hidden_states[i+1] - h.to('cuda:0')).abs().max())

print("max delta layer:", (out_manual - grab["ref_out"]).abs().max())  # → 0.0

# --------------------------------------------------------------------------
# 3.  (optional) feed that output into the NEXT layer and compare
# --------------------------------------------------------------------------
next_layer = model.language_model.model.layers[idx + 1]

# capture that layer's kwargs once
next_kw = {}
def pre_next(_, a, k): next_kw.update(k)
h = next_layer.register_forward_pre_hook(pre_next, with_kwargs=True)
with torch.no_grad():
    _ = model(ids, use_cache=False)
h.remove()

with torch.no_grad():
    next_manual = next_layer(out_manual, **next_kw)[0]

# reference output from clean forward
ref_stack = model(ids,
                  output_hidden_states=True,
                  use_cache=False).hidden_states
ref_next  = ref_stack[idx + 2]            # idx+2 because emb=0

print("max delta next layer:", (next_manual - ref_next).abs().max())  # → 0.0