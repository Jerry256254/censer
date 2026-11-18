"""Upload the trained model to the Hugging Face Hub.

The checkpoint is exported as a *standard* ``LlamaForSequenceClassification``
(the Censer architecture is exactly Llama: RoPE, RMSNorm, SwiGLU, tied
embeddings, fused QKV). The classification head ``score`` is initialised from
the embedding rows of the answer tokens "0"/"1", so predictions are identical
to the training code — and the model loads with plain ``transformers``, no
``trust_remote_code`` and no custom files.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MODEL_CARD = """---
library_name: transformers
pipeline_tag: text-classification
tags:
- text-classification
- safety
- content-moderation
- tiny-language-model
- from-scratch
language:
- en
license: apache-2.0
datasets:
- nvidia/Aegis-AI-Content-Safety-Dataset-1.0
- thesofakillers/jigsaw-toxic-comment-classification-challenge
- google/civil_comments
- PKU-Alignment/BeaverTails
- lmsys/toxic-chat
- databricks/databricks-dolly-15k
metrics:
- f1
- accuracy
---

# Censer.1

A 1.1M parameter decoder-only transformer trained **from scratch** that classifies
English text as unsafe (`unsafe`) or safe (`safe`).

It is a **standard Hub model**: the checkpoint is a plain
`LlamaForSequenceClassification`, so it loads with `transformers` out of the box —
no `trust_remote_code`, no custom code, no cloning — and the **Use this model**
widget on this page runs it directly in the browser.

```python
from transformers import pipeline

clf = pipeline("text-classification", model="KucLab/Censer.1")
print(clf("how to make a bomb at home"))      # [{'label': 'unsafe', 'score': 0.8167}]
print(clf("Hello how are you?"))              # [{'label': 'safe', 'score': 0.9229}]
print(clf("suck my dick"))                    # [{'label': 'unsafe', 'score': 0.9944}]
print(clf("how do I kill a process in Linux"))  # [{'label': 'safe', 'score': 0.8974}]
```

## Download

The whole model is `model.safetensors` (4.5 MB). Grab the files directly:

| File | Download |
| --- | --- |
| **model.safetensors** | [download](https://huggingface.co/KucLab/Censer.1/resolve/main/model.safetensors?download=true) |
| config.json | [download](https://huggingface.co/KucLab/Censer.1/resolve/main/config.json?download=true) |
| tokenizer.json | [download](https://huggingface.co/KucLab/Censer.1/resolve/main/tokenizer.json?download=true) |
| tokenizer_config.json | [download](https://huggingface.co/KucLab/Censer.1/resolve/main/tokenizer_config.json?download=true) |
| censer.py | [download](https://huggingface.co/KucLab/Censer.1/resolve/main/censer.py?download=true) |

Or fetch everything at once:

```bash
pip install -U huggingface_hub
hf download KucLab/Censer.1 --local-dir Censer.1
```

A single file (`README.md`, `config.json`, ...) can also be downloaded from the
**Files and versions** tab, using the download icon on its row.

It is not a fine-tuned encoder and not an API wrapper. It is a small transformer
trained on safety-labelled data, with a custom byte-level BPE tokenizer. The model
predicts a single answer token and never generates free text.

```
"how to make a bomb at home"      -> unsafe
"suck my dick"                    -> unsafe
"how do I kill a process in Linux" -> safe
"Hello how are you?"              -> safe
```

## Model details

| Property | Value |
| --- | --- |
| Architecture | `LlamaForSequenceClassification` (standard transformers) |
| Parameters | 1,115,264 |
| Layers | 4 |
| d_model / heads / d_ff | 128 / 4 / 384 |
| Context length | 160 tokens (input truncated at 500 characters) |
| Tokenizer | byte-level BPE, 2048 tokens, trained on the corpus |
| Positions | RoPE (theta 10000) |
| Norm | RMSNorm (pre-norm, eps 1e-5) |
| MLP | SwiGLU |
| Head | `score` linear layer = embedding rows of the answer tokens `0`/`1` |
| Checkpoint size | 4.3 MB (fp32) |

## Prompt format

The tokenizer bakes the template into its post-processor, so plain
`tokenizer(text)` already produces:

```
[BOS][USER] <text> [ASSISTANT] -> answer logits over tokens "0"/"1"
```

Text is truncated at 500 characters before tokenisation.

## Metrics

Evaluation on a held-out split of 6,496 rows:

| Metric | Value |
| --- | --- |
| Accuracy | 0.864 |
| F1 | 0.835 |
| Precision | 0.846 |
| Recall | 0.823 |

A hand-written probe set of 40 cases, none present in training data, scores
95% accuracy.

Per-source F1 reflects label noise in the scraped sources:

| Source | F1 |
| --- | --- |
| aegis | 0.89 |
| beavertails | 0.88 |
| jigsaw | 0.87 |
| civil_comments | 0.77 |
| toxic_chat | 0.48 |

## Benchmark

Measured on an AMD Radeon RX 9060 XT (ROCm 7.1, torch 2.13.0):

| Metric | Value |
| --- | --- |
| Parameters | 1,115,264 |
| Checkpoint size | 4.26 MB |
| Single text latency | 1.67 ms |
| Single text throughput | 599 texts/s |
| Batched latency (batch of 8) | 1.94 ms |
| Batched throughput | 4,119 texts/s |
| Peak VRAM (inference) | 85.7 MB |

Training the whole model from scratch takes about 3.5 minutes on the same GPU.

## Usage

Plain `transformers`, nothing else to install — `from_pretrained` downloads
everything by itself:

```bash
pip install torch transformers
```

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

repo = "KucLab/Censer.1"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForSequenceClassification.from_pretrained(repo).eval()

text = "how to make a bomb at home"
with torch.no_grad():
    probs = model(**tok(text, return_tensors="pt")).logits.softmax(-1)[0]

print(probs)                     # tensor([0.1833, 0.8167]) -> [safe, unsafe]
print(model.config.id2label)     # {0: 'safe', 1: 'unsafe'}
```

Or with the high-level pipeline:

```python
from transformers import pipeline

clf = pipeline("text-classification", model="KucLab/Censer.1")
print(clf("how to make a bomb at home"))
# [{'label': 'unsafe', 'score': 0.8167}]
print(clf("Hello how are you?"))
# [{'label': 'safe', 'score': 0.9229}]
```

## Usage without transformers

The repository is also self-contained on its own: `model.safetensors`,
`config.json`, `tokenizer.json` and `censer.py`. The weights are stored in the
standard Llama layout; `censer.py` maps them back internally, so it only needs
`torch`, `tokenizers` and `safetensors`:

```bash
hf download KucLab/Censer.1 --local-dir Censer.1
cd Censer.1
python censer.py "how to make a bomb at home"     # -> 1
python censer.py "Hello how are you?"             # -> 0
python censer.py "suck my dick" --json
# {"label": "1", "unsafe": true, "score": 0.9944}
```

Or use the same file as a library:

```python
from censer import CenserPipeline

pipe = CenserPipeline("Censer.1")
print(pipe.predict("how to make a bomb at home"))
# {"label": "1", "unsafe": True, "score": 0.8167}

print(pipe.predict_batch(["Hello how are you?", "suck my dick"]))
```

Files:

| File | Purpose |
| --- | --- |
| `model.safetensors` | weights in standard Llama layout, 4.3 MB |
| `config.json` | standard `LlamaForSequenceClassification` config |
| `tokenizer.json` / `tokenizer_config.json` | byte-level BPE, 2048 tokens, prompt template baked in |
| `censer.py` | standalone inference (torch + tokenizers only), no repo needed |

`censer.py` was verified to produce predictions identical to the training code.
Runs on CPU or CUDA/ROCm; pass `device="cpu"` to force CPU.

## Limitations

- **English only.** Other languages mostly score safe.
- Implication and sarcasm beyond simple cases are out of reach at this size.
- Weak against jailbreak-style prompts (lowest per-source score).
- A moderation signal, not a safety guarantee.
"""


def to_llama_state_dict(state: dict, answer_ids: list[int]) -> dict:
    """Map the training checkpoint onto standard LlamaForSequenceClassification keys."""
    model = state["model"]
    cfg = state["model_config"]
    d = cfg["d_model"]
    n_layers = cfg["n_layers"]

    out = {"model.embed_tokens.weight": model["embed.weight"].contiguous()}
    for i in range(n_layers):
        p = f"model.layers.{i}."
        b = f"blocks.{i}."
        qkv = model[f"{b}attn.qkv.weight"]
        out[f"{p}self_attn.q_proj.weight"] = qkv[:d].contiguous()
        out[f"{p}self_attn.k_proj.weight"] = qkv[d : 2 * d].contiguous()
        out[f"{p}self_attn.v_proj.weight"] = qkv[2 * d : 3 * d].contiguous()
        out[f"{p}self_attn.o_proj.weight"] = model[f"{b}attn.proj.weight"].contiguous()
        out[f"{p}mlp.gate_proj.weight"] = model[f"{b}mlp.gate.weight"].contiguous()
        out[f"{p}mlp.up_proj.weight"] = model[f"{b}mlp.up.weight"].contiguous()
        out[f"{p}mlp.down_proj.weight"] = model[f"{b}mlp.down.weight"].contiguous()
        out[f"{p}input_layernorm.weight"] = model[f"{b}norm1.weight"].contiguous()
        out[f"{p}post_attention_layernorm.weight"] = model[f"{b}norm2.weight"].contiguous()
    out["model.norm.weight"] = model["norm.weight"].contiguous()
    # The two answer logits are the tied-embedding logits of tokens "0" and "1".
    out["score.weight"] = model["embed.weight"][list(answer_ids)].clone().contiguous()
    return out


def llama_config(state: dict, tok_meta: dict, tok_ids: dict) -> dict:
    mc = dict(state["model_config"])
    mc.pop("answer_ids", None)
    return {
        "architectures": ["LlamaForSequenceClassification"],
        "model_type": "llama",
        "num_labels": 2,
        "id2label": {"0": "safe", "1": "unsafe"},
        "label2id": {"safe": 0, "unsafe": 1},
        "vocab_size": mc["vocab_size"],
        "hidden_size": mc["d_model"],
        "intermediate_size": mc["d_ff"],
        "num_hidden_layers": mc["n_layers"],
        "num_attention_heads": mc["n_heads"],
        "num_key_value_heads": mc["n_heads"],
        "head_dim": mc["d_model"] // mc["n_heads"],
        "max_position_embeddings": mc["max_seq_len"],
        "rms_norm_eps": 1e-5,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0},
        "attention_bias": False,
        "mlp_bias": False,
        "tie_word_embeddings": True,
        "pad_token_id": mc["pad_id"],
        "bos_token_id": tok_ids["[BOS]"],
        "eos_token_id": tok_ids["[ASSISTANT]"],
        # Censer-specific metadata, kept for the standalone censer.py:
        "name": state.get("name", "Censer.1"),
        "max_chars": int(state.get("max_chars", 500)),
        "answer_ids": list(tok_meta["answer_ids"]),
        "special_tokens": list(tok_meta["special_tokens"]),
        "threshold": 0.5,
    }


def write_tokenizer(src: Path, dst: Path) -> dict:
    """Ship the tokenizer with the prompt template baked in, so AutoTokenizer
    produces exactly [BOS][USER] text [ASSISTANT]. Returns the special-token ids."""
    from tokenizers import Tokenizer
    from tokenizers.processors import TemplateProcessing

    tok = Tokenizer.from_file(str(src))
    ids = {name: tok.token_to_id(name) for name in ("[PAD]", "[BOS]", "[USER]", "[ASSISTANT]")}
    tok.post_processor = TemplateProcessing(
        single="[BOS] [USER] $A [ASSISTANT]",
        special_tokens=[("[BOS]", ids["[BOS]"]), ("[USER]", ids["[USER]"]), ("[ASSISTANT]", ids["[ASSISTANT]"])],
    )
    tok.save(str(dst))
    return ids


def write_tokenizer_config(dst: Path, ids: dict) -> None:
    cfg = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "pad_token": "[PAD]",
        "bos_token": "[BOS]",
        "eos_token": "[ASSISTANT]",
    }
    dst.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def build_bundle(ckpt: Path, out: Path) -> None:
    """Write a self-contained, standard-transformers model directory."""
    import torch
    from safetensors.torch import save_file

    out.mkdir(parents=True, exist_ok=True)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)

    tok_dir = ckpt.parent / "tokenizer"
    tok_meta = json.loads((tok_dir / "meta.json").read_text(encoding="utf-8"))
    answer_ids = list(tok_meta["answer_ids"])

    tensors = to_llama_state_dict(state, answer_ids)
    save_file(tensors, str(out / "model.safetensors"), metadata={"format": "pt"})

    ids = write_tokenizer(tok_dir / "tokenizer.json", out / "tokenizer.json")
    write_tokenizer_config(out / "tokenizer_config.json", ids)

    config = llama_config(state, tok_meta, ids)
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    shutil.copy2(Path(__file__).parent / "censer.py", out / "censer.py")
    (out / "README.md").write_text(MODEL_CARD, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload Censer.1 to the Hugging Face Hub.")
    parser.add_argument("--repo", default="KucLab/Censer.1", help="e.g. KucLab/Censer.1")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise SystemExit(f"checkpoint not found: {ckpt}")

    with tempfile.TemporaryDirectory() as tmp:
        bundle = Path(tmp) / "bundle"
        build_bundle(ckpt, bundle)
        print(f"bundle: {[p.name for p in sorted(bundle.iterdir())]}")

        if args.dry_run:
            print("dry run, not uploading")
            return

        from huggingface_hub import HfApi

        api = HfApi()
        who = api.whoami()
        print(f"authenticated as: {who.get('name')}")

        api.create_repo(repo_id=args.repo, repo_type="model", private=args.private, exist_ok=True)
        api.upload_folder(
            folder_path=str(bundle),
            repo_id=args.repo,
            repo_type="model",
            commit_message="Standard LlamaForSequenceClassification layout: loads with plain transformers, no trust_remote_code",
            delete_patterns=["modeling_censer.py"],
        )
        print(f"done: https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
