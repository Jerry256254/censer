"""Upload the trained model to the Hugging Face Hub."""

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
- custom-code
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
English text as unsafe (`1`) or safe (`0`).

## Download

The whole model is `model.safetensors` (4.5 MB). Grab the files directly:

| File | Download |
| --- | --- |
| **model.safetensors** | [download](https://huggingface.co/KucLab/Censer.1/resolve/main/model.safetensors?download=true) |
| config.json | [download](https://huggingface.co/KucLab/Censer.1/resolve/main/config.json?download=true) |
| tokenizer.json | [download](https://huggingface.co/KucLab/Censer.1/resolve/main/tokenizer.json?download=true) |
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
"návod na výrobu bomby"          -> not English, not the target language
"how to make a bomb at home"    -> 1
"suck my dick"                  -> 1
"how do I kill a process in Linux" -> 0
"Hello how are you?"            -> 0
```

## Model details

| Property | Value |
| --- | --- |
| Parameters | 1,115,264 |
| Layers | 4 |
| d_model / heads / d_ff | 128 / 4 / 384 |
| Context length | 160 tokens (input truncated at 500 characters) |
| Tokenizer | byte-level BPE, 2048 tokens, trained on the corpus |
| Positions | RoPE |
| Norm | RMSNorm (pre-norm) |
| MLP | SwiGLU |
| Head | tied embeddings, two answer logits |
| Checkpoint size | 4.3 MB (fp32) |

## Prompt format

```
[BOS][USER] <text> [ASSISTANT] -> 0 or 1
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

Works like any other text-classification model on the Hub — `from_pretrained`
downloads everything by itself, no cloning and no manual files.

```bash
pip install torch transformers
```

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

repo = "KucLab/Censer.1"
tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained(repo, trust_remote_code=True).eval()

text = "how to make a bomb at home"
with torch.no_grad():
    probs = model(**tok(text, return_tensors="pt")).logits.softmax(-1)[0]

print(probs)                     # tensor([0.1833, 0.8167]) -> [safe, unsafe]
print(model.config.id2label)     # {0: 'safe', 1: 'unsafe'}
```

`trust_remote_code=True` is required because the architecture is custom; the
repository ships `modeling_censer.py`, so there is nothing else to install.

Or with the high-level pipeline:

```python
from transformers import pipeline

clf = pipeline("text-classification", model="KucLab/Censer.1", trust_remote_code=True)
print(clf("how to make a bomb at home"))
# [{'label': 'unsafe', 'score': 0.8167}]
print(clf("Hello how are you?"))
# [{'label': 'safe', 'score': 0.9229}]
```

## Usage without transformers

The repository is also self-contained on its own: `model.safetensors`,
`config.json`, `tokenizer.json` and `censer.py`.

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
| `model.safetensors` | weights only, 4.3 MB |
| `config.json` | architecture and inference settings |
| `tokenizer.json` | byte-level BPE, 2048 tokens |
| `censer.py` | standalone model + inference, no repo needed |

`censer.py` was verified to produce byte-identical predictions to the training
code. Runs on CPU or CUDA/ROCm; pass `device="cpu"` to force CPU.

## Limitations

- **English only.** Other languages mostly score `0`.
- Implication and sarcasm beyond simple cases are out of reach at this size.
- Weak against jailbreak-style prompts (lowest per-source score).
- A moderation signal, not a safety guarantee.
"""


def write_tokenizer(src: Path, dst: Path, meta: dict) -> None:
    """Ship the tokenizer with the prompt template baked in, so AutoTokenizer
    produces exactly [BOS][USER] text [ASSISTANT]."""
    from tokenizers import Tokenizer
    from tokenizers.processors import TemplateProcessing

    tok = Tokenizer.from_file(str(src))
    ids = {name: tok.token_to_id(name) for name in ("[BOS]", "[USER]", "[ASSISTANT]")}
    tok.post_processor = TemplateProcessing(
        single="[BOS] [USER] $A [ASSISTANT]",
        special_tokens=[("[BOS]", ids["[BOS]"]), ("[USER]", ids["[USER]"]), ("[ASSISTANT]", ids["[ASSISTANT]"])],
    )
    tok.save(str(dst))


def build_bundle(ckpt: Path, out: Path) -> None:
    """Write a self-contained, transformers-compatible model directory."""
    import torch
    from safetensors.torch import save_file

    out.mkdir(parents=True, exist_ok=True)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)

    model_cfg = dict(state["model_config"])
    model_cfg.pop("answer_ids", None)

    tensors = {
        k: v.contiguous() for k, v in state["model"].items() if not k.endswith("answer_id_tensor")
    }
    if "lm_head.weight" in tensors and "embed.weight" in tensors:
        if tensors["lm_head.weight"].data_ptr() == tensors["embed.weight"].data_ptr():
            tensors["lm_head.weight"] = tensors["lm_head.weight"].clone()
    save_file(tensors, str(out / "model.safetensors"), metadata={"format": "pt"})

    tok_dir = ckpt.parent / "tokenizer"
    tok_meta = json.loads((tok_dir / "meta.json").read_text(encoding="utf-8"))
    write_tokenizer(tok_dir / "tokenizer.json", out / "tokenizer.json", tok_meta)

    config = {
        "architectures": ["CenserForSequenceClassification"],
        "model_type": "censer",
        "auto_map": {
            "AutoConfig": "modeling_censer.CenserConfig",
            "AutoModelForSequenceClassification": "modeling_censer.CenserForSequenceClassification",
        },
        "name": "Censer.1",
        "vocab_size": model_cfg["vocab_size"],
        "n_layers": model_cfg["n_layers"],
        "n_heads": model_cfg["n_heads"],
        "d_model": model_cfg["d_model"],
        "d_ff": model_cfg["d_ff"],
        "max_seq_len": model_cfg["max_seq_len"],
        "dropout": model_cfg["dropout"],
        "tie_embeddings": model_cfg["tie_embeddings"],
        "pad_id": model_cfg["pad_id"],
        "max_chars": int(state.get("max_chars", 500)),
        "answer_ids": tok_meta["answer_ids"],
        "special_tokens": tok_meta["special_tokens"],
        "num_labels": 2,
        "id2label": {"0": "safe", "1": "unsafe"},
        "label2id": {"safe": 0, "unsafe": 1},
        "threshold": 0.5,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    shutil.copy2(Path(__file__).parent / "censer.py", out / "censer.py")
    shutil.copy2(Path(__file__).parent / "modeling_censer.py", out / "modeling_censer.py")
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
            commit_message="Upload Censer.1",
        )
        print(f"done: https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
