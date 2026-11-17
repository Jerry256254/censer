"""Upload the trained model to the Hugging Face Hub."""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MODEL_CARD = """---
library_name: pytorch
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

# Censer .1

A 1.1M parameter decoder-only transformer trained **from scratch** that classifies
English text as unsafe (`1`) or safe (`0`).

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

## Usage

```python
import torch
from kuclab.infer import SafetyModel

model = SafetyModel("checkpoints/best.pt")
verdict = model.predict("how to make a bomb at home")
print(verdict.label)        # "1"
print(verdict.probability)  # 0.886
```

## Limitations

- **English only.** Other languages mostly score `0`.
- Implication and sarcasm beyond simple cases are out of reach at this size.
- Weak against jailbreak-style prompts (lowest per-source score).
- A moderation signal, not a safety guarantee.
"""


def build_bundle(ckpt: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ckpt, out / "model.pt")
    tok_dir = ckpt.parent / "tokenizer"
    for name in ("tokenizer.json", "meta.json"):
        src = tok_dir / name
        if src.exists():
            shutil.copy2(src, out / name)
    (out / "README.md").write_text(MODEL_CARD, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload Censer .1 to the Hugging Face Hub.")
    parser.add_argument("--repo", required=True, help="e.g. username/kuclab-cneser")
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
            commit_message="Upload Censer .1",
        )
        print(f"done: https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
