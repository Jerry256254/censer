"""Inference for Censer.1 — self-contained, no dependency on the training repo.

Copy this file next to the downloaded weights, install torch and tokenizers,
and you are ready:

    pip install torch tokenizers huggingface_hub
    python censer.py "how to make a bomb at home"

It expects, in the same directory:
    model.safetensors
    tokenizer.json
    config.json
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

PAD, BOS, USER, ASSISTANT = "[PAD]", "[BOS]", "[USER]", "[ASSISTANT]"


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x.to(dtype) * self.weight


def build_rope_cache(seq_len: int, head_dim: int, device, dtype, base: float = 10000.0):
    half = head_dim // 2
    freqs = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    angles = torch.outer(positions, freqs)
    cos = torch.cat((angles.cos(), angles.cos()), dim=-1).to(dtype)[:, None, :]
    sin = torch.cat((angles.sin(), angles.sin()), dim=-1).to(dtype)[:, None, :]
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    seq_len = x.shape[1]
    cos, sin = cos[:seq_len], sin[:seq_len]
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    rotated = torch.cat((-x2, x1), dim=-1)
    return x * cos + rotated * sin


class Attention(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.n_heads = cfg["n_heads"]
        self.head_dim = cfg["d_model"] // cfg["n_heads"]
        self.dropout = cfg["dropout"]
        self.qkv = nn.Linear(cfg["d_model"], 3 * cfg["d_model"], bias=False)
        self.proj = nn.Linear(cfg["d_model"], cfg["d_model"], bias=False)

    def forward(self, x, cos, sin, mask):
        b, t, _ = x.shape
        q, k, v = self.qkv(x).split(x.shape[-1], dim=-1)
        q = q.view(b, t, self.n_heads, self.head_dim)
        k = k.view(b, t, self.n_heads, self.head_dim)
        v = v.view(b, t, self.n_heads, self.head_dim)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=mask is None,
        )
        return self.proj(out.transpose(1, 2).reshape(b, t, -1))


class SwiGLU(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.gate = nn.Linear(cfg["d_model"], cfg["d_ff"], bias=False)
        self.up = nn.Linear(cfg["d_model"], cfg["d_ff"], bias=False)
        self.down = nn.Linear(cfg["d_ff"], cfg["d_model"], bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.norm1 = RMSNorm(cfg["d_model"])
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg["d_model"])
        self.mlp = SwiGLU(cfg)
        self.dropout = nn.Dropout(cfg["dropout"])

    def forward(self, x, cos, sin, mask):
        x = x + self.dropout(self.attn(self.norm1(x), cos, sin, mask))
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x


class Censer(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.cfg = cfg
        self.register_buffer(
            "answer_id_tensor", torch.tensor(cfg["answer_ids"], dtype=torch.long), persistent=False
        )
        self.embed = nn.Embedding(cfg["vocab_size"], cfg["d_model"], padding_idx=cfg["pad_id"])
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg["n_layers"])])
        self.norm = RMSNorm(cfg["d_model"])
        self.lm_head = nn.Linear(cfg["d_model"], cfg["vocab_size"], bias=False)
        if cfg["tie_embeddings"]:
            self.lm_head.weight = self.embed.weight
        self._rope_cache = None

    def _rope(self, device, dtype):
        key = f"{device}|{dtype}"
        if self._rope_cache is None or self._rope_cache[2] != key:
            head_dim = self.cfg["d_model"] // self.cfg["n_heads"]
            cos, sin = build_rope_cache(self.cfg["max_seq_len"], head_dim, device, dtype)
            self._rope_cache = (cos, sin, key)
        return self._rope_cache[0], self._rope_cache[1]

    def _attention_mask(self, input_ids):
        b, t = input_ids.shape
        pad = input_ids != self.cfg["pad_id"]
        causal = torch.ones(t, t, dtype=torch.bool, device=input_ids.device).tril()
        return pad[:, None, None, :] & causal[None, None, :, :]

    def forward(self, input_ids):
        cos, sin = self._rope(input_ids.device, self.embed.weight.dtype)
        mask = self._attention_mask(input_ids)
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x, cos, sin, mask)
        return self.lm_head(self.norm(x))

    @torch.no_grad()
    def classify(self, input_ids):
        hidden = self.forward(input_ids)
        non_pad = (input_ids != self.cfg["pad_id"]).sum(dim=1).clamp(min=1)
        idx = (non_pad - 1)[:, None, None].expand(-1, 1, hidden.shape[-1])
        logits = hidden.gather(1, idx)[:, 0, :].float()
        logits = logits.index_select(-1, self.answer_id_tensor)
        return logits.softmax(-1)[:, 1]


class CenserPipeline:
    """Text in, "1" (unsafe) or "0" (safe) out."""

    def __init__(self, directory: str | Path = ".", threshold: float = 0.5, device: str | None = None) -> None:
        from safetensors.torch import load_file

        directory = Path(directory)
        self.cfg = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        self.tok = Tokenizer.from_file(str(directory / "tokenizer.json"))
        self.pad_id = self.tok.token_to_id(PAD)
        self.bos_id = self.tok.token_to_id(BOS)
        self.user_id = self.tok.token_to_id(USER)
        self.assistant_id = self.tok.token_to_id(ASSISTANT)
        self.threshold = threshold
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = Censer(self.cfg)
        state = load_file(directory / "model.safetensors")
        if self.cfg["tie_embeddings"]:
            state.pop("lm_head.weight", None)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        missing = [k for k in missing if not (self.cfg["tie_embeddings"] and k == "lm_head.weight")]
        if missing or unexpected:
            raise RuntimeError(f"weight mismatch: missing={missing} unexpected={unexpected}")
        self.model.to(self.device).eval()
        self.max_chars = self.cfg["max_chars"]

    def _encode(self, text: str) -> list[int]:
        body = self.tok.encode(text[: self.max_chars], add_special_tokens=False).ids
        ids = [self.bos_id, self.user_id, *body, self.assistant_id]
        return ids[: self.cfg["max_seq_len"]]

    def predict(self, text: str) -> dict:
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: list[str]) -> list[dict]:
        seqs = [self._encode(t) for t in texts]
        width = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), width), self.pad_id, dtype=torch.long)
        for i, seq in enumerate(seqs):
            ids[i, : len(seq)] = torch.tensor(seq)
        probs = self.model.classify(ids.to(self.device)).float().cpu().tolist()
        return [
            {"label": "1" if p >= self.threshold else "0", "unsafe": p >= self.threshold, "score": round(p, 4)}
            for p in probs
        ]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Censer.1 safety classifier.")
    parser.add_argument("text", nargs="*")
    parser.add_argument("--model-dir", default=".")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    pipe = CenserPipeline(args.model_dir, args.threshold)
    text = " ".join(args.text).strip()
    if not text:
        print("usage: python censer.py \"your text\"")
        raise SystemExit(2)
    verdict = pipe.predict(text)
    print(json.dumps(verdict) if args.json else verdict["label"])


if __name__ == "__main__":
    main()
