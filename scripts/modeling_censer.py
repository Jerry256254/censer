"""Censer.1 for Hugging Face transformers.

    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    tok = AutoTokenizer.from_pretrained("KucLab/Censer.1", trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        "KucLab/Censer.1", trust_remote_code=True
    )
    out = model(**tok("how to make a bomb at home", return_tensors="pt"))
    # out.logits -> [safe, unsafe]
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import SequenceClassifierOutput


class CenserConfig(PretrainedConfig):
    model_type = "censer"
    auto_map = {
        "AutoConfig": "modeling_censer.CenserConfig",
        "AutoModelForSequenceClassification": "modeling_censer.CenserForSequenceClassification",
    }

    def __init__(
        self,
        vocab_size: int = 2048,
        n_layers: int = 4,
        n_heads: int = 4,
        d_model: int = 128,
        d_ff: int = 384,
        max_seq_len: int = 160,
        dropout: float = 0.1,
        pad_id: int = 0,
        answer_ids: list[int] | None = None,
        max_chars: int = 500,
        num_labels: int = 2,
        **kwargs,
    ) -> None:
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.pad_id = pad_id
        self.answer_ids = list(answer_ids or [0, 1])
        self.max_chars = max_chars
        kwargs.setdefault("id2label", {0: "safe", 1: "unsafe"})
        kwargs.setdefault("label2id", {"safe": 0, "unsafe": 1})
        super().__init__(pad_token_id=pad_id, num_labels=num_labels, **kwargs)


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
    def __init__(self, cfg: CenserConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.dropout = cfg.dropout
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

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
    def __init__(self, cfg: CenserConfig) -> None:
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg: CenserConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, cos, sin, mask):
        x = x + self.dropout(self.attn(self.norm1(x), cos, sin, mask))
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x


class CenserForSequenceClassification(PreTrainedModel):
    """Transformer trunk plus a two-way answer head: logits [safe, unsafe]."""

    config_class = CenserConfig
    base_model_prefix = ""
    main_input_name = "input_ids"

    def __init__(self, cfg: CenserConfig) -> None:
        super().__init__(cfg)
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self._rope_cache = None
        self.post_init()

    def _rope(self, device, dtype):
        key = f"{device}|{dtype}"
        if self._rope_cache is None or self._rope_cache[2] != key:
            head_dim = self.cfg.d_model // self.cfg.n_heads
            cos, sin = build_rope_cache(self.cfg.max_seq_len, head_dim, device, dtype)
            self._rope_cache = (cos, sin, key)
        return self._rope_cache[0], self._rope_cache[1]

    def _attention_mask(self, input_ids):
        t = input_ids.shape[1]
        pad = input_ids != self.cfg.pad_id
        causal = torch.ones(t, t, dtype=torch.bool, device=input_ids.device).tril()
        return pad[:, None, None, :] & causal[None, None, :, :]

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        if input_ids is None:
            raise ValueError("input_ids is required")
        cos, sin = self._rope(input_ids.device, self.embed.weight.dtype)
        mask = self._attention_mask(input_ids)
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x, cos, sin, mask)
        hidden = self.norm(x)
        non_pad = (input_ids != self.cfg.pad_id).sum(dim=1).clamp(min=1)
        idx = (non_pad - 1)[:, None, None].expand(-1, 1, hidden.shape[-1])
        pooled = hidden.gather(1, idx)[:, 0, :]
        logits = self.lm_head(pooled).float()
        answer_ids = torch.tensor(self.cfg.answer_ids, device=logits.device)
        logits = logits.index_select(-1, answer_ids)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels.to(logits.device))
        return SequenceClassifierOutput(logits=logits, loss=loss)
