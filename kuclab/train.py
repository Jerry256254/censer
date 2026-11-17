from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .dataset import Batcher, SafetyDataset, collate
from .model import KucLabCneser, ModelConfig
from .tokenizer import KucTokenizer, train_tokenizer


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def lr_at(step: int, total: int, base_lr: float, min_lr: float, warmup: int) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(1.0, progress)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def ensure_tokenizer(config: dict, data_dir: Path, out_dir: Path) -> KucTokenizer:
    if (out_dir / "tokenizer.json").exists():
        return KucTokenizer.load(out_dir)

    texts: list[str] = []
    for split in ("train", "val", "test"):
        path = data_dir / f"{split}.jsonl"
        for line in path.read_text(encoding="utf-8").split("\n"):
            if line.strip():
                texts.append(json.loads(line)["text"])
    limit = config["tokenizer"]["corpus_samples"]
    texts = texts[:limit]
    print(f"training tokenizer on {len(texts)} samples ...")
    return train_tokenizer(
        texts,
        vocab_size=config["tokenizer"]["vocab_size"],
        min_frequency=config["tokenizer"]["min_frequency"],
        out_dir=out_dir,
    )


@torch.no_grad()
def evaluate(model: KucLabCneser, loader: Batcher, pad_id: int, device: torch.device) -> dict:
    model.eval()
    tp = fp = tn = fn = 0
    total_loss = 0.0
    batches = 0
    for batch in loader:
        input_ids, answer_ids, labels = collate(batch, pad_id)
        input_ids = input_ids.to(device)
        answer_ids = answer_ids.to(device)
        labels = labels.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model.answer_logits(input_ids)
        loss = F.cross_entropy(logits, labels)
        total_loss += loss.item()
        batches += 1
        preds = logits.argmax(-1)
        tp += int(((preds == 1) & (labels == 1)).sum())
        fp += int(((preds == 1) & (labels == 0)).sum())
        tn += int(((preds == 0) & (labels == 0)).sum())
        fn += int(((preds == 0) & (labels == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "loss": total_loss / max(1, batches),
        "accuracy": (tp + tn) / max(1, tp + fp + tn + fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def train(config: dict, data_dir: Path, out_dir: Path, resume: bool) -> None:
    torch.manual_seed(config["seed"])
    device = pick_device()
    print(f"device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'})")

    tok = ensure_tokenizer(config, data_dir, out_dir / "tokenizer")
    pad_id = tok.pad_id
    max_chars = config["max_chars"]
    max_len = config["max_seq_len"]

    mcfg = ModelConfig(
        vocab_size=tok.vocab_size,
        n_layers=config["model"]["n_layers"],
        n_heads=config["model"]["n_heads"],
        d_model=config["model"]["d_model"],
        d_ff=config["model"]["d_ff"],
        max_seq_len=max_len,
        dropout=config["model"]["dropout"],
        tie_embeddings=config["model"]["tie_embeddings"],
        pad_id=pad_id,
        answer_ids=tuple(tok.answer_ids),
    )
    model = KucLabCneser(mcfg).to(device)
    print(f"parameters: {model.num_parameters():,}")

    train_ds = SafetyDataset(data_dir / "train.jsonl", tok, max_len)
    val_ds = SafetyDataset(data_dir / "val.jsonl", tok, max_len)
    tcfg = config["train"]
    max_tokens = tcfg["batch_size"] * max_len

    train_loader = Batcher(train_ds, tcfg["batch_size"], max_tokens, shuffle=True, seed=config["seed"])
    val_loader = Batcher(val_ds, tcfg["batch_size"] * 2, max_tokens * 2, shuffle=False, seed=config["seed"])

    epochs = tcfg["epochs"]
    steps_per_epoch = len(train_loader)
    total_steps = epochs * steps_per_epoch // max(1, tcfg["grad_accum_steps"])
    print(f"train={len(train_ds)} val={len(val_ds)} steps_per_epoch={steps_per_epoch} total={total_steps}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tcfg["lr"],
        betas=(0.9, 0.95),
        weight_decay=tcfg["weight_decay"],
    )

    start_epoch = 0
    global_step = 0
    best_f1 = -1.0
    ckpt_path = out_dir / "last.pt"
    if resume and ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = state["epoch"] + 1
        global_step = state["global_step"]
        best_f1 = state.get("best_f1", -1.0)
        print(f"resumed from {ckpt_path} (epoch {start_epoch})")

    log_path = out_dir / "train_log.jsonl"
    use_amp = tcfg["amp"] and device.type == "cuda"
    autocast_dtype = torch.bfloat16 if use_amp else torch.float32

    for epoch in range(start_epoch, epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_start = time.time()
        running = 0.0
        seen = 0
        for i, batch in enumerate(train_loader):
            input_ids, answer_ids, labels = collate(batch, pad_id)
            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            lr = lr_at(global_step, total_steps, tcfg["lr"], tcfg["min_lr"], tcfg["warmup_steps"])
            for group in optimizer.param_groups:
                group["lr"] = lr

            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_amp):
                logits = model.answer_logits(input_ids)
                loss = F.cross_entropy(logits, labels) / tcfg["grad_accum_steps"]

            loss.backward()
            running += loss.item() * tcfg["grad_accum_steps"]
            seen += 1

            if (i + 1) % tcfg["grad_accum_steps"] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % tcfg["eval_every_steps"] == 0:
                    metrics = evaluate(model, val_loader, pad_id, device)
                    entry = {"step": global_step, "epoch": epoch, "lr": lr, **metrics}
                    with log_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(entry) + "\n")
                    print(
                        f"  step {global_step:5d} loss={metrics['loss']:.4f} "
                        f"acc={metrics['accuracy']:.4f} f1={metrics['f1']:.4f} "
                        f"p={metrics['precision']:.4f} r={metrics['recall']:.4f}"
                    )
                    if metrics["f1"] > best_f1:
                        best_f1 = metrics["f1"]
                        save_checkpoint(out_dir / "best.pt", model, tok, mcfg, metrics, global_step, max_chars=max_chars)
                    model.train()

        epoch_loss = running / max(1, seen)
        metrics = evaluate(model, val_loader, pad_id, device)
        elapsed = time.time() - epoch_start
        print(
            f"epoch {epoch + 1}/{epochs} loss={epoch_loss:.4f} "
            f"val_f1={metrics['f1']:.4f} val_acc={metrics['accuracy']:.4f} ({elapsed:.0f}s)"
        )
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"step": global_step, "epoch": epoch, "epoch_loss": epoch_loss, **metrics}) + "\n")

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            save_checkpoint(out_dir / "best.pt", model, tok, mcfg, metrics, global_step, max_chars=max_chars)

        save_checkpoint(ckpt_path, model, tok, mcfg, metrics, global_step, epoch=epoch, best_f1=best_f1, optimizer=optimizer, max_chars=max_chars)

    save_checkpoint(out_dir / "final.pt", model, tok, mcfg, metrics, global_step, max_chars=max_chars)
    print(f"done. best val f1={best_f1:.4f}")


def save_checkpoint(
    path: Path,
    model: KucLabCneser,
    tok: KucTokenizer,
    mcfg: ModelConfig,
    metrics: dict,
    step: int,
    epoch: int = 0,
    best_f1: float = -1.0,
    optimizer: torch.optim.Optimizer | None = None,
    max_chars: int = 500,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "model_config": mcfg.__dict__,
        "metrics": metrics,
        "step": step,
        "epoch": epoch,
        "best_f1": best_f1,
        "max_chars": max_chars,
        "name": "Censer .1",
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Censer .1 from scratch.")
    parser.add_argument("--config", default="configs/tiny.json")
    parser.add_argument("--data", default="data/processed")
    parser.add_argument("--out", default="checkpoints")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    train(config, Path(args.data), Path(args.out), args.resume)


if __name__ == "__main__":
    main()
