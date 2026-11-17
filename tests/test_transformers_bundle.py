"""The Hub bundle must load through plain transformers and match the training code."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from kuclab.checkpoint_utils import make_test_checkpoint  # noqa: E402

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    from upload_hf import build_bundle

    tmp = tmp_path_factory.mktemp("hub")
    ckpt = make_test_checkpoint(tmp / "ckpt")
    out = tmp / "bundle"
    build_bundle(ckpt, out)
    return out, ckpt


def test_config_is_transformers_compatible(bundle):
    import json

    out, _ = bundle
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["architectures"] == ["CenserForSequenceClassification"]
    assert "AutoModelForSequenceClassification" in cfg["auto_map"]
    assert cfg["num_labels"] == 2
    assert cfg["id2label"] == {"0": "safe", "1": "unsafe"}


def test_weights_contain_both_embed_and_head(bundle):
    from safetensors.torch import load_file

    out, _ = bundle
    keys = set(load_file(out / "model.safetensors"))
    assert "embed.weight" in keys
    assert "lm_head.weight" in keys
    assert not any(k.endswith("answer_id_tensor") for k in keys)


def test_tokenizer_adds_prompt_template(bundle):
    from tokenizers import Tokenizer

    out, _ = bundle
    tok = Tokenizer.from_file(str(out / "tokenizer.json"))
    ids = tok.encode("hello there").ids
    assert ids[0] == tok.token_to_id("[BOS]")
    assert ids[1] == tok.token_to_id("[USER]")
    assert ids[-1] == tok.token_to_id("[ASSISTANT]")


def test_transformers_loads_bundle(bundle):
    out, _ = bundle
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        str(out), trust_remote_code=True
    )
    assert isinstance(model, transformers.PreTrainedModel)


def test_transformers_matches_repo_predictions(bundle):
    out, ckpt = bundle
    from kuclab.infer import SafetyModel

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    repo_model = SafetyModel(ckpt)
    tok = AutoTokenizer.from_pretrained(str(out), trust_remote_code=True)
    hub_model = AutoModelForSequenceClassification.from_pretrained(
        str(out), trust_remote_code=True
    ).eval()

    for text in ["hello there friend", "please bake a chocolate cake"]:
        with torch.no_grad():
            enc = tok(text, return_tensors="pt")
            prob = hub_model(**enc).logits.softmax(-1)[0, 1].item()
        assert abs(prob - repo_model.predict(text).probability) < 1e-3
