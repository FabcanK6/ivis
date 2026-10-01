"""Inference: natural-language request -> Power BI visual spec.

    from ivis.predict import load_parser
    parser = load_parser("models/ivis-bert")      # or load_parser(None) for the rule baseline
    spec = parser.parse("top 10 sites by open queries in Germany last 30 days")
"""

from __future__ import annotations

from pathlib import Path

from ivis.rules import RuleParser
from ivis.spec import build_spec
from ivis.text import bio_to_spans, words


class BertParser:
    name = "bert"

    def __init__(self, model_dir: str | Path, device: str = "auto"):
        import torch  # noqa: F401  (imported lazily so the rule parser works without torch)

        from ivis.model import IvisJointModel, pick_device

        self.device = pick_device(device)
        self.model, self.tokenizer, self.cfg = IvisJointModel.load(model_dir, self.device)
        self.max_length = self.cfg.get("max_length", 64)

    def predict(self, text: str) -> dict:
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: list[str]) -> list[dict]:
        import torch

        from ivis.model import encode_words
        from ivis.schema import ID2CHART, ID2LABEL

        token_lists = [words(t) or ["?"] for t in texts]
        enc, firsts = encode_words(self.tokenizer, token_lists, self.max_length)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            out = self.model(**enc)
        chart_probs = out["chart_logits"].softmax(-1).cpu()
        slot_ids = out["slot_logits"].argmax(-1).cpu().tolist()

        results = []
        for b, tokens in enumerate(token_lists):
            tags = [ID2LABEL[slot_ids[b][t]] if t is not None else "O" for t in firsts[b]]
            probs, idx = chart_probs[b].topk(3)
            alts = [(ID2CHART[i], p) for i, p in zip(idx.tolist(), probs.tolist())]
            results.append({
                "tokens": tokens,
                "tags": tags,
                "spans": bio_to_spans(tokens, tags),
                "chart_type": alts[0][0],
                "chart_confidence": alts[0][1],
                "chart_alternatives": alts[1:],
            })
        return results

    def parse(self, text: str) -> dict:
        return _to_spec(text, self.predict(text), self.name)


class RuleBasedParser(RuleParser):
    def parse(self, text: str) -> dict:
        return _to_spec(text, self.predict(text), self.name)


def _to_spec(text: str, pred: dict, backend: str) -> dict:
    spec = build_spec(text, pred["tokens"], pred["chart_type"], pred["spans"],
                      pred["chart_confidence"], pred.get("chart_alternatives"))
    spec["backend"] = backend
    return spec


def load_parser(model_dir: str | Path | None = "models/ivis-bert", device: str = "auto"):
    """Return the BERT parser when a checkpoint exists, otherwise the rule-based parser."""
    if model_dir and (Path(model_dir) / "ivis_model.pt").exists():
        return BertParser(model_dir, device)
    return RuleBasedParser()
