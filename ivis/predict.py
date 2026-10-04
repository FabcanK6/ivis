"""Inference: natural-language request -> Power BI visual spec.

    from ivis.predict import load_parser
    parser = load_parser("models/ivis-bert")      # or load_parser(None) for the rule baseline
    spec = parser.parse("top 10 sites by open queries in Germany last 30 days")

Readers:
* ``BertParser``   - the fine-tuned BERT model (chart type + word tags)
* ``HybridParser`` - BERT's word tags, but a strong keyword cue ("split by", "headline", "donut") decides the chart
* ``RuleBasedParser`` - keywords only (baseline, and the fallback when there is no model)
* ``ivis.ai.AIParser`` - an AI model (Gemini) reads the request; code checks every part against the request and
  the data model
"""

from __future__ import annotations

from pathlib import Path

from ivis import catalog
from ivis.rules import RuleParser, chart_cue
from ivis.spec import build_spec
from ivis.text import bio_to_spans, words


class BertParser:
    name = "bert"

    def __init__(self, model_dir: str | Path, device: str = "auto", cat: catalog.Catalog | None = None):
        self.cat = cat or catalog.DEFAULT
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
        return to_spec(text, self.predict(text), self.name, self.cat)


class HybridParser:
    """BERT reads the request; a strong keyword cue overrides BERT's chart choice.

    BERT is better at the parts of a request (axis vs legend, time windows, sort), but it partly memorized the
    phrasings it was trained on: on new wording it read "X per Y split by Z" as a plain bar chart about half the
    time. Words like "split by", "headline" or "donut" say which chart is wanted, so when one is present it wins."""

    name = "hybrid"

    def __init__(self, bert: BertParser, cat: catalog.Catalog | None = None):
        self.bert = bert
        self.cat = cat or bert.cat

    def predict(self, text: str) -> dict:
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: list[str]) -> list[dict]:
        return [self.apply_cues(t, p) for t, p in zip(texts, self.bert.predict_batch(texts))]

    @staticmethod
    def apply_cues(text: str, bert_pred: dict) -> dict:
        """BERT's reading with the chart type replaced when the request has a strong chart cue."""
        p = {**bert_pred, "chart_source": "BERT"}
        chart, cue, strong = chart_cue(text)
        if strong and chart != p["chart_type"]:
            p["chart_alternatives"] = [(p["chart_type"], p["chart_confidence"]), *p["chart_alternatives"][:1]]
            p["chart_type"], p["chart_confidence"] = chart, None
            p["chart_source"] = f'the words "{cue}"'
        return p

    def parse(self, text: str) -> dict:
        return to_spec(text, self.predict(text), self.name, self.cat)


class RuleBasedParser(RuleParser):
    def parse(self, text: str) -> dict:
        return to_spec(text, self.predict(text), self.name, self.cat)


def to_spec(text: str, pred: dict, backend: str, cat: catalog.Catalog | None = None) -> dict:
    spec = build_spec(text, pred["tokens"], pred["chart_type"], pred["spans"],
                      pred["chart_confidence"], pred.get("chart_alternatives"), cat=cat)
    spec["backend"] = backend
    if pred.get("chart_source"):
        spec["chart_source"] = pred["chart_source"]
    return spec


def load_parser(model_dir: str | Path | None = "models/ivis-bert", device: str = "auto", hybrid: bool = True,
                cat: catalog.Catalog | None = None):
    """The hybrid reader (BERT + strong keyword cues) when a checkpoint exists, otherwise the keyword parser."""
    if model_dir and (Path(model_dir) / "ivis_model.pt").exists():
        bert = BertParser(model_dir, device, cat=cat)
        return HybridParser(bert) if hybrid else bert
    return RuleBasedParser(cat)
