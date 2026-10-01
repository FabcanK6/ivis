"""Joint chart-type classifier + slot tagger on top of a BERT-style encoder.

    words ──► WordPiece ──► BERT encoder ──┬─► [CLS] ─► Linear ─► chart type (11 classes)
                                           └─► tokens ─► Linear ─► BIO slot tag per word

Both heads share the encoder and are trained together (sum of two
cross-entropy losses), the standard "JointBERT" set-up for intent detection
and slot filling. Any Hugging Face encoder with a fast tokenizer works:
``bert-base-uncased``, ``distilbert-base-uncased``, ``roberta-base``,
``microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext``, ...
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

from ivis.schema import BIO_LABELS, CHART_TYPES, LABEL2ID

WEIGHTS_NAME = "ivis_model.pt"
CONFIG_NAME = "ivis_config.json"


class IvisJointModel(nn.Module):
    def __init__(self, encoder: nn.Module, num_charts: int, num_labels: int,
                 dropout: float = 0.1, slot_loss_weight: float = 1.0):
        super().__init__()
        self.encoder = encoder
        hidden = encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.chart_head = nn.Linear(hidden, num_charts)
        self.slot_head = nn.Linear(hidden, num_labels)
        self.slot_loss_weight = slot_loss_weight
        self._accepts_token_types = "token_type_ids" in inspect.signature(encoder.forward).parameters

    @classmethod
    def from_encoder_name(cls, name: str, **kwargs) -> IvisJointModel:
        encoder = AutoModel.from_pretrained(name)
        return cls(encoder, num_charts=len(CHART_TYPES), num_labels=len(BIO_LABELS), **kwargs)

    def forward(self, input_ids, attention_mask, token_type_ids=None,
                chart_labels=None, slot_labels=None) -> dict:
        enc_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None and self._accepts_token_types:
            enc_kwargs["token_type_ids"] = token_type_ids
        hidden = self.encoder(**enc_kwargs).last_hidden_state          # (B, T, H)
        chart_logits = self.chart_head(self.dropout(hidden[:, 0]))      # (B, C)
        slot_logits = self.slot_head(self.dropout(hidden))              # (B, T, L)

        out = {"chart_logits": chart_logits, "slot_logits": slot_logits}
        if chart_labels is not None and slot_labels is not None:
            ce = nn.CrossEntropyLoss(ignore_index=-100)
            chart_loss = ce(chart_logits, chart_labels)
            slot_loss = ce(slot_logits.reshape(-1, slot_logits.size(-1)), slot_labels.reshape(-1))
            out.update(loss=chart_loss + self.slot_loss_weight * slot_loss,
                       chart_loss=chart_loss.detach(), slot_loss=slot_loss.detach())
        return out

    # ------------------------------------------------------------------ io
    def save(self, out_dir: str | Path, tokenizer, base_model: str, max_length: int, extra: dict | None = None):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.encoder.config.save_pretrained(out)
        tokenizer.save_pretrained(out)
        torch.save(self.state_dict(), out / WEIGHTS_NAME)
        cfg = {
            "base_model": base_model,
            "chart_types": CHART_TYPES,
            "bio_labels": BIO_LABELS,
            "max_length": max_length,
            "dropout": self.dropout.p,
            **(extra or {}),
        }
        (out / CONFIG_NAME).write_text(json.dumps(cfg, indent=2))

    @classmethod
    def load(cls, model_dir: str | Path, device: str | torch.device = "cpu"):
        model_dir = Path(model_dir)
        cfg = json.loads((model_dir / CONFIG_NAME).read_text())
        if cfg["chart_types"] != CHART_TYPES or cfg["bio_labels"] != BIO_LABELS:
            raise ValueError("Checkpoint label set differs from ivis.schema; retrain or restore the old schema.")
        encoder = AutoModel.from_config(AutoConfig.from_pretrained(model_dir))
        model = cls(encoder, len(CHART_TYPES), len(BIO_LABELS), dropout=cfg.get("dropout", 0.1))
        state = torch.load(model_dir / WEIGHTS_NAME, map_location=device, weights_only=True)
        model.load_state_dict(state)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        return model.to(device).eval(), tokenizer, cfg


def encode_words(tokenizer, batch_tokens: list[list[str]], max_length: int,
                 batch_tags: list[list[str]] | None = None):
    """Tokenize pre-split words and align word-level BIO tags to WordPiece tokens.

    Only the first sub-token of each word carries a label; the rest (and the
    special tokens) get -100 so the loss ignores them. Returns the encoding and,
    per example, the token index of each word's first sub-token (or None when
    the word was truncated away).
    """
    if not tokenizer.is_fast:
        raise ValueError("iVIS needs a *fast* tokenizer (word_ids()).")
    enc = tokenizer(batch_tokens, is_split_into_words=True, truncation=True, max_length=max_length,
                    padding=True, return_tensors="pt")
    first_index: list[list[int | None]] = []
    labels = []
    for b, words in enumerate(batch_tokens):
        word_ids = enc.word_ids(batch_index=b)
        firsts: list[int | None] = [None] * len(words)
        row = [-100] * len(word_ids)
        prev = None
        for t, w in enumerate(word_ids):
            if w is not None and w != prev:
                firsts[w] = t
                if batch_tags is not None:
                    row[t] = LABEL2ID[batch_tags[b][w]]
            prev = w
        first_index.append(firsts)
        labels.append(row)
    if batch_tags is not None:
        enc["slot_labels"] = torch.tensor(labels, dtype=torch.long)
    return enc, first_index


def pick_device(pref: str = "auto") -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
