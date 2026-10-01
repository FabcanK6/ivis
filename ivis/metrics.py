"""Evaluation metrics (pure Python, no seqeval dependency)."""

from __future__ import annotations

from collections import Counter, defaultdict

from ivis.text import bio_to_spans


def span_prf(gold_tags: list[list[str]], pred_tags: list[list[str]]) -> dict:
    """Exact-match span precision/recall/F1, micro-averaged and per slot type."""
    tp, fp, fn = Counter(), Counter(), Counter()
    for g, p in zip(gold_tags, pred_tags):
        dummy = [""] * len(g)
        gs = {s.as_tuple() for s in bio_to_spans(dummy, g)}
        ps = {s.as_tuple() for s in bio_to_spans(dummy, p)}
        for s in gs & ps:
            tp[s[0]] += 1
        for s in ps - gs:
            fp[s[0]] += 1
        for s in gs - ps:
            fn[s[0]] += 1

    def prf(t, f_p, f_n):
        p = t / (t + f_p) if t + f_p else 0.0
        r = t / (t + f_n) if t + f_n else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        return {"precision": p, "recall": r, "f1": f, "support": t + f_n}

    labels = sorted(set(tp) | set(fp) | set(fn))
    out = {"micro": prf(sum(tp.values()), sum(fp.values()), sum(fn.values()))}
    out["per_slot"] = {lab: prf(tp[lab], fp[lab], fn[lab]) for lab in labels}
    return out


def evaluate_predictions(examples: list[dict], preds: list[dict]) -> dict:
    """``examples`` are dataset rows; ``preds`` hold ``chart_type`` and word-level ``tags``."""
    n = len(examples)
    correct = sum(e["chart_type"] == p["chart_type"] for e, p in zip(examples, preds))
    exact = sum(e["chart_type"] == p["chart_type"] and e["tags"] == p["tags"] for e, p in zip(examples, preds))
    confusion: dict[str, Counter] = defaultdict(Counter)
    for e, p in zip(examples, preds):
        confusion[e["chart_type"]][p["chart_type"]] += 1
    per_chart = {c: row[c] / sum(row.values()) for c, row in confusion.items()}
    result = {
        "n": n,
        "chart_accuracy": correct / n if n else 0.0,
        "frame_exact_match": exact / n if n else 0.0,
        "slots": span_prf([e["tags"] for e in examples], [p["tags"] for p in preds]),
        "chart_accuracy_per_type": dict(sorted(per_chart.items())),
        "confusion": {k: dict(v) for k, v in sorted(confusion.items())},
    }
    if any("unseen_template" in e for e in examples):
        for flag, name in ((False, "seen_templates"), (True, "unseen_templates")):
            idx = [i for i, e in enumerate(examples) if e.get("unseen_template", False) == flag]
            if idx:
                sub_e = [examples[i] for i in idx]
                sub_p = [preds[i] for i in idx]
                result[name] = {
                    "n": len(idx),
                    "chart_accuracy": sum(e["chart_type"] == p["chart_type"] for e, p in zip(sub_e, sub_p)) / len(idx),
                    "slot_f1": span_prf([e["tags"] for e in sub_e], [p["tags"] for p in sub_p])["micro"]["f1"],
                }
    return result


def format_report(r: dict) -> str:
    lines = [
        f"examples           : {r['n']}",
        f"chart-type accuracy: {r['chart_accuracy']:.3f}",
        f"slot micro F1      : {r['slots']['micro']['f1']:.3f}",
        f"exact frame match  : {r['frame_exact_match']:.3f}",
    ]
    for k in ("seen_templates", "unseen_templates"):
        if k in r:
            lines.append(f"  {k:<17}: n={r[k]['n']:<5} chart acc={r[k]['chart_accuracy']:.3f}  slot F1={r[k]['slot_f1']:.3f}")
    lines.append("\nper-slot F1:")
    for lab, m in r["slots"]["per_slot"].items():
        lines.append(f"  {lab:<9} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} (n={m['support']})")
    lines.append("\nper-chart accuracy:")
    for c, a in r["chart_accuracy_per_type"].items():
        lines.append(f"  {c:<12} {a:.3f}")
    return "\n".join(lines)
