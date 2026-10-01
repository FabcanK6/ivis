import random
import unittest

from ivis.data.generate import build_splits, make_example
from ivis.data.templates import TEMPLATES
from ivis.schema import BIO_LABELS, CHART_TYPES
from ivis.text import bio_to_spans, detokenize, spans_to_bio, words


class TestText(unittest.TestCase):
    def test_words_roundtrip(self):
        for text in ["Top 10 sites with the most open queries in Germany?",
                     "SAEs for ONC-301 since January, please",
                     "for tomorrow's study meeting: SDV % by CRA"]:
            self.assertEqual(words(detokenize(words(text))), words(text))

    def test_bio_roundtrip(self):
        spans = [("METRIC", 0, 2), ("GROUP_BY", 3, 4), ("TIME", 5, 8)]
        tags = spans_to_bio(9, spans)
        self.assertEqual([s.as_tuple() for s in bio_to_spans(["w"] * 9, tags)], spans)

    def test_stray_inside_tag_starts_span(self):
        spans = bio_to_spans(["a", "b", "c"], ["O", "I-METRIC", "I-METRIC"])
        self.assertEqual([s.as_tuple() for s in spans], [("METRIC", 1, 3)])


class TestGenerator(unittest.TestCase):
    def test_every_template_expands(self):
        rng = random.Random(0)
        for chart, templates in TEMPLATES.items():
            self.assertIn(chart, CHART_TYPES)
            for i in range(len(templates)):
                for _ in range(20):
                    ex = make_example(rng, chart, i)
                    self.assertEqual(len(ex["tokens"]), len(ex["tags"]))
                    self.assertEqual(words(ex["text"]), ex["tokens"])
                    self.assertTrue(set(ex["tags"]) <= set(BIO_LABELS))
                    self.assertTrue(any(t.endswith("METRIC") for t in ex["tags"]))

    def test_heldout_templates_only_in_test(self):
        train, val, test, held = build_splits(500, 50, 200, seed=1)
        held_ids = {f"{c}/{i}" for c, ids in held.items() for i in ids}
        self.assertTrue(held_ids)
        self.assertFalse({r["template_id"] for r in train + val} & held_ids)
        self.assertTrue({r["template_id"] for r in test} & held_ids)

    def test_deterministic(self):
        a = build_splits(50, 5, 10, seed=7)[0]
        b = build_splits(50, 5, 10, seed=7)[0]
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
