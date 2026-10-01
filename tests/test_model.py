"""Model tests. Skipped automatically when torch/transformers are not installed.

Builds a tiny randomly initialised BERT (no download needed) and checks the
forward pass, label alignment, and save/load round trip.
"""
import tempfile
import unittest

try:
    import torch
    from transformers import BertConfig, BertModel, BertTokenizerFast
    HAVE_TORCH = True
except ImportError:  # pragma: no cover
    HAVE_TORCH = False


@unittest.skipUnless(HAVE_TORCH, "torch/transformers not installed")
class TestJointModel(unittest.TestCase):
    def setUp(self):
        from ivis.catalog import DIMENSIONS, MEASURES
        from ivis.text import words

        vocab = {"[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "?", ",", ":", "#"}
        for m in MEASURES:
            for s in m.synonyms:
                vocab.update(w.lower() for w in words(s))
        for d in DIMENSIONS:
            for s in d.synonyms + d.values:
                vocab.update(w.lower() for w in words(s))
        vocab.update("top 10 by in the of for last 30 days show chart".split())
        self.tmp = tempfile.TemporaryDirectory()
        vocab_file = f"{self.tmp.name}/vocab.txt"
        with open(vocab_file, "w") as f:
            f.write("\n".join(sorted(vocab)))
        self.tokenizer = BertTokenizerFast(vocab_file=vocab_file)
        cfg = BertConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                         intermediate_size=64, max_position_embeddings=128)
        from ivis.model import IvisJointModel
        from ivis.schema import BIO_LABELS, CHART_TYPES
        self.model = IvisJointModel(BertModel(cfg), len(CHART_TYPES), len(BIO_LABELS))

    def tearDown(self):
        self.tmp.cleanup()

    def test_forward_and_alignment(self):
        from ivis.model import encode_words
        tokens = [["show", "queries", "by", "site"], ["SAEs", "in", "Germany"]]
        tags = [["O", "B-METRIC", "O", "B-GROUP_BY"], ["B-METRIC", "O", "B-FILTER"]]
        enc, firsts = encode_words(self.tokenizer, tokens, 32, tags)
        self.assertEqual(len(firsts[0]), 4)
        labelled = (enc["slot_labels"][0] != -100).sum().item()
        self.assertEqual(labelled, 4)
        out = self.model(**dict(enc), chart_labels=torch.tensor([0, 1]))
        self.assertTrue(torch.isfinite(out["loss"]))
        out["loss"].backward()

    def test_save_load_and_parse(self):
        from ivis.model import IvisJointModel
        from ivis.predict import BertParser
        d = f"{self.tmp.name}/ckpt"
        self.model.save(d, self.tokenizer, base_model="tiny-test", max_length=32)
        model, tok, cfg = IvisJointModel.load(d)
        self.assertEqual(cfg["base_model"], "tiny-test")
        spec = BertParser(d, device="cpu").parse("top 10 sites by open queries in Germany")
        self.assertIn("powerbi_visual", spec)


if __name__ == "__main__":
    unittest.main()
