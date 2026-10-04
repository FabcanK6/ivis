"""Hybrid reader, your own data model, the AI reader (fake model), the Power BI visual file and the benchmark."""
import json
import unittest

from ivis import catalog as C
from ivis.ai import AIParser
from ivis.benchmark import gold_spec, score, test_split, visual_key
from ivis.data.generate import make_example
from ivis.pbir import to_visual_json
from ivis.predict import HybridParser, RuleBasedParser, to_spec
from ivis.rules import chart_cue
from ivis.text import words

MY_ROWS = [
    {"Kind": "Measure", "Field": "Sales[Revenue]", "Also called": "turnover", "Values": "",
     "Default aggregation": "sum"},
    {"Kind": "Measure", "Field": "Sales[Order Count]", "Also called": "", "Values": "", "Default aggregation": "count"},
    {"Kind": "Column", "Field": "Product[Category]", "Also called": "product line", "Values": "Bikes, Helmets",
     "Default aggregation": ""},
    {"Kind": "Location column", "Field": "Store[Country]", "Also called": "", "Values": "Germany, France",
     "Default aggregation": ""},
    {"Kind": "Date column", "Field": "Calendar[Date]", "Also called": "", "Values": "", "Default aggregation": ""},
    {"Kind": "Date column", "Field": "Calendar[Month]", "Also called": "", "Values": "", "Default aggregation": ""},
]


class FakeBert:
    """Stands in for BERT: always answers 'bar' with 90% confidence and no parts."""

    cat = C.DEFAULT

    def predict_batch(self, texts):
        return [{"tokens": words(t), "tags": ["O"] * len(words(t)), "spans": [], "chart_type": "bar",
                 "chart_confidence": 0.9, "chart_alternatives": [("table", 0.05)]} for t in texts]


class FakeAI:
    model = "gemini-test-flash"

    def __init__(self, answer):
        self.answer = answer
        self.asked = []

    def generate_json(self, system, prompt, schema):
        self.asked.append((system, prompt))
        return self.answer


class TestHybrid(unittest.TestCase):
    def test_strong_cue_overrides_bert(self):
        p = HybridParser(FakeBert()).predict("show queries per site split by country")
        self.assertEqual((p["chart_type"], p["chart_source"]), ("stacked_bar", 'the words "split by"'))
        self.assertEqual(p["chart_alternatives"][0][0], "bar")  # BERT's choice is kept as an alternative
        spec = to_spec("show queries per site split by country", p, "hybrid")
        self.assertEqual(spec["chart_source"], 'the words "split by"')

    def test_weak_cue_leaves_bert_alone(self):
        self.assertEqual(chart_cue("how many queries by site?")[2], False)
        p = HybridParser(FakeBert()).predict("how many queries by site?")
        self.assertEqual((p["chart_type"], p["chart_source"]), ("bar", "BERT"))

    def test_strong_cues_are_precise_on_validation_data(self):
        import random

        from ivis.data.templates import TEMPLATES

        rng = random.Random(3)
        hits = right = 0
        for chart, temps in TEMPLATES.items():
            for i in range(len(temps)):
                for _ in range(5):
                    ex = make_example(rng, chart, i)
                    c, _, strong = chart_cue(ex["text"])
                    if strong:
                        hits += 1
                        right += c == chart
        self.assertGreater(hits, 200)
        self.assertGreaterEqual(right / hits, 0.97)


class TestOwnDataModel(unittest.TestCase):
    def setUp(self):
        self.cat, self.problems = C.Catalog.from_rows(MY_ROWS, name="Shop")

    def test_rows_become_a_catalog(self):
        self.assertEqual(self.problems, [])
        self.assertEqual(self.cat.date_field, "Calendar[Date]")
        self.assertEqual(self.cat.default_time_axis().field, "Calendar[Month]")
        self.assertIn("orders", self.cat.measure_by_field("Sales[Order Count]").synonyms)  # from "Order Count"
        self.assertIn("monthly", self.cat.dimension_by_field("Calendar[Month]").synonyms)
        _, problems = C.Catalog.from_rows([{"Kind": "Column", "Field": "Revenue"}])
        self.assertEqual(len(problems), 2)  # not Table[Column], and no measure at all
        rows = self.cat.to_rows()
        self.assertEqual(C.Catalog.from_rows(rows)[0].fingerprint(), C.Catalog.from_rows(rows)[0].fingerprint())

    def test_keyword_reader_uses_it(self):
        spec = RuleBasedParser(self.cat).parse("total revenue by product line in Germany last month")
        self.assertEqual([m["field"] for m in spec["measures"]], ["Sales[Revenue]"])
        self.assertEqual([g["field"] for g in spec["group_by"]], ["Product[Category]"])
        self.assertEqual([(f["field"], f["values"]) for f in spec["filters"]], [("Store[Country]", ["Germany"])])
        self.assertEqual(spec["time_filter"]["field"], "Calendar[Date]")
        trend = RuleBasedParser(self.cat).parse("trend of orders")
        self.assertEqual(trend["group_by"][0]["field"], "Calendar[Month]")  # trend charts get the month axis
        self.assertEqual(trend["data_model"], "Shop")


class TestAIReader(unittest.TestCase):
    def test_answer_is_checked_against_request_and_data_model(self):
        text = "How fast do queries close by site in Germany last 30 days?"
        answer = {"chart_type": "bar", "parts": [
            {"role": "measure", "text": "How fast do queries close", "field": "Queries[Days Open]"},
            {"role": "axis", "text": "site", "field": "Site[Site Name]"},
            {"role": "filter", "text": "Germany", "field": "Site[Country]"},
            {"role": "time", "text": "last 30 days"},
            {"role": "legend", "text": "by therapeutic area", "field": "Study[Therapeutic Area]"},  # not in request
        ]}
        client = FakeAI(answer)
        spec = AIParser(client).parse(text)
        self.assertEqual(spec["measures"][0]["field"], "Queries[Days Open]")  # words alone would not resolve
        self.assertEqual(spec["group_by"][0]["field"], "Site[Site Name]")
        self.assertEqual(spec["filters"][0]["values"], ["Germany"])
        self.assertEqual(spec["time_filter"]["timeUnitsCount"], 30)
        self.assertEqual(spec["series"], [])
        self.assertIn("not in the request", spec["warnings"][0])
        self.assertEqual((spec["backend"], spec["ai_model"]), ("ai", "gemini-test-flash"))
        self.assertIn("Queries[Days Open]", client.asked[0][0])  # the data model is in the instructions

    def test_wrong_field_is_ignored(self):
        answer = {"chart_type": "pie", "parts": [
            {"role": "measure", "text": "SAEs", "field": "Safety[Made Up]"},
            {"role": "axis", "text": "country", "field": "Queries[Query Count]"},  # a measure used as an axis
        ]}
        spec = AIParser(FakeAI(answer)).parse("share of SAEs by country")
        self.assertEqual(spec["measures"][0]["field"], "Safety[SAE Count]")  # matched by its words instead
        self.assertEqual(spec["group_by"][0]["field"], "Site[Country]")
        self.assertEqual(sum("not a" in w for w in spec["warnings"]), 2)
        bad = AIParser(FakeAI({"chart_type": "sunburst", "parts": []})).parse("SAEs")
        self.assertEqual(bad["chart_type"], "bar")


class TestPowerBIFile(unittest.TestCase):
    def test_bar_with_filter_and_sort(self):
        spec = RuleBasedParser().parse("top 10 sites with the most open queries in Germany last 30 days")
        v, notes = to_visual_json(spec)
        q = v["visual"]["query"]["queryState"]
        self.assertTrue(v["$schema"].endswith("/schema.json"))
        self.assertEqual(v["visual"]["visualType"], "clusteredBarChart")
        self.assertEqual(q["Category"]["projections"][0]["queryRef"], "Site.Site Name")
        self.assertIn("Measure", q["Y"]["projections"][0]["field"])
        self.assertEqual(v["visual"]["query"]["sortDefinition"]["sort"][0]["direction"], "Descending")
        flt = v["filterConfig"]["filters"]
        self.assertEqual({f["field"]["Column"]["Property"] for f in flt}, {"Status", "Country"})
        values = [x for f in flt for x in f["filter"]["Where"][0]["Condition"]["In"]["Values"]]
        self.assertIn([{"Literal": {"Value": "'Germany'"}}], values)
        self.assertTrue(any("Top 10" in n for n in notes) and any("last 30 days" in n for n in notes))
        json.dumps(v)

    def test_scatter_card_and_quotes(self):
        v, _ = to_visual_json(RuleBasedParser().parse("query aging vs SDV completion by site"))
        self.assertEqual(set(v["visual"]["query"]["queryState"]), {"Category", "X", "Y"})
        v, _ = to_visual_json(RuleBasedParser().parse("how many SAEs do we have?"))
        self.assertEqual((v["visual"]["visualType"], list(v["visual"]["query"]["queryState"])), ("card", ["Values"]))
        cat = C.Catalog.from_rows([*MY_ROWS, {"Kind": "Column", "Field": "Store[Name]", "Values": "O'Brien's"}])[0]
        v, _ = to_visual_json(RuleBasedParser(cat).parse("revenue by product line at O'Brien's"), cat)
        self.assertIn("'O''Brien''s'", json.dumps(v))


class TestBenchmark(unittest.TestCase):
    def test_labels_score_perfectly_and_label_fix(self):
        rows = test_split()[:60]
        rep = score(rows, [{"tokens": r["tokens"], "tags": r["tags"], "chart_type": r["chart_type"]} for r in rows])
        self.assertEqual((rep["same_visual"], rep["frame_exact_match"]), (1.0, 1.0))
        self.assertEqual(visual_key(gold_spec(rows[0])), visual_key(gold_spec(rows[0])))
        import random

        from ivis.data.templates import TEMPLATES

        i = next(k for k, t in enumerate(TEMPLATES["pie"]) if "contribution" in t)
        ex = make_example(random.Random(1), "pie", i)
        k = [t.lower() for t in ex["tokens"]].index("total")
        self.assertEqual(ex["tags"][k], "B-AGG")  # "total" before a measure is an aggregation everywhere


if __name__ == "__main__":
    unittest.main()
