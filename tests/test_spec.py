import unittest

from ivis.catalog import resolve_dimension, resolve_filter_value, resolve_measure
from ivis.devops import to_json_patch, to_markdown
from ivis.metrics import span_prf
from ivis.predict import RuleBasedParser
from ivis.spec import build_spec, parse_time, parse_topn
from ivis.text import Span, words


class TestCatalog(unittest.TestCase):
    def test_measures(self):
        self.assertEqual(resolve_measure("open queries").key, "queries")
        self.assertEqual(resolve_measure("SAEs").key, "saes")
        self.assertEqual(resolve_measure("enrolment").key, "enrolled")  # fuzzy

    def test_dimensions(self):
        self.assertEqual(resolve_dimension("sites").key, "site")
        self.assertEqual(resolve_dimension("monthly").key, "month")

    def test_filter_values(self):
        self.assertEqual(resolve_filter_value("germany")[0].key, "country")
        dim, val = resolve_filter_value("site 342")
        self.assertEqual((dim.key, val), ("site", "Site 342"))
        dim, val = resolve_filter_value("onc-999")
        self.assertEqual((dim.key, val), ("study", "ONC-999"))
        dim, val = resolve_filter_value("Okafor", context=["for", "CRA"])
        self.assertEqual((dim.key, val), ("cra", "Okafor"))


class TestNormalizers(unittest.TestCase):
    def test_time(self):
        t = parse_time("last 30 days")
        self.assertEqual((t["operator"], t["timeUnitsCount"], t["timeUnit"]), ("InLast", 30, "Days"))
        self.assertEqual(parse_time("past two weeks")["timeUnitsCount"], 2)
        self.assertEqual(parse_time("this quarter")["operator"], "InThis")
        self.assertTrue(parse_time("YTD")["toDate"])
        self.assertEqual(parse_time("Q3 2025")["quarter"], 3)
        self.assertEqual(parse_time("since March")["month"], 3)
        self.assertEqual(parse_time("2025")["year"], 2025)

    def test_topn(self):
        self.assertEqual(parse_topn("top ten"), {"n": 10, "direction": "desc", "text": "top ten"})
        self.assertEqual(parse_topn("bottom 5")["direction"], "asc")


class TestSpec(unittest.TestCase):
    def _spec(self, text, chart, spans):
        toks = words(text)
        return build_spec(text, toks, chart, [Span(lab, s, e, " ".join(toks[s:e])) for lab, s, e in spans])

    def test_bar_spec(self):
        text = "top 10 sites with the most open queries in Germany last 30 days"
        spec = self._spec(text, "bar", [("TOPN", 0, 2), ("GROUP_BY", 2, 3), ("SORT", 5, 6), ("FILTER", 6, 7),
                                        ("METRIC", 7, 8), ("FILTER", 9, 10), ("TIME", 10, 13)])
        self.assertEqual(spec["powerbi_visual"], "clusteredBarChart")
        self.assertEqual(spec["field_wells"], {"Category": ["Site[Site Name]"], "Y": ["Queries[Query Count]"]})
        self.assertEqual({f["field"] for f in spec["filters"]}, {"Queries[Status]", "Site[Country]"})
        self.assertEqual(spec["top_n"]["n"], 10)
        self.assertEqual(spec["warnings"], [])

    def test_line_defaults_to_month(self):
        spec = self._spec("trend of SAEs", "line", [("METRIC", 2, 3)])
        self.assertEqual(spec["group_by"][0]["field"], "Date[Month]")

    def test_card_drops_grouping(self):
        spec = self._spec("how many queries by site", "card", [("AGG", 0, 2), ("METRIC", 2, 3), ("GROUP_BY", 4, 5)])
        self.assertEqual(spec["group_by"], [])
        self.assertTrue(spec["warnings"])

    def test_devops(self):
        spec = self._spec("queries by site", "bar", [("METRIC", 0, 1), ("GROUP_BY", 2, 3)])
        self.assertIn("Query Count by Site", to_markdown(spec))
        self.assertTrue(all(op["path"].startswith("/fields/") for op in to_json_patch(spec)))


class TestRuleParser(unittest.TestCase):
    def test_end_to_end(self):
        spec = RuleBasedParser().parse("Can we get a chart of top 10 sites with the most open queries in Germany?")
        self.assertEqual(spec["chart_type"], "bar")
        self.assertEqual(spec["measures"][0]["field"], "Queries[Query Count]")
        self.assertEqual(spec["top_n"]["n"], 10)

    def test_line(self):
        spec = RuleBasedParser().parse("trend of SAEs by month for ONC-301 this year")
        self.assertEqual(spec["chart_type"], "line")
        self.assertEqual(spec["filters"][0]["values"], ["ONC-301"])


class TestMetrics(unittest.TestCase):
    def test_perfect_and_partial(self):
        gold = [["B-METRIC", "I-METRIC", "O", "B-GROUP_BY"]]
        self.assertEqual(span_prf(gold, gold)["micro"]["f1"], 1.0)
        pred = [["B-METRIC", "O", "O", "B-GROUP_BY"]]
        self.assertAlmostEqual(span_prf(gold, pred)["micro"]["f1"], 0.5)


if __name__ == "__main__":
    unittest.main()
