# iVIS: Intelligent Visualization Insight Synthesizer

[![tests](https://github.com/FabcanK6/ivis/actions/workflows/tests.yml/badge.svg)](https://github.com/FabcanK6/ivis/actions/workflows/tests.yml)
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ivis-fabcank6.streamlit.app)
[![Model on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Model-ivis--bert-yellow)](https://huggingface.co/FabcanK6/ivis-bert)

**Turn plain-English dashboard requests into Power BI visuals.**

**[▶ Open the live app](https://ivis-fabcank6.streamlit.app)**

Clinical stakeholders tend to ask for dashboards in loose terms, for example *"Can we get a chart of top queries by site?"*, and a BI developer then has to work out the chart type, measure, axis, filters and date window. iVIS does that step automatically. It reads the request and returns a structured, Power BI-ready spec, a **visual file you can drop into a Power BI report**, and a DevOps work item.

- **Three ways to read a request:** a fine-tuned BERT model with keyword cues (fast and free, the default), an **AI reader** (Google Gemini, free tier) whose every answer is checked by code, or keywords only
- **Your own data model:** describe your dataset's measures and columns in the app (or upload a CSV), and requests are mapped onto your fields; it is kept in your own browser
- **Power BI visual file:** a PBIR `visual.json` with the visual type, fields, sort, title and filters, plus a short list of what to finish in Power BI
- **Accuracy check in the app:** run any reader on the 2,000-request test set and see where it goes wrong

```text
"Can we get a chart of the top 10 sites with the most open queries in Germany over the last 30 days?"
                                            │
                                            ▼
 chart: bar → clusteredBarChart     measure: Queries[Query Count] (count)
 axis:  Site[Site Name]             filters: Queries[Status] = open, Site[Country] = Germany
 top N: 10 by Query Count (desc)    date:    InLast 30 Days on Date[Date]
```

---

## Demo

**Request:** *"Trend of SAEs by month for ONC-301 this year"*

BERT tags each part of the request:

![Detected spans](docs/app-spans.png)

and iVIS turns it into a Power BI line chart spec (preview uses mock data):

![Line chart preview](docs/app-preview.png)

plus a ready-to-file DevOps ticket:

![DevOps ticket](docs/app-ticket.png)

---

## How it works

```mermaid
flowchart LR
    A[Stakeholder request] --> B[Word tokenizer]
    B --> C[BERT encoder]
    C -->|CLS vector| D[Chart-type head<br/>11 classes]
    C -->|token vectors| E[Slot head<br/>BIO tags]
    D --> F[Spec builder]
    E --> F
    G[(Semantic catalog<br/>measures · dimensions · values)] --> F
    K[Keyword cues] -->|strong cue overrides chart| F
    L[AI reader: Gemini] -->|parts + fields, checked by code| F
    F --> H[Power BI visual spec JSON]
    F --> M[PBIR visual.json]
    F --> I[Azure DevOps work item]
    H --> J[Streamlit app]
```

iVIS treats the task as **joint intent classification + slot filling** (the "JointBERT" set-up):

| Component | What it does |
|---|---|
| **Chart-type head** | Classifies the request into one of 11 visuals: `bar`, `stacked_bar`, `line`, `area`, `pie`, `donut`, `table`, `matrix`, `card`, `scatter`, `map`. |
| **Slot head** | Tags each word with BIO labels for `METRIC`, `AGG`, `GROUP_BY`, `SERIES`, `FILTER`, `TIME`, `TOPN`, `SORT`. |
| **Semantic catalog** (`ivis/catalog.py`) | Maps detected spans to real fields, e.g. "open queries" → `Queries[Query Count]` + filter `Queries[Status] = open`. |
| **Spec builder** (`ivis/spec.py`) | Normalizes time windows into Power BI relative-date filters, resolves top-N and sort, applies chart-specific defaults (a line chart with no axis gets `Date[Month]`), fills the visual's field wells and lists anything it couldn't resolve under `warnings`. |
| **Rule baseline** (`ivis/rules.py`) | The original keyword/lexicon parser. Used as a baseline for evaluation and as a fallback when there's no trained checkpoint. |
| **Hybrid reader** (`ivis/predict.py`) | BERT's word tags, but a *strong* chart cue in the request ("split by", "headline", "donut", "crosstab") decides the chart. Each strong cue was checked on the validation split (1,302 hits, all correct); weak cues such as "how many" or "percentage" never override BERT. |
| **AI reader** (`ivis/ai.py`, `ivis/llm.py`) | Gemini returns the chart type and each part of the request with the exact words and the field it chose from the data model. Code drops any part whose words are not in the request and ignores any field that is not in the data model (or is the wrong kind), then the same spec builder runs. Answers are cached for a day; the client falls back across free models and skips models whose daily quota is spent. |
| **Visual file** (`ivis/pbir.py`) | Writes the spec as a PBIR `visual.json`: visual type, fields per role, sort, title and categorical filters. Relative date windows and top N are listed for the user to add in Power BI. |
| **Benchmark** (`ivis/benchmark.py`) | Rebuilds the standard test split and scores any reader, including a *same visual* score that compares the visuals built from the reading and from the labels. |

The reader only decides *what was asked for*, and the catalog decides *which fields that means*. To point iVIS at your own dataset, describe it in the app's **Data model** tab (or build a `Catalog` with `Catalog.from_rows`). The keyword and AI readers use it directly; BERT learned the built-in model's wording, so with a very different dataset the AI reader maps requests best.

---

## Quickstart

```bash
git clone https://github.com/FabcanK6/ivis.git
cd ivis
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**Run it:**

```bash
# web app: downloads the trained model from the Hugging Face Hub on first run
streamlit run app/streamlit_app.py

# command line (uses the BERT model in models/ivis-bert if present, otherwise the rule parser)
python -m ivis.cli "how many screen failures do we have at Site 104 since January?"
python -m ivis.cli --format devops "protocol deviations by country broken down by category in 2025"
python -m ivis.cli --format visual "share of SAEs by country this year" > visual.json      # Power BI visual file
GEMINI_API_KEY=... python -m ivis.cli --backend ai "how fast do queries close by site?"     # AI reader
```

**Train the BERT model:**

```bash
# 1. Generate synthetic training data (20k train / 2k val / 2k test)
python -m ivis.data.generate --out data

# 2. Fine-tune (GPU recommended; on CPU try --model prajjwal1/bert-tiny first)
python -m ivis.train --model bert-base-uncased --out models/ivis-bert --epochs 4

# 3. Evaluate on the test split and compare with the rule baseline
python -m ivis.evaluate --data data/test.jsonl --model models/ivis-bert
python -m ivis.evaluate --data data/test.jsonl --backend rules

# Or run all of the above:
./scripts/train_bert.sh
```

If you don't have a GPU, open `notebooks/train_on_colab.ipynb` in Google Colab with a T4 runtime.

Once `models/ivis-bert/` exists, both the CLI and the Streamlit app pick it up automatically. To use a different directory, set `IVIS_MODEL_DIR`; to pull a different checkpoint from the Hugging Face Hub, set `IVIS_HF_MODEL`.

### Other encoders

Any Hugging Face encoder with a fast tokenizer works with `--model`:

| Model | Notes |
|---|---|
| `bert-base-uncased` | Default. |
| `distilbert-base-uncased` | About 2× faster, slightly lower accuracy. |
| `prajjwal1/bert-tiny` | For CPU smoke tests. |
| `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext` | Biomedical vocabulary; worth trying with real clinical requests. |
| `roberta-base` | Supported (`add_prefix_space` is handled for you). |

---

## Training data

Real stakeholder requests are scarce and often confidential, so iVIS starts from a **template-based synthetic generator** (`ivis/data/generate.py`):

* About 85 request templates across the 11 chart types. The phrasing is deliberately indirect ("is enrollment going up or down?", "what share of AEs comes from each country?"), so the model has to learn chart cues instead of looking for the word "pie".
* Slots are filled from the clinical-operations catalog: 14 measures (queries, SAEs, deviations, enrollment, SDV %, ...), 20 dimensions (site, country, study, CRA, visit, arm, ...), filter values, 25 time expressions, top-N phrases and sort words.
* Augmentation adds conversational prefixes and suffixes ("for the steering committee, ..."), random casing, dropped punctuation and occasional typos.
* **Held-out templates:** one template per chart type never appears in train/val. Half of the test set uses those templates, which measures generalization to unseen phrasing.

Each row looks like this:

```json
{"text": "show open queries by site in Germany last 30 days",
 "tokens": ["show", "open", "queries", "by", "site", "in", "Germany", "last", "30", "days"],
 "tags":   ["O", "B-FILTER", "B-METRIC", "O", "B-GROUP_BY", "O", "B-FILTER", "B-TIME", "I-TIME", "I-TIME"],
 "chart_type": "bar", "template_id": "bar/1"}
```

`data/sample.jsonl` has 150 example rows. Full splits are generated on demand and git-ignored.

**Adding real data:** append real requests in the same JSONL format to `data/train.jsonl`. Even a few hundred hand-labelled requests from real user stories will help the model more than additional synthetic data.

---

## Results

Test set: 2,000 requests. Half use phrasings seen during training; the other half come from 10 held-out templates the model never saw. Measured on the live app's **Accuracy check** tab, with the corrected labels.

| Reader | Chart type right | Same visual | New wording: chart right | New wording: same visual | Slot micro-F1 | Exact frame match |
|---|---|---|---|---|---|---|
| Keywords only (`ivis/rules.py`) | 94.5% | 73.0% | 95.6% | **76.0%** | 0.904 | 63.8% |
| BERT only (`bert-base-uncased`, 4 epochs) | 92.6% | 83.4% | 85.3% | 66.8% | **0.962** | 83.0% |
| **BERT + keyword cues** (default) | **99.6%** | **85.1%** | **99.4%** | 70.5% | **0.962** | **84.8%** |
| AI (Gemini Flash, fixed sample of 40) | **100%** | **87.5%** | **100%** | 75.0% | n/a | n/a |

*Same visual* means the Power BI visual built from the reading matches the one built from the labels (chart, measures, axis, legend, filters, time window, top N). The AI reader is scored on a fixed sample of 40 (half new wording) because each request uses free AI quota, so its numbers carry more uncertainty than the others (one request is 2.5 points). Its first run scored 77.5% same visual; most misses were trend requests where it read "over time" as a time window, which its instructions and a code check now rule out. Its remaining misses were "over time" read as a date axis and a filter value that kept its column word ("CRA Smith"), both now handled in code. It also maps wording that is not in the data model's synonyms (live example: "how fast are queries getting closed" → `Queries[Days Open]`) and reads a user's own data model.

- **Strong chart cues close BERT's chart gap.** On new wording BERT picked the right chart 85.3% of the time; with the cues it is 99.4% (and 99.6% overall).
- **The label fix mattered.** With "contribution to total" labelled consistently, BERT's exact frame match is 83.0% (it was 79.6% against the old labels) and its slot F1 0.962.
- **What is left is in the parts, not the chart.** On new wording, BERT-based readers still build the same visual less often than keywords (70.5% vs 76.0%): some parts of unfamiliar phrasings are tagged differently. Retraining on more varied phrasing is the next step.

The first published results (before the label fix and the new readers):

| Parser | Chart-type acc. | Slot micro-F1 | Exact frame match | Unseen phrasing (chart acc. / slot F1) |
|---|---|---|---|---|
| Rule baseline (`ivis/rules.py`) | **0.946** | 0.897 | 0.600 | **0.956** / 0.909 |
| BERT (`bert-base-uncased`, 4 epochs, Colab T4) | 0.926 | **0.954** | **0.796** | 0.853 / 0.906 |

*Exact frame match* means the chart type and every slot tag are correct.

| Slot F1 | AGG | FILTER | GROUP_BY | METRIC | SERIES | SORT | TIME | TOPN |
|---|---|---|---|---|---|---|---|---|
| Rules | 0.829 | 0.959 | 0.860 | 0.998 | 0.623 | 0.821 | 0.873 | 1.000 |
| BERT | 0.886 | 1.000 | 0.898 | 0.996 | 0.801 | 1.000 | 0.999 | 1.000 |

**What the first results showed**

- **BERT is much better at pulling out the details.** Exact frame match rises from 60% to 80%. The biggest gains are in telling the axis apart from the legend (SERIES 0.62 → 0.80), time windows (0.87 → 1.00) and sort order (0.82 → 1.00).
- **BERT partly memorizes phrasing.** It is near perfect on familiar wording (chart acc. 0.999, slot F1 1.000) but drops to 0.853 chart accuracy on unseen wording, below the rule baseline. Most of those misses are stacked bars ("X per Y split by Z" read as a plain bar, 0.48) and cards (0.85).
- **Some errors come from the labels, not the model.** For example, "total" is tagged as an aggregation in some templates but not in "contribution to total", and "over time" is never tagged as a time axis.

**Next improvements:** more varied phrasing in the generator, evaluating on real stakeholder requests, and retraining BERT on the corrected labels.

> Synthetic data overstates real-world accuracy. Before relying on these numbers, evaluate on a small hand-labelled set of real requests.

---

## Output spec

`parser.parse(text)` returns:

```jsonc
{
  "title": "Query Count by Site",
  "chart_type": "bar",
  "powerbi_visual": "clusteredBarChart",
  "measures":  [{"name": "Query Count", "field": "Queries[Query Count]", "aggregation": "count"}],
  "group_by":  [{"name": "Site", "field": "Site[Site Name]"}],
  "series":    [],
  "filters":   [{"field": "Queries[Status]", "operator": "In", "values": ["open"]},
                {"field": "Site[Country]",   "operator": "In", "values": ["Germany"]}],
  "time_filter": {"field": "Date[Date]", "filterType": "RelativeDate", "operator": "InLast",
                  "timeUnitsCount": 30, "timeUnit": "Days"},
  "top_n": {"n": 10, "direction": "desc", "by": "Query Count"},
  "sort":  {"by": "Query Count", "direction": "desc"},
  "field_wells": {"Category": ["Site[Site Name]"], "Y": ["Queries[Query Count]"]},
  "confidence": {"chart_type": 0.97, "alternatives": [{"chart_type": "table", "p": 0.02}]},
  "spans": [...],
  "warnings": []
}
```

`ivis.devops.to_json_patch(spec)` produces the body for the Azure DevOps *Create Work Item* REST call (`POST .../_apis/wit/workitems/$User%20Story?api-version=7.1`, content type `application/json-patch+json`). The work item includes the title, the original request, acceptance criteria and open questions.

---

## Python API

```python
from ivis.predict import load_parser
from ivis.devops import to_markdown

parser = load_parser("models/ivis-bert")      # falls back to the rule parser if the checkpoint is missing
spec = parser.parse("Trend of SAEs by month for ONC-301 this year")
print(spec["powerbi_visual"], spec["field_wells"])
print(to_markdown(spec))
```

---

## Repository layout

```
ivis/
├── ivis/
│   ├── schema.py          # chart types, slot labels, Power BI visual names
│   ├── catalog.py         # data model (Catalog): measures, dimensions, values → Power BI fields; built-in + your own
│   ├── text.py            # word tokenizer + BIO helpers
│   ├── data/
│   │   ├── templates.py   # request templates per chart type
│   │   └── generate.py    # synthetic dataset generator (train/val/test + held-out templates)
│   ├── model.py           # JointBERT: encoder + chart head + slot head, save/load
│   ├── train.py           # fine-tuning loop (AdamW, warmup, fp16, best-checkpoint saving)
│   ├── evaluate.py        # accuracy, span F1, seen vs unseen templates, error samples
│   ├── metrics.py         # span-level P/R/F1 (no seqeval dependency)
│   ├── predict.py         # BertParser / RuleBasedParser → spec
│   ├── spec.py            # spans → Power BI spec (time, top-N, sort, defaults, field wells)
│   ├── devops.py          # spec → Azure DevOps work item / markdown
│   ├── rules.py           # keyword baseline + chart cues (strong cues used by the hybrid reader)
│   ├── ai.py              # AI reader: Gemini's answer checked against the request and the data model
│   ├── llm.py             # Gemini client (standard library; model fallback, quota memory)
│   ├── pbir.py            # spec → Power BI PBIR visual.json
│   ├── benchmark.py       # standard test split + "same visual" scoring for any reader
│   └── cli.py             # `python -m ivis.cli "..."`
├── app/streamlit_app.py   # UI: request → preview, visual file, ticket, spec; Data model tab; Accuracy check tab
├── .streamlit/config.toml # app settings
├── notebooks/train_on_colab.ipynb
├── scripts/train_bert.sh
├── tests/                 # unittest/pytest; model tests build a tiny random BERT (no download)
├── docs/                  # README screenshots
└── data/sample.jsonl
```

Run the tests with `pytest` or `python -m unittest discover -s tests`.

---

## Deployment

| | |
|---|---|
| Live app | [ivis-fabcank6.streamlit.app](https://ivis-fabcank6.streamlit.app) on Streamlit Community Cloud; redeploys on every push to `main` |
| Model | [FabcanK6/ivis-bert](https://huggingface.co/FabcanK6/ivis-bert) on the Hugging Face Hub, downloaded by the app on cold start |
| CI | GitHub Actions: lint, unit tests and a baseline evaluation on every push |
| AI reader | Optional: add `GEMINI_API_KEY` to the app's Streamlit secrets (a free key from Google AI Studio). Each browser session can make 30 new AI readings on the shared key |
| Data models | A user's own data model is stored in their browser (`localStorage`), never on the server |
| Fallback | If the model can't be loaded, the app serves the rule-based parser and shows which parser answered |

---

## Roadmap

- [ ] Evaluate on a hand-labelled set of real requests
- [ ] More varied phrasing and consistent labels in the generator
- [x] Hybrid parser: strong keyword cues override BERT's chart choice
- [x] Optional AI (LLM) mode, compared with BERT on the same test set (Accuracy check tab)
- [x] Bring your own data model
- [x] Generate a Power BI visual file (PBIR `visual.json`) from the spec
- [ ] Feedback capture in the app → corrected specs become training data
- [ ] Retrain BERT on the corrected labels and more varied phrasing
- [ ] Relative date and top N filters inside the visual file

## License

MIT © Fabian Msafiri
