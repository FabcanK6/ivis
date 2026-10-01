#!/usr/bin/env bash
# End-to-end: generate data -> fine-tune BERT -> evaluate -> compare with the rule baseline.
set -euo pipefail
MODEL="${MODEL:-bert-base-uncased}"
OUT="${OUT:-models/ivis-bert}"
EPOCHS="${EPOCHS:-4}"

python -m ivis.data.generate --out data --n-train 20000 --n-val 2000 --n-test 2000
python -m ivis.train --model "$MODEL" --out "$OUT" --epochs "$EPOCHS" --train data/train.jsonl --val data/val.jsonl "$@"
mkdir -p reports
python -m ivis.evaluate --data data/test.jsonl --model "$OUT" --report reports/bert_test.json
python -m ivis.evaluate --data data/test.jsonl --backend rules --report reports/rules_test.json --errors 0
