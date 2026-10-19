#!/bin/bash
# The public 0.3 suite through the engine, against the server serve.sh started; then the kit's own scorer.
#   pip install -r requirements.txt   (the kit itself: pip install -e <decision-index checkout>, commit 9eb2dbe)
PORT=${PORT:-8000}
MODEL=${MODEL:-morriszjm/AnyJev-Tacitus-31B}
OUT=${OUT:-runs/anyjev-tacitus-31b-v1}
PYTHONPATH=$(dirname "$0"):$PYTHONPATH python -m decision_index run --engine anyjev_engine:AnyJevEngine \
  --option base_url=http://127.0.0.1:$PORT --option model=$MODEL \
  --option level=L0 --option max_rotations=2 --option big_rotations=2 --option bidi=block --option prior=none \
  --out $OUT --compact
python -m decision_index score --edition 0.3 --results $OUT/results.jsonl --out $OUT
