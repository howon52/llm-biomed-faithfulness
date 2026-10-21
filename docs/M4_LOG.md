# M4 experiment log

One entry per run, appended as it happens. Numbers come from `m4/report.py` output
checked into `runs/`; nothing here is retyped from memory. Decisions reference the
hypotheses and gate in `M4_RUNBOOK.md`.

## 2026-10-21 — pilot, base model, remote

- **Where**: Lambda Cloud, 1× A10 (23 GB), us-west-1, instance `581733b1…`, Lambda Stack 24.04.
  Credit, not cash (~$48.46 available at launch).
- **Serving**: `vllm 0.32.0` in a clean venv (`~/venv`); the first `pip install --user`
  collided with Lambda Stack's system scipy (`ImportError: cannot import name 'Inf' from numpy`)
  — install into a venv without system site-packages. `vllm serve Qwen/Qwen3-4B-Instruct-2507
  --dtype bfloat16 --port 8000`, reached from the Mac through `ssh -N -L 8000:localhost:8000`.
- **Spend cap**: `m4/lambda_watchdog.sh 2` armed at 12:21 → fires 14:21.
- **Command**:
  `uv run python m4/generate.py --backend openai --model Qwen/Qwen3-4B-Instruct-2507 --input data/m4/eval_scifact.jsonl --n 1 --temperature 0.7 --out runs/pilot/base_eval.jsonl`
  then `uv run python m4/report.py runs/pilot/base_eval.jsonl --json runs/pilot/base_eval.json`.
- **Setup defects hit, in order** (each cost one relaunch): `pip install --user` → numpy 2 vs
  system scipy (`ImportError: Inf`) → use a venv; default `max_model_len` 262144 needs 36 GB KV
  → `--max-model-len 8192`; torch.compile needs `ninja` → `pip install ninja` and put
  `~/venv/bin` on PATH. Box up 12:17–12:55 (38 min, ~$0.47 of credit); generation itself 1.7 min.
- **Result** (`runs/pilot/base_eval_report.txt`, 120 rows, 1 sample, T=0.7, bootstrap clustered by doc_id):
  parse 100 %, truncated 0 %; accuracy 74.2 [66.1, 81.7]; recall SUPPORTED 75.0, REFUTED 85.0,
  **INSUFFICIENT 62.5 [46.5, 77.8]**; numeric fabrication 14.1 %; grounding precision 51.5 / recall
  81.2; reasoning words p10/p50/p90/max 99/174/260/524; proxy 0.49 (at saturation 400).
  Confusion: INSUFFICIENT→REFUTED is the dominant error (11 of 40); SUPPORTED→INSUFFICIENT 7 of 40.
- **Gate decision**: **HEADROOM OK** — 62.5 % sits mid-range, so H2 can fall and H3 can hold.
  `PROXY_LENGTH_SATURATION` set to **260** (p90) per the runbook. Numbers parse (74 % of responses
  carry one); grounding precision is at 0 or 1 in 32 % of rows — usable as secondary, not headline.
- **Open**: proxy weighting (still 0.5/0.5); base→REFUTED bias on INSUFFICIENT claims is worth a
  sentence in the writeup — the collapse H2 predicts already has a direction at base.
- **Earlier the same day**: an Ollama pilot on the Mac mini (`qwen3.5:4b`, 3 rows) stalled the
  machine with four sessions open — local inference dropped as a route (runbook v2.1).
