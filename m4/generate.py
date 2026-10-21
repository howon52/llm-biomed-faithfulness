"""Sample responses from a model for every row of a claim/abstract jsonl.

One output schema for every backend, so report.py, make_pairs.py and the SFT
selection step never care where a response came from:

    {"case_id": ..., "i": <sample index>, "response": ..., "model": ..., "tag": ...}

Backends:
    openai  -- any OpenAI-compatible chat endpoint, stdlib only. The pilot and
               every stage run this against `vllm serve` on the Lambda box through
               an SSH tunnel (no auth); the Hugging Face router (Nscale serves the
               exact model) is the fallback, with the key in --api-key-env.
    ollama  -- local HTTP API. Kept for when the Mac is otherwise idle.
    vllm    -- offline vllm.LLM on the GPU box; optional LoRA adapter for the
               SFT / DPO stages. Imported lazily so this file runs on a laptop.

The prompt is the M1 prompt, imported rather than copied: the pilot, the GPU run
and the BioLit probe must all see exactly what the rubric runs saw.

Resumable: rows whose case_id already has `--n` samples in --out are skipped, so a
30-minute Ollama run that dies at row 80 restarts at row 80.

    uv run python m4/generate.py --backend openai --base-url http://localhost:8000/v1 \
        --model Qwen/Qwen3-4B-Instruct-2507 \
        --input data/m4/eval_scifact.jsonl --n 1 --temperature 0.7 \
        --out runs/pilot/base_eval.jsonl
"""

import argparse
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Runnable as `python m4/x.py` from the repo root, as the runbook does.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biolit_faithfulness_eval import RESEARCHER_PROMPT, _format_input

MAX_NEW_TOKENS = 2048  # 4B models ramble; qwen3.5:4b hit 1024 on 1 of 6 smoke samples


def messages_for(row: dict) -> list[dict]:
    return [
        {"role": "system", "content": RESEARCHER_PROMPT},
        {"role": "user", "content": _format_input(row["abstract"], row["claim"])},
    ]


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def already_done(out: Path, model: str) -> dict[str, int]:
    """case_id -> number of samples already written. Refuses to resume a file
    sampled from a different model: the aborted Ollama pilot left rows at this
    path that must not be mixed into the router run."""
    counts: dict[str, int] = {}
    if out.exists():
        for r in load_jsonl(out):
            if r.get("model", model) != model:
                sys.exit(f"{out} holds samples from {r['model']!r}, not {model!r}; "
                         "delete it or pick another --out")
            counts[r["case_id"]] = counts.get(r["case_id"], 0) + 1
    return counts


# --- backends ----------------------------------------------------------------


class Ollama:
    def __init__(self, model: str, host: str, think: bool | None):
        self.model, self.host, self.think = model, host.rstrip("/"), think

    def sample(self, messages: list[dict], n: int, temperature: float, seed: int
               ) -> list[tuple[str, bool]]:
        """(text, truncated) per sample. A response cut off by the token cap has no
        verdict line, and the parser would pick up a stray mention mid-reasoning;
        report.py treats truncated as unparsed."""
        out = []
        for k in range(n):
            body = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                # Distinct seed per sample: Ollama's sampler is seeded, so n
                # identical requests would be n identical responses.
                "options": {"temperature": temperature, "seed": seed + k,
                            "num_predict": MAX_NEW_TOKENS},
            }
            if self.think is not None:
                body["think"] = self.think
            req = urllib.request.Request(
                f"{self.host}/api/chat", data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=600) as resp:
                r = json.load(resp)
            out.append((r["message"]["content"], r.get("done_reason") == "length"))
        return out


class OpenAICompat:
    """Chat completions over HTTP. One request per sample: `n` and `seed` are not
    honoured by every provider, temperature is."""

    def __init__(self, model: str, base_url: str, api_key: str):
        self.model, self.base_url, self.api_key = model, base_url.rstrip("/"), api_key

    def sample(self, messages: list[dict], n: int, temperature: float, seed: int
               ) -> list[tuple[str, bool]]:
        out = []
        for k in range(n):
            body = {"model": self.model, "messages": messages, "temperature": temperature,
                    "max_tokens": MAX_NEW_TOKENS, "seed": seed + k}
            headers = {"Content-Type": "application/json"}
            if self.api_key:  # vllm serve has no auth; the HF router needs a token
                headers["Authorization"] = f"Bearer {self.api_key}"
            req = urllib.request.Request(
                f"{self.base_url}/chat/completions", data=json.dumps(body).encode(),
                headers=headers,
            )
            with urllib.request.urlopen(req, timeout=600) as resp:
                choice = json.load(resp)["choices"][0]
            out.append((choice["message"]["content"], choice.get("finish_reason") == "length"))
        return out


class VLLM:
    def __init__(self, model: str, adapter: str | None, seed: int):
        from vllm import LLM, SamplingParams  # noqa: F401 -- GPU box only
        from vllm.lora.request import LoRARequest

        self._SamplingParams = SamplingParams
        self.llm = LLM(model=model, enable_lora=adapter is not None, max_lora_rank=16,
                       seed=seed, dtype="bfloat16")
        self.lora = LoRARequest("adapter", 1, adapter) if adapter else None

    def sample_batch(self, convs: list[list[dict]], n: int, temperature: float
                     ) -> list[list[tuple[str, bool]]]:
        params = self._SamplingParams(n=n, temperature=temperature, max_tokens=MAX_NEW_TOKENS)
        outs = self.llm.chat(convs, params, lora_request=self.lora, use_tqdm=True)
        return [[(o.text, o.finish_reason == "length") for o in r.outputs] for r in outs]


# --- driver ------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--backend", choices=["openai", "ollama", "vllm"], required=True)
    ap.add_argument("--model", required=True, help="Ollama tag or HF id")
    ap.add_argument("--adapter", help="vllm only: path to a LoRA adapter (SFT / DPO stage)")
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=1, help="samples per row")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=20261020)
    ap.add_argument("--limit", type=int, help="first N rows only (smoke test)")
    ap.add_argument("--tag", default="", help="free label written to every line, e.g. base/sft/dpo")
    ap.add_argument("--host", default="http://localhost:11434", help="ollama only")
    ap.add_argument("--base-url", default="http://localhost:8000/v1", help="openai only")
    ap.add_argument("--api-key-env", default="HF_TOKEN",
                    help="openai only: env var holding the key; unset = no auth header (vllm serve)")
    ap.add_argument("--workers", type=int, default=8, help="openai only: concurrent requests")
    ap.add_argument("--no-think", dest="think", action="store_const", const=False, default=None,
                    help="ollama only: disable thinking on models that have it (qwen3.5). "
                         "Omit for non-thinking models such as qwen3:*-instruct-2507.")
    args = ap.parse_args(argv)

    rows = load_jsonl(args.input)[: args.limit]
    done = already_done(args.out, args.model)
    todo = [r for r in rows if done.get(r["case_id"], 0) < args.n]
    print(f"{len(rows)} rows, {len(rows) - len(todo)} already complete, {len(todo)} to go",
          file=sys.stderr)
    if not todo:
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def record(row: dict, i: int, text: str, truncated: bool) -> dict:
        return {"case_id": row["case_id"], "i": i, "response": text, "truncated": truncated,
                "model": args.model, "adapter": args.adapter, "tag": args.tag,
                "temperature": args.temperature}

    t0 = time.time()
    with args.out.open("a") as f:
        if args.backend in ("ollama", "openai"):
            if args.backend == "ollama":
                be, workers = Ollama(args.model, args.host, args.think), 1
            else:
                be = OpenAICompat(args.model, args.base_url, os.environ.get(args.api_key_env, ""))
                workers = args.workers

            def one(j_row):
                j, row = j_row
                have = done.get(row["case_id"], 0)
                # Seed offset by row index so two rows never share a sampler seed.
                return row, have, be.sample(messages_for(row), args.n - have, args.temperature,
                                            args.seed + 1000 * j + have)

            with ThreadPoolExecutor(workers) as pool:
                for j, (row, have, texts) in enumerate(pool.map(one, enumerate(todo, 1)), 1):
                    for k, (text, trunc) in enumerate(texts):
                        f.write(json.dumps(record(row, have + k, text, trunc)) + "\n")
                    f.flush()
                    if j % 10 == 0 or j == len(todo):
                        el = time.time() - t0
                        print(f"  {j}/{len(todo)}  {el/60:.1f} min elapsed, "
                              f"~{el/j*(len(todo)-j)/60:.1f} min left", file=sys.stderr)
        else:
            be = VLLM(args.model, args.adapter, args.seed)
            # ponytail: resume at row granularity only -- a partially sampled row is
            # resampled in full. vllm batches the whole set, so partial rows are rare.
            todo = [r for r in todo if done.get(r["case_id"], 0) == 0]
            for row, texts in zip(todo, be.sample_batch([messages_for(r) for r in todo],
                                                        args.n, args.temperature)):
                for k, (text, trunc) in enumerate(texts):
                    f.write(json.dumps(record(row, k, text, trunc)) + "\n")
    print(f"wrote {args.out}  ({(time.time()-t0)/60:.1f} min)", file=sys.stderr)


if __name__ == "__main__":
    main()
