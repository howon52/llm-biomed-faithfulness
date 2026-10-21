"""Deterministic M4 report: metrics per stage, confusion matrices, clustered
bootstrap CIs, the pilot's four answers, and the proxy-vs-faithfulness plot.

    uv run python m4/report.py runs/pilot/base_eval.jsonl                  # pilot
    uv run python m4/report.py --labels base,sft,dpo \
        runs/gpu/base.jsonl runs/gpu/sft.jsonl runs/gpu/dpo.jsonl --plot runs/gpu/m4.png

Every statistic is a ratio of two sums over responses (numerator, denominator), so
the clustered bootstrap is one matrix multiply per resample: cluster = doc_id, 2000
resamples, 95 % percentile interval. Rows sharing an abstract are resampled
together, which is what the 77 abstracts shared between train and dev require.

Unparsed responses are reported as a parse rate and EXCLUDED from accuracy and
recall, never averaged in as wrong answers.
"""

import argparse
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np

# Runnable as `python m4/x.py` from the repo root, as the runbook does.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from m4.metrics import VERDICTS, all_metrics

N_BOOT = 2000
SEED = 20261020
GATE_LO, GATE_HI = 0.15, 0.95  # headroom for H2, see M4_RUNBOOK "Gate"

# Each statistic is (numerator, denominator) for one response's metric dict.
# A None metric contributes (0, 0) so it drops out of the ratio.
STATS = {
    "truncated": lambda m: (m["truncated"], 1),
    "parse_rate": lambda m: (m["parsed_verdict"] is not None, 1),
    "accuracy": lambda m: (m["verdict_correct"], m["parsed_verdict"] is not None),
    **{
        f"recall_{v.split()[0].lower()}": (
            lambda m, v=v: (m["gold_verdict"] == v and m["parsed_verdict"] == v,
                            m["gold_verdict"] == v and m["parsed_verdict"] is not None)
        )
        for v in VERDICTS
    },
    "numeric_fabrication": lambda m: (m["n_unsupported_numbers"], m["n_numbers"]),
    "grounding_precision": lambda m: _opt(m["grounding_precision"]),
    "grounding_recall": lambda m: _opt(m["grounding_recall"]),
    "n_words": lambda m: (m["n_words"], 1),
    "hedge_rate": lambda m: (m["hedge_rate"], 1),
    "assertive_rate": lambda m: (m["assertive_rate"], 1),
    "proxy": lambda m: (m["proxy"], 1),
}


def _opt(x):
    return (0.0, 0) if x is None else (x, 1)


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def as_metric_row(row: dict) -> dict:
    """Make a dataset row usable by all_metrics.

    SciFact rows already carry doc_id, abstract_sentences and rationale_sentences.
    BioLit rows (data/claim_abstract_pairs.jsonl) have none of them: cluster on
    pmid, split the abstract on sentence ends, and report grounding as None.
    """
    r = dict(row)
    r.setdefault("doc_id", r.get("pmid", r["case_id"]))
    r.setdefault("abstract_sentences", re.split(r"(?<=[.!?])\s+", r["abstract"].strip()))
    r.setdefault("rationale_sentences", [])
    return r


def score(generations: list[dict], data: dict[str, dict]) -> list[dict]:
    missing = {g["case_id"] for g in generations} - set(data)
    if missing:
        sys.exit(f"{len(missing)} case_ids not in --data, e.g. {sorted(missing)[:3]}")
    out = []
    for g in generations:
        m = all_metrics(g["response"], data[g["case_id"]])
        # Cut off by the token cap: no verdict line exists, so whatever the parser
        # found was a mention mid-reasoning. Unparsed, not wrong.
        m["truncated"] = bool(g.get("truncated", False))
        if m["truncated"]:
            m["parsed_verdict"], m["verdict_correct"] = None, False
        out.append(m)
    return out


def clustered_bootstrap(metrics: list[dict], n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """{stat: (point, lo, hi)} with clusters = doc_id resampled as units."""
    names = list(STATS)
    clusters: dict = {}
    for m in metrics:
        clusters.setdefault(m["doc_id"], []).append(
            [STATS[s](m) for s in names]
        )
    # (n_clusters, n_stats, 2): per-cluster numerator and denominator sums.
    C = np.array([np.sum(np.array(rows, dtype=float), axis=0) for rows in clusters.values()])
    tot = C.sum(axis=0)
    point = _ratio(tot)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(C), size=(n_boot, len(C)))
    boot = np.stack([_ratio(C[i].sum(axis=0)) for i in idx])  # (n_boot, n_stats)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN column = stat n/a
        lo, hi = np.nanpercentile(boot, [2.5, 97.5], axis=0)
    return {s: (point[k], lo[k], hi[k]) for k, s in enumerate(names)}


def _ratio(sums: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(sums[:, 1] > 0, sums[:, 0] / sums[:, 1], np.nan)


def confusion(metrics: list[dict]) -> dict[str, dict[str, int]]:
    labels = [*VERDICTS, None]
    table = {g: {str(p): 0 for p in labels} for g in VERDICTS}
    for m in metrics:
        table[m["gold_verdict"]][str(m["parsed_verdict"])] += 1
    return table


def length_percentiles(metrics: list[dict]) -> dict[str, float]:
    n = np.array([m["n_words"] for m in metrics])
    return {f"p{p}": float(np.percentile(n, p)) for p in (10, 50, 90)} | {"max": float(n.max())}


# --- printing ----------------------------------------------------------------


def fmt(point, lo, hi, pct: bool) -> str:
    if np.isnan(point):
        return "n/a"
    if pct:
        return f"{100*point:5.1f} [{100*lo:5.1f}, {100*hi:5.1f}]"
    return f"{point:5.2f} [{lo:5.2f}, {hi:5.2f}]"


def print_table(labels: list[str], cis: list[dict]) -> None:
    pct = {"truncated", "parse_rate", "accuracy", "numeric_fabrication", "grounding_precision",
           "grounding_recall"} | {s for s in STATS if s.startswith("recall_")}
    w = 24
    print(f"{'stat':<22}" + "".join(f"{l:>{w}}" for l in labels))
    for s in STATS:
        print(f"{s:<22}" + "".join(f"{fmt(*ci[s], s in pct):>{w}}" for ci in cis))
    print("  (% for rates; mean [95% CI], bootstrap clustered by doc_id)")


def print_confusion(table: dict) -> None:
    cols = [*VERDICTS, "None"]
    print(f"{'gold \\ parsed':<24}" + "".join(f"{c[:12]:>14}" for c in cols))
    for g in VERDICTS:
        print(f"{g:<24}" + "".join(f"{table[g][c]:>14}" for c in cols))


def print_pilot_answers(ci: dict, metrics: list[dict]) -> None:
    """The four questions the free pilot exists to answer (M4_RUNBOOK)."""
    parse = ci["parse_rate"][0]
    rec = ci["recall_insufficient"]
    pcts = length_percentiles(metrics)
    prec = [m["grounding_precision"] for m in metrics if m["grounding_precision"] is not None]
    degenerate = np.mean([p in (0.0, 1.0) for p in prec]) if prec else float("nan")
    with_numbers = np.mean([m["n_numbers"] > 0 for m in metrics])

    print("\nPilot answers")
    print(f"  1. format:   parse rate {100*parse:.1f} %, truncated {100*ci['truncated'][0]:.1f} %")
    verdict = ("HEADROOM OK" if GATE_LO <= rec[0] <= GATE_HI
               else "NO HEADROOM -- change design before renting a GPU")
    print(f"  2. H2 room:  INSUFFICIENT recall {100*rec[0]:.1f} % "
          f"[{100*rec[1]:.1f}, {100*rec[2]:.1f}]  ->  {verdict} (gate {GATE_LO}..{GATE_HI})")
    print(f"  3. length:   reasoning words p10/p50/p90/max = "
          f"{pcts['p10']:.0f}/{pcts['p50']:.0f}/{pcts['p90']:.0f}/{pcts['max']:.0f}"
          f"  ->  PROXY_LENGTH_SATURATION ~ {pcts['p90']:.0f} (p90)")
    print(f"  4. metrics:  grounding precision at exactly 0 or 1 in {100*degenerate:.0f} % of "
          f"scored rows; {100*with_numbers:.0f} % of responses contain a number")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("generations", nargs="+", type=Path, help="one jsonl per stage")
    ap.add_argument("--data", type=Path, default=Path("data/m4/eval_scifact.jsonl"))
    ap.add_argument("--labels", help="comma-separated stage names; default = file stems")
    ap.add_argument("--boot", type=int, default=N_BOOT)
    ap.add_argument("--plot", type=Path, help="write proxy-vs-INSUFFICIENT-recall png")
    ap.add_argument("--json", type=Path, help="dump all numbers for the writeup")
    args = ap.parse_args(argv)

    labels = args.labels.split(",") if args.labels else [p.stem for p in args.generations]
    assert len(labels) == len(args.generations), "one label per generation file"
    data = {r["case_id"]: as_metric_row(r) for r in load_jsonl(args.data)}

    stages = []
    for label, path in zip(labels, args.generations):
        metrics = score(load_jsonl(path), data)
        stages.append({"label": label, "n": len(metrics), "metrics": metrics,
                       "ci": clustered_bootstrap(metrics, args.boot),
                       "confusion": confusion(metrics),
                       "length": length_percentiles(metrics)})

    print(f"{len(data)} data rows; responses per stage: "
          + ", ".join(f"{s['label']}={s['n']}" for s in stages))
    print()
    print_table(labels, [s["ci"] for s in stages])
    for s in stages:
        print(f"\nConfusion -- {s['label']}")
        print_confusion(s["confusion"])
    if len(stages) == 1:
        print_pilot_answers(stages[0]["ci"], stages[0]["metrics"])

    if args.json:
        out = [{k: v for k, v in s.items() if k != "metrics"} for s in stages]
        for s in out:
            s["ci"] = {k: [None if np.isnan(x) else float(x) for x in v] for k, v in s["ci"].items()}
        args.json.write_text(json.dumps(out, indent=1))
        print(f"\nwrote {args.json}")
    if args.plot:
        plot(stages, args.plot)
        print(f"wrote {args.plot}")


def plot(stages: list[dict], out: Path) -> None:
    """Proxy (what DPO optimised) against INSUFFICIENT recall (what it cost)."""
    import matplotlib  # dev dependency; only needed here

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = range(len(stages))
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax2 = ax1.twinx()
    for ax, stat, color, label in (
        (ax1, "proxy", "tab:red", "proxy reward (DPO target)"),
        (ax2, "recall_insufficient", "tab:blue", "INSUFFICIENT EVIDENCE recall"),
    ):
        p, lo, hi = zip(*(s["ci"][stat] for s in stages))
        p, lo, hi = map(np.array, (p, lo, hi))
        ax.errorbar(x, p, yerr=[p - lo, hi - p], color=color, marker="o", capsize=4, label=label)
        ax.set_ylabel(label, color=color)
    ax1.set_xticks(list(x), [s["label"] for s in stages])
    ax2.set_ylim(0, 1)
    fig.suptitle("M4: proxy up, uncertainty class down?")
    fig.tight_layout()
    fig.savefig(out, dpi=150)


if __name__ == "__main__":
    main()
