"""Build M4 splits from the raw SciFact release.

SciFact's HF dataset is a legacy loading script, which modern `datasets` will not
run, so we read the canonical tarball directly:

    curl -sSLO https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz
    tar xzf data.tar.gz          # -> data/{corpus,claims_train,claims_dev}.jsonl

Usage:
    python3 -I m4/build_splits.py <raw_data_dir> <out_dir>

Writes three files, disjoint by construction:
    sft_train.jsonl     SciFact train, for SFT
    dpo_prompts.jsonl   SciFact train, prompts only (DPO responses are sampled later)
    eval_scifact.jsonl  SciFact dev, label-balanced, for measurement

BioLit's 12 hand-authored pairs stay where they are and are never written here:
they are the held-out perturbation probe, reported separately.

Licence: SciFact is CC-BY-NC-2.0. Fine for a portfolio writeup, not for a product.
"""

import json
import random
import sys
from pathlib import Path

# SciFact labels -> the verdict vocabulary the BioLit prompt already asks for, so
# one deterministic parser serves both datasets.
VERDICT = {"SUPPORT": "SUPPORTED", "CONTRADICT": "REFUTED", None: "INSUFFICIENT EVIDENCE"}

# ponytail: drop the long tail rather than add a chunking strategy. SciFact
# abstracts run to 367 sentences (mean 8.9); past ~40 the prompt crowds a 4B
# model's context once the rubric and reasoning are added. Revisit only if the
# retained set turns out too small, which at these counts it does not.
MAX_ABSTRACT_SENTENCES = 40

EVAL_PER_CLASS = 40  # 120 eval claims, balanced three ways
SEED = 20261020


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f]


def flatten(claims: list[dict], corpus: dict[int, dict]) -> list[dict]:
    """One row per (claim, abstract). Gold rationale sentences come along.

    A claim with evidence is paired with its evidence document. A claim with no
    evidence is paired with the document that cited it, which is what makes it an
    honest INSUFFICIENT EVIDENCE case rather than an unrelated abstract.
    """
    out = []
    for c in claims:
        evidence = c.get("evidence") or {}
        if evidence:
            targets = [(int(doc_id), items) for doc_id, items in evidence.items()]
        else:
            cited = c.get("cited_doc_ids") or []
            targets = [(int(cited[0]), [])] if cited else []

        for doc_id, items in targets:
            doc = corpus.get(doc_id)
            if doc is None or len(doc["abstract"]) > MAX_ABSTRACT_SENTENCES:
                continue

            # A doc can carry several rationales under one label; union the
            # sentence indices and keep the label. Mixed labels on one doc do not
            # occur in SciFact, so take the first.
            labels = {it["label"] for it in items}
            assert len(labels) <= 1, f"claim {c['id']} doc {doc_id} has {labels}"
            rationale = sorted({s for it in items for s in it["sentences"]})

            out.append(
                {
                    "case_id": f"scifact-{c['id']}-{doc_id}",
                    "claim": c["claim"],
                    "doc_id": doc_id,
                    "title": doc["title"],
                    # Kept as a sentence list: the gold rationale is sentence
                    # INDICES, so flattening to a string here would throw away the
                    # only thing that makes span grounding deterministic.
                    "abstract_sentences": doc["abstract"],
                    "abstract": " ".join(doc["abstract"]),
                    "verdict": VERDICT[next(iter(labels), None)],
                    "rationale_sentences": rationale,
                }
            )
    return out


def balance(rows: list[dict], per_class: int, rng: random.Random) -> list[dict]:
    """Equal numbers of each verdict, so accuracy is not a prior in disguise."""
    by_verdict: dict[str, list[dict]] = {}
    for r in rows:
        by_verdict.setdefault(r["verdict"], []).append(r)

    picked = []
    for verdict in sorted(by_verdict):
        pool = sorted(by_verdict[verdict], key=lambda r: r["case_id"])
        rng.shuffle(pool)
        if len(pool) < per_class:
            print(f"  warning: {verdict} has only {len(pool)} rows, wanted {per_class}")
        picked.extend(pool[:per_class])
    rng.shuffle(picked)
    return picked


def write(path: Path, rows: list[dict], keys: list[str] | None = None) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps({k: r[k] for k in keys} if keys else r) + "\n")
    print(f"  {path.name}: {len(rows)} rows")


def main() -> None:
    raw, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    corpus = {int(d["doc_id"]): d for d in load_jsonl(raw / "corpus.jsonl")}
    train = flatten(load_jsonl(raw / "claims_train.jsonl"), corpus)
    dev = flatten(load_jsonl(raw / "claims_dev.jsonl"), corpus)

    # Guard the thing that would quietly invalidate the whole experiment: a claim
    # or an abstract appearing on both sides of the train/eval boundary.
    #
    # SciFact's own splits are not clean on claim text: 2 claims appear in both
    # claims_train and claims_dev under different ids (same text, same cited doc,
    # same label -- duplicate annotations). Drop them from the EVAL side, before
    # balancing, so training data is untouched and the balance still fills.
    train_claims = {r["claim"] for r in train}
    train_docs = {r["doc_id"] for r in train}
    dev_clean = [r for r in dev if r["claim"] not in train_claims]
    dropped = len(dev) - len(dev_clean)
    if dropped:
        print(f"  dropped {dropped} eval row(s) whose claim text is also in train")

    eval_rows = balance(dev_clean, EVAL_PER_CLASS, rng)
    assert not [r for r in eval_rows if r["claim"] in train_claims], "claim leak survived"
    leaked_docs = sorted({r["doc_id"] for r in eval_rows} & train_docs)
    if leaked_docs:
        # Abstract reuse across splits is not contamination of the labels, but it
        # does make claim-level CIs too narrow, so the bootstrap must cluster by
        # doc_id. Recorded here so the reporting step cannot forget.
        print(f"  note: {len(leaked_docs)} abstracts appear in both splits "
              f"-> cluster the bootstrap by doc_id")

    print("writing splits:")
    write(out / "sft_train.jsonl", train)
    write(out / "dpo_prompts.jsonl", train, keys=["case_id", "claim", "abstract", "verdict"])
    write(out / "eval_scifact.jsonl", eval_rows)

    counts: dict[str, int] = {}
    for r in eval_rows:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    print(f"  eval balance: {counts}")


if __name__ == "__main__":
    main()
