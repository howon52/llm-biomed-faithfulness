"""Tests for m4/generate.py and m4/report.py. No model, no network."""

import json
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from m4 import generate, report
from m4.metrics import all_metrics

ROW = {
    "case_id": "c1",
    "doc_id": 1,
    "claim": "Dose reduces CRP.",
    "abstract_sentences": ["The 50 mg dose cut CRP by 32%.", "Placebo did not."],
    "abstract": "The 50 mg dose cut CRP by 32%. Placebo did not.",
    "verdict": "SUPPORTED",
    "rationale_sentences": [0],
}


def write_jsonl(path: Path, rows) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


# --- generate ----------------------------------------------------------------

def test_prompt_is_the_m1_prompt_with_abstract_before_claim():
    msgs = generate.messages_for(ROW)
    assert msgs[0]["content"] == generate.RESEARCHER_PROMPT
    assert msgs[1]["content"].index("Abstract:") < msgs[1]["content"].index("Claim to evaluate:")


def test_ollama_sends_a_distinct_seed_per_sample_and_omits_think_by_default():
    bodies = []

    def fake_urlopen(req, timeout):
        bodies.append(json.loads(req.data))
        return mock.MagicMock(__enter__=lambda s: s, __exit__=lambda *a: None,
                              read=lambda: json.dumps({"message": {"content": "ok\nSUPPORTED"}}).encode())

    with mock.patch.object(generate.urllib.request, "urlopen", fake_urlopen):
        out = generate.Ollama("m", "http://h", think=None).sample(
            generate.messages_for(ROW), n=3, temperature=0.7, seed=5)
    assert out == [("ok\nSUPPORTED", False)] * 3
    assert [b["options"]["seed"] for b in bodies] == [5, 6, 7]
    assert all("think" not in b for b in bodies)
    assert all(b["options"]["temperature"] == 0.7 for b in bodies)


def test_openai_backend_sends_bearer_key_and_flags_length_finish():
    bodies = []

    def fake_urlopen(req, timeout):
        bodies.append((req.get_header("Authorization"), json.loads(req.data)))
        return mock.MagicMock(__enter__=lambda s: s, __exit__=lambda *a: None,
                              read=lambda: json.dumps({"choices": [{"message": {"content": "cut"},
                                                                     "finish_reason": "length"}]}).encode())

    with mock.patch.object(generate.urllib.request, "urlopen", fake_urlopen):
        out = generate.OpenAICompat("Qwen/x:nscale", "https://h/v1/", "tok").sample(
            generate.messages_for(ROW), n=1, temperature=0.7, seed=1)
    assert out == [("cut", True)]
    assert bodies[0][0] == "Bearer tok"
    assert bodies[0][1]["model"] == "Qwen/x:nscale" and bodies[0][1]["max_tokens"] == generate.MAX_NEW_TOKENS


def test_resume_refuses_a_file_sampled_from_another_model(tmp_path):
    out = write_jsonl(tmp_path / "out.jsonl", [{"case_id": "c1", "i": 0, "response": "x", "model": "old"}])
    with pytest.raises(SystemExit, match="old"):
        generate.already_done(out, "new")


def test_resume_skips_rows_that_already_have_n_samples(tmp_path):
    inp = write_jsonl(tmp_path / "in.jsonl", [ROW, {**ROW, "case_id": "c2"}])
    out = write_jsonl(tmp_path / "out.jsonl", [{"case_id": "c1", "i": 0, "response": "x"}])
    calls = []

    class FakeOllama:
        def __init__(self, *a):
            pass

        def sample(self, messages, n, temperature, seed):
            calls.append(n)
            return [("r\nREFUTED", False)] * n

    with mock.patch.object(generate, "Ollama", FakeOllama):
        generate.main(["--backend", "ollama", "--model", "m", "--input", str(inp),
                       "--out", str(out), "--n", "1", "--tag", "base"])
    rows = generate.load_jsonl(out)
    assert calls == [1], "only the missing row is sampled"
    assert [r["case_id"] for r in rows] == ["c1", "c2"]
    assert rows[1] == {"case_id": "c2", "i": 0, "response": "r\nREFUTED", "truncated": False, "model": "m",
                       "adapter": None, "tag": "base", "temperature": 0.7}


# --- report ------------------------------------------------------------------

def metric(gold, parsed, doc_id=1, **over):
    text = "The 50 mg dose cut CRP by 32%.\n" + (parsed or "no verdict here")
    m = all_metrics(text, {**ROW, "verdict": gold, "doc_id": doc_id})
    m["truncated"] = False
    m.update(over)
    return m


def test_recall_and_accuracy_exclude_unparsed_responses():
    ms = [metric("SUPPORTED", "SUPPORTED"), metric("SUPPORTED", None),
          metric("INSUFFICIENT EVIDENCE", "SUPPORTED")]
    ci = report.clustered_bootstrap(ms, n_boot=10)
    assert ci["parse_rate"][0] == pytest.approx(2 / 3)
    assert ci["accuracy"][0] == pytest.approx(1 / 2)
    assert ci["recall_supported"][0] == 1.0
    assert ci["recall_insufficient"][0] == 0.0
    assert np.isnan(ci["recall_refuted"][0]), "no gold REFUTED rows -> n/a, not 0"


def test_numeric_fabrication_is_pooled_over_numbers_not_averaged_over_rows():
    ms = [metric("SUPPORTED", "SUPPORTED", n_numbers=9, n_unsupported_numbers=0),
          metric("SUPPORTED", "SUPPORTED", n_numbers=1, n_unsupported_numbers=1)]
    assert report.clustered_bootstrap(ms, n_boot=10)["numeric_fabrication"][0] == pytest.approx(0.1)


def test_bootstrap_resamples_whole_clusters():
    """Two clusters, each internally homogeneous: every resample is all-correct,
    all-wrong, or exactly the point estimate, so no CI endpoint can fall
    strictly between 0 and 0.5 or between 0.5 and 1. Row-level resampling would."""
    ms = [metric("SUPPORTED", "SUPPORTED", doc_id=1)] * 5 + \
         [metric("SUPPORTED", "REFUTED", doc_id=2)] * 5
    ci = report.clustered_bootstrap(ms, n_boot=200)
    assert ci["accuracy"][0] == 0.5
    assert (ci["accuracy"][1], ci["accuracy"][2]) == (0.0, 1.0)


def test_truncated_response_counts_as_unparsed_even_if_a_verdict_word_appears():
    data = {"c1": ROW}
    ms = report.score([{"case_id": "c1", "response": "It is REFUTED or SUPPORTED, let me", "truncated": True}], data)
    assert ms[0]["parsed_verdict"] is None and ms[0]["truncated"] is True


def test_confusion_counts_unparsed_in_their_own_column():
    table = report.confusion([metric("REFUTED", None), metric("REFUTED", "REFUTED")])
    assert table["REFUTED"] == {"SUPPORTED": 0, "REFUTED": 1, "INSUFFICIENT EVIDENCE": 0, "None": 1}


def test_biolit_rows_get_doc_id_and_sentences_filled_in():
    r = report.as_metric_row({"case_id": "b1", "pmid": "123", "abstract": "One. Two? Three.",
                              "claim": "c", "verdict": "REFUTED"})
    assert r["doc_id"] == "123"
    assert r["abstract_sentences"] == ["One.", "Two?", "Three."]
    assert r["rationale_sentences"] == []


def test_report_end_to_end_prints_pilot_gate(tmp_path, capsys):
    data = write_jsonl(tmp_path / "data.jsonl", [ROW, {**ROW, "case_id": "c2", "doc_id": 2,
                                                       "verdict": "INSUFFICIENT EVIDENCE",
                                                       "rationale_sentences": []}])
    gen = write_jsonl(tmp_path / "base.jsonl", [
        {"case_id": "c1", "i": 0, "response": "The 50 mg dose cut CRP by 32%.\nSUPPORTED"},
        {"case_id": "c2", "i": 0, "response": "Nothing here settles it.\nINSUFFICIENT EVIDENCE"},
    ])
    report.main([str(gen), "--data", str(data), "--boot", "20", "--json", str(tmp_path / "o.json")])
    out = capsys.readouterr().out
    assert "Pilot answers" in out and "NO HEADROOM" in out  # recall 1.0 > gate
    assert json.loads((tmp_path / "o.json").read_text())[0]["label"] == "base"
