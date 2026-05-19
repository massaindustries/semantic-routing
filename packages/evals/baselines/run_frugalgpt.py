#!/usr/bin/env python3
"""FrugalGPT zero-shot run su Dataset A.

Strategy:
- Carica DistilBERT scorer pretrained dal repo FrugalGPT (HEADLINES_Model2024).
- Cascade qwen -> ds4 -> kimi usando cached responses dal subset verbose.
- Per ogni query: scorer.predict(query+response_i); se score > 1 - threshold_i, accetta modello i.
- Output: modello scelto + #call cascade.

NOTA: scorer pretrained su (HEADLINES_questions, gpt-3.5_responses). Su nostro setup
(Dataset A queries, qwen/ds4/kimi responses) è transfer learning charity-mode.
Risultato è weak baseline che documenta come FrugalGPT non sia plug-and-play.

Threshold da cascade_strategy.json del paper (HEADLINES setup). Vedi STRATEGY_PATH.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, DistilBertTokenizerFast

os.environ.setdefault("HF_TOKEN", Path("/root/.hf_token_regolo").read_text().strip()
                     if Path("/root/.hf_token_regolo").exists() else "")

from datasets import load_dataset

REPO = "massaindustries/dataset-A-routing"
OUT = Path("/root/forkGO/external_comparison/predictions/frugalgpt.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)

STRATEGY_PATH = Path("/root/forkGO/external_comparison/FrugalGPT/strategy/HEADLINES_Model2024")
SCORER_PATH = STRATEGY_PATH / "openaichat/gpt-4o-mini"  # charity: 1 scorer per tutti (cheapest, most generic)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOKENIZER = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

CASCADE = ["qwen", "ds4", "kimi"]
# Budget point dal cascade_strategy.json del paper. Charity: usiamo budget medio
# che mappa a thres ~[0.32, 0.86, 1.0] => promuove cascade utilizzo bilanciato.
BUDGET_KEY = "0.00022263157894736844"


def scorer_text(query: str, response: str) -> str:
    """Format from FrugalGPT scoring.py: 'Q:<query> <response>'."""
    return f"Q: {query}\nA: {response}"


def load_scorer():
    """Carica DistilBERT pretrained dal modello di reference (gpt-4o-mini)."""
    print(f"[scorer] loading from {SCORER_PATH}")
    model = AutoModelForSequenceClassification.from_pretrained(str(SCORER_PATH))
    model = model.to(DEVICE).eval()
    return model


def predict_score(model, query: str, response: str) -> float:
    """Probability of correctness (class 1)."""
    text = scorer_text(query, response)
    enc = TOKENIZER(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc)
    probs = F.softmax(out.logits, dim=-1).cpu().numpy()[0]
    return float(probs[1] if len(probs) > 1 else probs[0])


def load_cascade_thresholds() -> dict[str, float]:
    """Estrae thresholds dal cascade_strategy.json budget point BUDGET_KEY.
    Mappa thres_list paper (gpt-4o-mini, llama70b, gpt-4-turbo) -> nostri (qwen, ds4, kimi)."""
    strat_file = STRATEGY_PATH / "cascade_strategy.json"
    data = json.loads(strat_file.read_text())
    budget = data["budget"][BUDGET_KEY]
    ths = budget["thres_list"]
    mapped = {CASCADE[i]: ths[i] for i in range(len(CASCADE) - 1)}
    print(f"[strategy] budget={BUDGET_KEY} thresholds(paper)={ths} -> mapped={mapped}")
    return mapped


def main():
    done_qids = set()
    if OUT.exists():
        with OUT.open() as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done_qids.add(rec["query_id"])
                except Exception:
                    pass
        print(f"[resume] {len(done_qids)} rows already in {OUT}")

    ds = load_dataset(REPO, "verbose", split="train")
    ds = ds.filter(lambda r: r["query_id"] != "_schema_anchor")
    print(f"[load] {len(ds)} verbose rows")

    scorer = load_scorer()
    thresholds = load_cascade_thresholds()
    print(f"[cascade] order={CASCADE} thresholds={thresholds}")

    t0 = time.time()
    n_new = 0
    with OUT.open("a") as fout:
        for i, row in enumerate(ds):
            qid = row["query_id"]
            if qid in done_qids:
                continue
            query = row["query"] or ""
            selected = None
            calls = 0
            scores = {}
            cumulative_cost = 0.0
            cumulative_in_tok = 0
            cumulative_out_tok = 0
            router_latency_ms = 0.0  # time spent in scorer + cascade decision logic (NO model calls)
            try:
                t_router = time.perf_counter()
                for j, m in enumerate(CASCADE):
                    response = row.get(f"{m}_response") or ""
                    calls += 1
                    cumulative_cost += float(row.get(f"{m}_cost_usd") or 0)
                    cumulative_in_tok += int(row.get(f"{m}_input_tokens") or 0)
                    cumulative_out_tok += int(row.get(f"{m}_completion_tokens") or 0)
                    if not response.strip():
                        scores[m] = None
                        continue
                    s = predict_score(scorer, query, response)
                    scores[m] = s
                    if j == len(CASCADE) - 1:
                        selected = m
                        break
                    th = thresholds.get(m, 0.5)
                    if s >= 1 - th:
                        selected = m
                        break
                if selected is None:
                    selected = CASCADE[-1]
                router_latency_ms = (time.perf_counter() - t_router) * 1000
                err = None
            except Exception as e:
                err = str(e)

            rec = {
                "query_id": qid,
                "dimension": row.get("dimension"),
                "frugal_selected": selected,
                "frugal_calls": calls,
                "frugal_router_latency_ms": router_latency_ms,
                "frugal_scores": scores,
                "frugal_cumulative_cost_usd": cumulative_cost,
                "frugal_cumulative_input_tokens": cumulative_in_tok,
                "frugal_cumulative_output_tokens": cumulative_out_tok,
                "error": err,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            n_new += 1

            if n_new % 50 == 0:
                elapsed = time.time() - t0
                rate = n_new / max(elapsed, 1e-9)
                eta = (len(ds) - len(done_qids) - n_new) / max(rate, 1e-9)
                print(f"[{n_new}] rate={rate:.2f}/s, eta={eta/60:.1f}min")

    print(f"[done] processed {n_new} rows in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
