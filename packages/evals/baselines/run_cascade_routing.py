#!/usr/bin/env python3
"""Cascade Routing zero-shot baseline su Dataset A.

Implementazione: il framework eth-sri/cascade-routing richiede QualityComputer +
CostComputer + fit su training data. Per zero-shot transferring a Dataset A,
implementiamo la VARIANTE MINIMALE seguendo paper Dekoninck et al.:

1. Embed query con sentence-transformer (all-MiniLM-L6-v2, pretrained).
2. Per ognuno dei 3 nostri modelli, alleniamo 1 logistic regression
   query_embedding -> P(correct) usando ROUTERBENCH come training set,
   mappando i modelli RouterBench ai nostri per tier di costo:
     qwen3.5-9b   -> mistralai/mistral-7b-chat       (cheapest, ~$0.07/M)
     ds4-flash    -> gpt-3.5-turbo-1106              (medium, ~$0.50/M)
     kimi2.6      -> gpt-4-1106-preview              (strongest, ~$1.0/M)
3. Cascade routing decision: per ogni query Dataset A,
     predict P(correct|model) per i 3 modelli,
     pick model massimizzando E[utility] = P(correct) - lambda * cost,
     con lambda calibrato su RouterBench medium-budget setpoint.

Output JSONL append-only.

NOTA paper friction: questa implementazione manualmente bypassa cascade-routing
framework. Lo paper richiederebbe custom QualityComputer/CostComputer + fit chiamato
attraverso il loro Router.fit() API. Documentiamo questo come "deployment friction"
weakness vs RouteLLM/Brick che sono drop-in.
"""
from __future__ import annotations

import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression

if Path("/root/.hf_token_regolo").exists():
    os.environ["HF_TOKEN"] = Path("/root/.hf_token_regolo").read_text().strip()

REPO = "massaindustries/dataset-A-routing"
OUT = Path("/root/forkGO/external_comparison/predictions/cascade_routing.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)

ROUTERBENCH_PATH = "/root/forkGO/external_comparison/cascade-routing/data/routerbench_0shot.csv"

MODEL_MAPPING = {
    "qwen": "mistralai/mistral-7b-chat",
    "ds4": "gpt-3.5-turbo-1106",
    "kimi": "gpt-4-1106-preview",
}
COST_USD_PER_QUERY = {
    "qwen": 0.07e-6,    # $0.07/1M input
    "ds4": 0.50e-6,
    "kimi": 1.00e-6,
}
LAMBDA = 0.5  # weight cost vs quality; calibrato medio su RouterBench


def embed_queries(texts, batch=128):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embs = model.encode(texts, batch_size=batch, show_progress_bar=True, convert_to_numpy=True)
    return embs


def fit_quality_estimators(cache_path: Path):
    if cache_path.exists():
        print(f"[fit] loading cached estimators from {cache_path}")
        return pickle.loads(cache_path.read_bytes())

    print(f"[fit] loading RouterBench from {ROUTERBENCH_PATH}")
    rb = pd.read_csv(ROUTERBENCH_PATH)
    rb_queries = rb["prompt"].astype(str).tolist()
    print(f"[fit] embedding {len(rb_queries)} RouterBench prompts")
    emb = embed_queries(rb_queries)

    estimators = {}
    for our_model, rb_model in MODEL_MAPPING.items():
        if rb_model not in rb.columns:
            raise RuntimeError(f"Column {rb_model} not in RouterBench")
        y = (rb[rb_model] > 0.5).astype(int).values
        print(f"[fit] training LogReg for {our_model} ({rb_model}): pos_rate={y.mean():.3f}")
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(emb, y)
        estimators[our_model] = clf
    print(f"[fit] saving cached estimators to {cache_path}")
    cache_path.write_bytes(pickle.dumps(estimators))
    return estimators


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

    cache = Path("/root/forkGO/external_comparison/predictions/_cascade_estimators.pkl")
    estimators = fit_quality_estimators(cache)

    ds = load_dataset(REPO, "results", split="train")
    ds = ds.filter(lambda r: r["query_id"] != "_schema_anchor")
    queries = [r["query"] or "" for r in ds]
    qids = [r["query_id"] for r in ds]
    dims = [r["dimension"] for r in ds]
    print(f"[load] {len(queries)} dataset A rows")

    pending_idx = [i for i, q in enumerate(qids) if q not in done_qids]
    print(f"[pending] {len(pending_idx)} to process (per-query encode, no batch)")
    if not pending_idx:
        return

    # Load sentence-transformer ONCE (model load is amortized, not part of per-query latency)
    from sentence_transformers import SentenceTransformer
    print(f"[init] loading sentence-transformer encoder")
    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    import time as _time
    t0 = _time.time()
    with OUT.open("a") as fout:
        for k, idx in enumerate(pending_idx):
            qid = qids[idx]
            dim = dims[idx]
            q = queries[idx]
            t_router = _time.perf_counter()
            emb = encoder.encode([q], show_progress_bar=False, convert_to_numpy=True)
            scores = {m: float(estimators[m].predict_proba(emb)[0, 1]) for m in MODEL_MAPPING}
            utility = {m: scores[m] - LAMBDA * COST_USD_PER_QUERY[m] * 1e6 for m in MODEL_MAPPING}
            selected = max(utility, key=utility.get)
            router_latency_ms = (_time.perf_counter() - t_router) * 1000
            rec = {
                "query_id": qid,
                "dimension": dim,
                "cascade_selected": selected,
                "cascade_router_latency_ms": router_latency_ms,
                "cascade_p_correct": scores,
                "cascade_utility": utility,
                "cascade_calls": 1,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if (k + 1) % 500 == 0:
                fout.flush()
                rate = (k + 1) / (_time.time() - t0)
                print(f"[{k+1}/{len(pending_idx)}] rate={rate:.2f}/s")

    print(f"[done] {len(pending_idx)} rows in {(_time.time() - t0):.1f}s")


if __name__ == "__main__":
    main()
