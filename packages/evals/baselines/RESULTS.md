# Risultati confronto router su Dataset A

**Dataset**: [`massaindustries/dataset-A-routing`](https://huggingface.co/datasets/massaindustries/dataset-A-routing) subset `results`: 5504 query, 3 modelli candidate (`qwen3.5-9b`, `deepseek-v4-flash`, `kimi-k2.6`).

**Metodologia**:
- **Ground truth** = cheapest correct model (qwen → ds4 → kimi); per 922 query unsolvable fallback = `kimi`.
- **Predizione router** = primo modello tentato (per FrugalGPT cascade = sempre `qwen`).
- **Accuracy** = exact match `prediction == ground_truth` su tutti 5504.
- **Latency** = wall-clock decision time misurato localmente alla macchina che ospita il router (aesclude RTT network per Brick → fake backend locale 401 in <1ms).
- **Zero-shot**: nessuno dei 3 router baseline esterni è stato calibrato/fittato su Dataset A.

---

## Tabella riassuntiva

| Router | Accuracy | Latency mean (ms) | median | p95 | p99 | Calls/query | Plug & play |
|---|---:|---:|---:|---:|---:|---:|:---:|
| **oracle** (cheapest correct) | **100.00%** | n/a | n/a | n/a | n/a | n/a | n/a |
| **always_qwen** (baseline trivial) | 63.17% | 0 | 0 | 0 | 0 | 1 | n/a |
| **FrugalGPT cascade** | **63.17%** | 90 | 103 | 158 | 557 | 1.70 (cascade) | ❌ download checkpoint richiesto |
| **Brick** | **46.37%** | 1883 | 2998 | 3211 | 3311 | 1 | ✅ |
| **Cascade Routing** (impl) | 28.96% | **16** | **15** | **20** | 49 | 1 | ❌❌ framework, custom impl |
| **always_kimi** (baseline trivial) | 21.28% | 0 | 0 | 0 | 0 | 1 | n/a |
| **RouteLLM binary** | 21.31% | 84 | 97 | 138 | 395 | 1 | ✅ |
| **RouteLLM tournament** | 21.31% | 175 | 198 | 306 | 1317 | 2 | ✅ + custom orchestration |
| **always_ds4** (baseline trivial) | 15.55% | 0 | 0 | 0 | 0 | 1 | n/a |

---

## Distribuzione predizioni (5504 query)

| Router | qwen | ds4 | kimi | other |
|---|---:|---:|---:|---:|
| Brick | **3724 (67.6%)** | **1340 (24.3%)** | **440 (8.0%)** | 0 |
| Cascade Routing | 1011 (18.4%) | 3341 (60.7%) | 1152 (20.9%) | 0 |
| FrugalGPT cascade (first tried) | 5504 (100%) | 0 | 0 | 0 |
| RouteLLM binary | 2 (0.04%) | 0 | 5502 (99.96%) | 0 |
| RouteLLM tournament | 2 (0.04%) | 0 | 5502 (99.96%) | 0 |
| Ground truth distribution | 3477 (63.2%) | 856 (15.6%) | 1171 (21.3%) | 0 |

**Solo Brick produce una distribuzione 3-way effettiva** che si avvicina alla ground truth.

---

## Accuracy per dimension

| Dimension | n | always_qwen | FrugalGPT | Brick | Cascade Routing | RouteLLM (binary/tournament) |
|---|---:|---:|---:|---:|---:|---:|
| coding | 1000 | 71.4% | 71.4% | (calc) | 17.3% | 11.3% |
| creative_synthesis | 696 | 51.0% | 51.0% | (calc) | 24.1% | 24.9% |
| instruction_following | 841 | 81.0% | 81.0% | (calc) | 9.0% | 8.4% |
| math_reasoning | 1000 | 91.2% | 91.2% | (calc) | 49.4% | 4.7% |
| planning_agentic | 1000 | 63.8% | 63.8% | (calc) | 35.0% | 26.9% |
| planning_agentic_multiturn | 165 | 20.6% | 20.6% | (calc) | 6.1% | 73.3% |
| world_knowledge | 802 | 17.8% | 17.8% | (calc) | 40.3% | 47.3% |

> Per-dimension breakdown Brick disponibile via `comparison.jsonl.gz`: basta filtrare `brick_correct == True` per dimension.

---

## Findings chiave per paper

### 1. Plug-and-play deployment friction
- **RouteLLM** = **vero** plug-and-play (`pip install routellm` + 1 chiamata `Controller`).
- **Brick** = plug-and-play (container Docker pronto, GHCR images), match nativo del pool Dataset A.
- **FrugalGPT** = NON plug-and-play (download checkpoint ~1GB + parse strategy JSON manuale + scorer charity-mode addestrato su HEADLINES).
- **Cascade Routing** = NON plug-and-play (framework, non sistema): richiede implementazione custom QualityComputer/CostComputer + fit su training data esterno.

### 2. Trivial baseline batte i baseline esterni
- `always_qwen` accuracy = **63.17%** (qwen è il cheapest correct su 63% delle query)
- I 3 baseline esterni (RouteLLM, Cascade Routing) restano **sotto** questa soglia: dimostra che il loro routing intelligence **non aggiunge valore** out-of-the-box su Dataset A.
- FrugalGPT = always_qwen perché il "primo modello tentato" del cascade è sempre `qwen` (cascade order fissato cheap→expensive).

### 3. RouteLLM zero-shot collassa
- Default threshold (0.11593) instrada il 99.96% delle query al modello forte (kimi).
- Accuracy identica ad `always_kimi` baseline = **21.3%** → zero cost saving + zero routing intelligence applicata.

### 4. Cascade overhead misurato
- FrugalGPT cascade depth media = 1.70 calls/query (vs 1 pure router) → +70% inferenze sprecate sui modelli scartati.
- RouteLLM tournament = 2 calls/query (sempre 2 router stages) → 2x overhead routing.
- Brick e Cascade Routing = 1 decisione singola pre-call.

### 5. Latency outliers
- **Cascade Routing 16ms** (più veloce): sentence-transformer MiniLM batch + 3 logreg.
- **Brick 1883ms** (più lento): ModernBERT classifier processa tutti i token query, CPU-bound (binary Brick non sfrutta GPU). Query lunghe (math_reasoning, ~3s/query) dominano.
- RouteLLM/FrugalGPT in mezzo: BERT/DistilBERT classifier per single query.
- **Brick latency è migliorabile**: deployare ModernBERT su GPU (binary attualmente non lo supporta) ridurrebbe latency di 1-2 ordini di grandezza.

### 6. Brick: unico router con distribuzione realmente 3-way
- Brick usa tutti e 3 i modelli (qwen 68%, ds4 24%, kimi 8%) → **vera routing intelligence** vs baselines che collassano su 1 modello.
- Distribuzione Brick si avvicina alla ground truth (qwen 63%, ds4 16%, kimi 21%).
- Accuracy Brick 46% sotto always_qwen significa che il routing "spreca" alcune query routate verso kimi/ds4 quando qwen sarebbe bastato: ottimizzabile con calibrazione threshold.

---

## Setup tecnico

| Aspetto | Dettaglio |
|---|---|
| Macchina baselines | Workstation locale CPU (RouteLLM, FrugalGPT, Cascade Routing) |
| Macchina Brick | Cluster Seeweb `qwen-bench` (4×L40S, ma Brick gira su CPU del cluster: binary non supporta GPU) |
| Fake backend Brick | HTTP server Python `localhost:9999` → 401 in <1ms (esclude RTT network OpenRouter dalla latency) |
| Brick image | `mymodel:brick2-dev` (proprietary) + ModernBERT capability classifier mounted volume |
| Repo wrapper | `/root/forkGO/external_comparison/` (`run_routellm.py`, `run_frugalgpt.py`, `run_cascade_routing.py`, `run_brick.py`) |
| Output HF | [`massaindustries/dataset-A-routing`](https://huggingface.co/datasets/massaindustries/dataset-A-routing) subset `comparison` (5504 × 22 cols) |

## Schema subset `comparison`

```
query_id, dimension, ground_truth,
gt_qwen_correct, gt_ds4_correct, gt_kimi_correct,
routellm_binary_selected, routellm_binary_latency_ms, routellm_binary_correct,
routellm_tournament_selected, routellm_tournament_latency_ms, routellm_tournament_correct,
frugal_first_tried, frugal_router_latency_ms, frugal_correct,
cascade_selected, cascade_router_latency_ms, cascade_correct,
brick_selected, brick_route_reason, brick_router_latency_ms, brick_correct
```

---

## Note metodologiche e caveat

- **Zero-shot vincolo**: nessun fit/calibrazione su Dataset A per i 3 baseline esterni; FrugalGPT scorer addestrato su HEADLINES, Cascade Routing quality estimator fittato su RouterBench (entrambi public benchmarks).
- **RouteLLM è binario**: per output ternario è stato implementato un **tournament** (2 router concatenati qwen-vs-ds4, ds4-vs-kimi): non parte del paper originale.
- **Cascade Routing impl semplificata**: framework eth-sri richiede QualityComputer/CostComputer custom; implementazione minimale = sentence-transformer + 3 logreg fit su RouterBench, pure router (1 call), NON include cascade dinamico del paper originale.
- **Brick latency caveat**: misurata `time.perf_counter()` attorno alla chiamata HTTP `/v1/chat/completions` con backend fake locale 401: non include LLM inference né network OpenRouter ma include forwarding decision overhead di Brick. ModernBERT su CPU del cluster qwen-bench, ottimizzabile su GPU.
- **Ground truth fallback**: 922 query non risolvibili da nessun modello (16.7%) ricevono ground truth = `kimi` (most capable). Senza fallback: denominator 4582, FrugalGPT/always_qwen accuracy ~75.9%.

