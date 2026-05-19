# `packages/datasets/`: HuggingFace download recipes

The Brick datasets and trained models live on the HuggingFace Hub. This package contains only **download scripts**: no data is checked in.

## What you can download

| Artifact | HF Repo | Script | Default `--out` |
|---|---|---|---|
| Dataset A (eval, 5,504 queries) | [`regolo/brick-dataset-A-routing-eval`](https://huggingface.co/datasets/regolo/brick-dataset-A-routing-eval) | [`scripts/download_dataset_a.py`](scripts/download_dataset_a.py) | `./data/dataset_a` |
| Dataset B (training, ~50k queries) | [`massaindustries/dataset-B-modernbert-train`](https://huggingface.co/datasets/massaindustries/dataset-B-modernbert-train) | [`scripts/download_dataset_b.py`](scripts/download_dataset_b.py) | `./data/dataset_b` |
| Capability classifier | [`regolo/brick-modernbert-capability-classifier`](https://huggingface.co/regolo/brick-modernbert-capability-classifier) | [`scripts/download_models.py`](scripts/download_models.py) | `./models/capability-classifier` |
| Complexity classifier | [`regolo/brick-complexity-2-eco`](https://huggingface.co/regolo/brick-complexity-2-eco) | (same script) | `./models/complexity-classifier` |

## Quick usage

```bash
# Dataset A (eval)
python packages/datasets/scripts/download_dataset_a.py --out ./data/dataset_a

# Dataset B (training)
python packages/datasets/scripts/download_dataset_b.py --out ./data/dataset_b

# Both trained classifiers (capability + complexity)
python packages/datasets/scripts/download_models.py --out ./models
```

All scripts accept:
- `--out <path>`: destination folder.
- `--revision <ref>`: HF Hub branch/tag/commit (default `main`).
- `--token <hf_token>`: auth token; defaults to `$HF_TOKEN`. Most datasets are public.

## Dependency

Only `huggingface_hub` is required:

```bash
uv pip install huggingface_hub
# or
pip install huggingface_hub
```

The workspace `pyproject.toml` pins this in `brick-datasets` dependencies.

## Dataset A structure (after download)

```
data/dataset_a/
├── dataset_card.md                     # provenance + license
├── evaluation_parameters_full.jsonl    # per-query inference params (max_tokens, temperature, etc.)
├── evaluation_parameters_masked.jsonl  # eval-only view (no model outputs)
├── dataset_A_routing/
│   ├── results/                        # per-query, per-model graded verdicts
│   │   └── train.jsonl.gz
│   └── verbose/                        # raw responses, thinking chains, judge votes
│       └── train.jsonl.gz
```

The `results/` view is what the aggregation script (`packages/evals/scripts/130_aggregate_results.py`) reads to produce the paper's final accuracy + cost table.

## Dataset B structure (after download)

```
data/dataset_b/
├── dataset_card.md
├── train.jsonl.gz                     # ~50k labeled queries (multi-label, 6 dims)
└── human_eval.jsonl                   # human-validated stratified subset (~2,000 queries)
```

Used to train [`regolo/brick-modernbert-capability-classifier`](https://huggingface.co/regolo/brick-modernbert-capability-classifier): see [`packages/training/README.md`](../training/README.md).

## License and citation

Both datasets are released under CC BY 4.0 alongside the paper. Cite as:

```bibtex
@misc{massa2026brick,
  title  = {Brick and the Mixture-of-Models ({MoM}) Paradigm:
            Bridging Open- and Closed-Weight {LLM} Pools},
  author = {Massa, Francesco and Cristofanilli, Marco},
  year   = {2026},
  url    = {https://github.com/regolo-ai/brick-SR1}
}
```
