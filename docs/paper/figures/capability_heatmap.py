"""Heatmap 4x6 confronto query p(x) vs skill matrix dei 3 modelli.

Source p(x): q_03886 (planning_agentic, brick_selected=ds4) da
  external_comparison/predictions/brick_debug_gpu.fullseq_partial.20260517T084101Z.jsonl
Source S:    external_comparison/predictions/brick_v2_skills_calibrated.json (calibrated v2).

Output: docs/figures/capability_heatmap.png + .pdf
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

ROOT = Path("/root/forkGO")
SKILL_JSON = ROOT / "external_comparison/predictions/brick_v2_skills_calibrated.json"
DEBUG_JSONL = ROOT / "external_comparison/predictions/brick_debug_gpu.fullseq_partial.20260517T084101Z.jsonl"
OUT_DIR = ROOT / "scientificv1/docs/figures"
QUERY_ID = "q_03886"

CAP_LABELS = ["coding", "creative", "IF", "math", "planning", "world"]
ROW_LABELS = ["query $p(x)$", r"\texttt{qwen3.5-9b}", r"\texttt{deepseek-v4-flash}", r"\texttt{kimi2.6}"]


def load_skill_matrix() -> tuple[np.ndarray, list[str]]:
    data = json.loads(SKILL_JSON.read_text())
    caps = data["capabilities"]
    rows = np.array([
        data["skill_vectors"]["qwen"],
        data["skill_vectors"]["ds4"],
        data["skill_vectors"]["kimi"],
    ])
    return rows, caps


def load_query_p(qid: str, caps: list[str]) -> np.ndarray:
    with DEBUG_JSONL.open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("query_id") == qid:
                cap = r["brick_debug"]["capability"]
                return np.array([cap[c] for c in caps])
    raise SystemExit(f"query {qid} not found in {DEBUG_JSONL}")


def main() -> None:
    skill, caps = load_skill_matrix()
    p = load_query_p(QUERY_ID, caps)
    matrix = np.vstack([p, skill])  # 4 x 6

    fig, ax = plt.subplots(figsize=(6.4, 2.7))
    im = ax.imshow(matrix, cmap="YlGnBu", vmin=0.0, vmax=1.0, aspect="auto")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            txt_color = "white" if val > 0.55 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color=txt_color, fontsize=8)

    # Red boxes on per-column max over MODEL ROWS only (rows 1..3, not the query row 0).
    model_block = matrix[1:, :]
    max_idx = np.argmax(model_block, axis=0) + 1  # offset by 1 for query row
    for j, i in enumerate(max_idx):
        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1.0, 1.0,
                               fill=False, edgecolor="red", linewidth=1.8))

    # Visual separator between query row and model rows.
    ax.axhline(y=0.5, color="black", linewidth=0.6)

    ax.set_xticks(range(len(CAP_LABELS)))
    ax.set_xticklabels(CAP_LABELS, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(ROW_LABELS)))
    ax.set_yticklabels(ROW_LABELS, fontsize=8)
    ax.tick_params(axis="both", which="both", length=0)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label("probability / skill", fontsize=8)

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out = OUT_DIR / f"capability_heatmap.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"[ok] {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
