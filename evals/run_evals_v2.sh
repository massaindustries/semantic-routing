#!/usr/bin/env bash
#
# run_evals_v2.sh — Piano Evals v2: Brick-only benchmarks with system prompts
#
# Launches a tmux session "evals-v2" that runs all benchmarks for Brick only.
# Output goes to stage4/ (preserves stage2/ originals for comparison).
#
# Usage:
#   ./run_evals_v2.sh                  # Launch tmux session with all phases
#   ./run_evals_v2.sh --phase 1        # Run only phase 1
#   ./run_evals_v2.sh --dry-run        # Dry run with --limit 5
#   ./run_evals_v2.sh --no-tmux        # Run in foreground (no tmux)
#
# Monitor:
#   tmux attach -t evals-v2
#
# Prerequisites:
#   - lm-eval installed at .venv path below
#   - Brick Docker container running on the eval server
#   - REGOLO_API_KEY env var set
#
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"

##############################################################################
# tmux launcher — re-exec inside a tmux session unless --no-tmux or already in
##############################################################################

USE_TMUX=true
for arg in "$@"; do
    [[ "$arg" == "--no-tmux" ]] && USE_TMUX=false
done

TMUX_SESSION="evals-v2"

if [[ "${USE_TMUX}" == "true" && -z "${TMUX:-}" && -z "${INSIDE_EVALS_TMUX:-}" ]]; then
    # Pre-flight check before launching tmux
    if [[ -z "${REGOLO_API_KEY:-}" ]]; then
        echo "ERROR: REGOLO_API_KEY is not set."
        echo "Export it before running: export REGOLO_API_KEY=sk-..."
        exit 1
    fi

    # Kill existing session if present
    tmux kill-session -t "${TMUX_SESSION}" 2>/dev/null || true

    # Build args without --no-tmux
    ARGS=()
    for arg in "$@"; do
        [[ "$arg" != "--no-tmux" ]] && ARGS+=("$arg")
    done

    echo "Launching tmux session '${TMUX_SESSION}'..."
    echo "  Attach with:  tmux attach -t ${TMUX_SESSION}"
    echo "  Kill with:    tmux kill-session -t ${TMUX_SESSION}"

    tmux new-session -d -s "${TMUX_SESSION}" \
        "INSIDE_EVALS_TMUX=1 REGOLO_API_KEY='${REGOLO_API_KEY}' bash '${SCRIPT_PATH}' --no-tmux ${ARGS[*]+"${ARGS[*]}"}"
    exit 0
fi

##############################################################################
# Configuration
##############################################################################

LM_EVAL="/home/rdseeweb/regolo-semantic-routing/.venv/bin/lm_eval"
EVALS_DIR="${SCRIPT_DIR}"
STAGE_DIR="${EVALS_DIR}/stage4"
LOGS_DIR="${STAGE_DIR}/logs"
DATE=$(date +%Y-%m-%d)

# Brick gateway (running on dedicated eval server)
BRICK_URL="http://213.171.186.210:8000/v1/chat/completions"

# Brick only — single model
MODEL="brick"
TOKENIZER="meta-llama/Llama-3.3-70B-Instruct"

##############################################################################
# CLI arguments
##############################################################################

PHASE="all"
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --phase)    PHASE="$2"; shift 2 ;;
        --dry-run)  DRY_RUN=true; shift ;;
        --no-tmux)  shift ;;  # already handled above
        -h|--help)
            head -22 "${BASH_SOURCE[0]}" | tail -20
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# REGOLO_API_KEY is always required (brick gateway forwards it to Regolo API)
if [[ -z "${REGOLO_API_KEY:-}" ]]; then
    echo "ERROR: REGOLO_API_KEY is not set."
    echo "Export it before running: export REGOLO_API_KEY=sk-..."
    exit 1
fi

# lm-eval's openai-chat-completions reads API key from OPENAI_API_KEY env var
export OPENAI_API_KEY="${REGOLO_API_KEY}"

mkdir -p "${LOGS_DIR}"

##############################################################################
# Counters
##############################################################################

TOTAL_RUN=0
TOTAL_SKIP=0
TOTAL_FAIL=0

##############################################################################
# System instruction selection
##############################################################################

get_system_instruction() {
    local task=$1
    case "${task}" in
        arc_challenge*|mmlu_pro*)
            echo "For multiple choice questions, end your response with \"the answer is (X)\" where X is the letter."
            ;;
        bbh_cot_zeroshot*)
            echo "End your response with the final answer on its own line."
            ;;
        minerva_math*)
            echo "Put your final numerical answer in \\boxed{}."
            ;;
        drop*)
            echo "Give a short, direct answer."
            ;;
        humaneval*|mbpp*)
            echo "Provide only the implementation code, no explanations or markdown."
            ;;
        ifeval*)
            echo "Follow the formatting instructions precisely."
            ;;
        truthfulqa*)
            echo "Answer concisely."
            ;;
        *)
            echo ""
            ;;
    esac
}

##############################################################################
# Core function — runs a single benchmark for Brick
##############################################################################

run_eval() {
    local task=$1
    local output_dir=$2
    local max_tokens=$3
    shift 3
    local extra_flags=("$@")

    local out_dir="${STAGE_DIR}/${output_dir}/brick"
    local log_file="${LOGS_DIR}/brick_${task//,/_}_${DATE}.log"

    # Idempotency: skip if results already exist
    if find "${out_dir}" -name "results*.json" -print -quit 2>/dev/null | grep -q .; then
        echo "[SKIP] brick / ${task} — results exist in ${out_dir}"
        (( TOTAL_SKIP++ )) || true
        return 0
    fi

    mkdir -p "${out_dir}"

    local model_type="openai-chat-completions"
    local model_args="model=${MODEL},base_url=${BRICK_URL},tokenizer_backend=huggingface,tokenizer=${TOKENIZER},stream=false,max_tokens=${max_tokens},temperature=0,top_p=1"

    # Dry run: append --limit 5
    if [[ "${DRY_RUN}" == "true" ]]; then
        extra_flags+=(--limit 5)
    fi

    # Select system instruction based on benchmark
    local system_instruction
    system_instruction="$(get_system_instruction "${task}")"

    echo "============================================================"
    echo "[RUN] brick / ${task}"
    echo "  Base URL: ${BRICK_URL}"
    echo "  Max tokens: ${max_tokens}"
    echo "  System instruction: ${system_instruction:0:80}..."
    echo "  Output: ${out_dir}"
    echo "  Log: ${log_file}"
    echo "  Started: $(date)"
    echo "============================================================"

    # IMPORTANT: Run lm-eval from /tmp to avoid CWD directory names
    # colliding with task names (lm-eval checks Path(task).is_dir()).
    (cd /tmp && "${LM_EVAL}" run \
        --model "${model_type}" \
        --model_args "${model_args}" \
        --tasks "${task}" \
        --output_path "${out_dir}" \
        --log_samples \
        --batch_size 1 \
        --apply_chat_template \
        --trust_remote_code \
        --system_instruction "${system_instruction}" \
        ${extra_flags[@]+"${extra_flags[@]}"} \
    ) 2>&1 | tee "${log_file}" || true

    local status=${PIPESTATUS[0]}

    if [[ ${status} -eq 0 ]]; then
        echo "[DONE] brick / ${task} — SUCCESS ($(date))"
        (( TOTAL_RUN++ )) || true
    else
        echo "[FAIL] brick / ${task} — exit code ${status} ($(date))"
        (( TOTAL_FAIL++ )) || true
    fi

    return ${status}
}

##############################################################################
# Phase 1 — Core benchmarks
##############################################################################

phase1() {
    echo ""
    echo "################################################################"
    echo "# PHASE 1 — Core benchmarks (Brick only)"
    echo "################################################################"

    echo ""
    echo "=== MMLU-Pro (5-shot, limit=500) ==="
    run_eval "mmlu_pro" "mmlu_pro" 2048 \
        --num_fewshot 5 \
        --limit 500 \
        --fewshot_as_multiturn True

    echo ""
    echo "=== ARC-Challenge Chat (0-shot, full) ==="
    run_eval "arc_challenge_chat" "arc_challenge" 100

    echo ""
    echo "=== TruthfulQA Gen (6-shot built-in, full) ==="
    run_eval "truthfulqa_gen" "truthfulqa" 256
}

##############################################################################
# Phase 2 — Extended benchmarks
##############################################################################

phase2() {
    echo ""
    echo "################################################################"
    echo "# PHASE 2 — Extended benchmarks (Brick only)"
    echo "################################################################"

    echo ""
    echo "=== IFEval (0-shot, full) ==="
    run_eval "ifeval" "ifeval" 1280

    echo ""
    echo "=== BBH CoT Zeroshot (0-shot, limit=50/subtask) ==="
    run_eval "bbh_cot_zeroshot" "bbh" 2048 \
        --limit 50

    echo ""
    echo "=== DROP (3-shot, limit=200) ==="
    run_eval "drop" "drop" 2048 \
        --num_fewshot 3 \
        --limit 200 \
        --fewshot_as_multiturn True

    echo ""
    echo "=== Minerva Math (4-shot, limit=100/subtask) ==="
    run_eval "minerva_math" "minerva_math" 2048 \
        --num_fewshot 4 \
        --limit 100 \
        --fewshot_as_multiturn True
}

##############################################################################
# Phase 3 — Code eval
##############################################################################

phase3() {
    echo ""
    echo "################################################################"
    echo "# PHASE 3 — Code eval (Brick only)"
    echo "################################################################"

    export HF_ALLOW_CODE_EVAL=1

    echo ""
    echo "=== HumanEval (0-shot, full) ==="
    run_eval "humaneval" "humaneval" 1024 \
        --confirm_run_unsafe_code

    echo ""
    echo "=== MBPP (3-shot, full) ==="
    run_eval "mbpp" "mbpp" 512 \
        --num_fewshot 3 \
        --fewshot_as_multiturn True \
        --confirm_run_unsafe_code
}

##############################################################################
# Main
##############################################################################

echo "========================================"
echo " Piano Evals v2 — Brick only"
echo " Date:       ${DATE}"
echo " Phase:      ${PHASE}"
echo " Dry run:    ${DRY_RUN}"
echo " lm_eval:    ${LM_EVAL}"
echo " Brick URL:  ${BRICK_URL}"
echo " Stage dir:  ${STAGE_DIR}"
echo "========================================"

case "${PHASE}" in
    1)   phase1 ;;
    2)   phase2 ;;
    3)   phase3 ;;
    all)
        phase1
        phase2
        phase3
        ;;
    *)
        echo "ERROR: Invalid phase '${PHASE}'. Use 1, 2, 3, or all."
        exit 1
        ;;
esac

echo ""
echo "========================================"
echo " Evaluation campaign complete!"
echo " Run:    ${TOTAL_RUN}"
echo " Skip:   ${TOTAL_SKIP}"
echo " Fail:   ${TOTAL_FAIL}"
echo " Results: ${STAGE_DIR}/"
echo " Logs:    ${LOGS_DIR}/"
echo "========================================"
