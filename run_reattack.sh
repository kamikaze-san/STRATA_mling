#!/bin/bash
# Adaptive re-attack (Stage 2, 18 Nov 2026): STRATA LEARNS against each target (--phase train, starting from the
# frozen held-out library via --hot_lifelong), on the 41 held-out goals that were never attacked, x 8 surfaces.
# Run twice, same settings: the base Q4_K_M, and the final model (base + final GGUF LoRA, as deployed).
# Usage (from MTMM/): ./run_reattack.sh base   |   ./run_reattack.sh final
set -e
cd "$(dirname "$0")"
WHICH=${1:?base or final}
BASE=~/.lmstudio/models/CohereLabs/tiny-aya-global-GGUF/tiny-aya-global-q4_k_m.gguf
LORA=../Multilingual_JB/checkpoints/hf/joint_all_v3_mtopd_s0/checkpoint-50/adapter.gguf
EXTRA=()
[ "$WHICH" = final ] && EXTRA=(--target_lora "$LORA" --target_lora_scale 1.0)
# --hot_lifelong loads library, attack log and summarizer log by run suffix: start both runs from the same learned
# library (reduced_seedoff, 199 strategies; the held-out runs used the same library frozen)
for f in lifelong_strategy_library.pkl lifelong_strategy_library.json lifelong_attack_log.json lifelong_summarizer_log.json; do
  src=logs/${f%.*}_reduced_seedoff.${f##*.}; dst=logs/${f%.*}_reattack_$WHICH.${f##*.}
  [ -f "$dst" ] || cp "$src" "$dst"
done
git diff > logs/reattack_$WHICH.code.diff 2>/dev/null || true; git rev-parse HEAD > logs/reattack_$WHICH.code.head 2>/dev/null || true
~/.lmstudio/bin/lms unload --all >/dev/null 2>&1 || true
~/.lmstudio/bin/lms load qwen/qwen3.6-35b-a3b --context-length 32768 -y >/dev/null 2>&1
WANDB_MODE=disabled caffeinate -i ../Multilingual_JB/.venv/bin/python main.py --modality text \
  --api_base http://localhost:1234/v1 --api_model qwen/qwen3.6-35b-a3b \
  --target_gguf "$BASE" "${EXTRA[@]}" \
  --target_profile tiny-aya-global-q4 --data data/reattack_goals.json --hot_lifelong --phase train \
  --lifelong_iterations 8 --epochs 2 --max_turns 3 --retrieval_scope strict --script_gate_retries 4 \
  --run_id reattack_$WHICH > logs/reattack_$WHICH.stdout.log 2>&1
