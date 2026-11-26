#!/bin/bash
# Adaptive re-attack (Stage 2, 18 Nov 2026): STRATA LEARNS against each target (--phase train, --hot_lifelong), on the
# 41 held-out goals that were never attacked, x 8 surfaces, same schedule every run.
#   base     the base Q4_K_M, starting from the frozen held-out library (reduced_seedoff, 199 strategies)
#   final    the earlier Stage 1 adapter (joint_all_v3_mtopd_s0 ck50), same start (cancelled 19 Nov, never run)
#   fb_warm  the paper's final model (F-b seed 0, official Q4_K_M + runtime LoRA, as deployed), starting WARM from the
#            library, attack log and summarizer log left by the finished base run: the attacker begins with everything
#            that broke the base model on these goals, then keeps learning against the final model (24 Nov 2026)
#   fb_cold  the same final model, starting from the same frozen library as base (reduced_seedoff, 199 strategies):
#            the base run's procedure with only the target changed, same 41 goals x 8 rounds budget (25 Nov 2026)
# Usage (from MTMM/): ./run_reattack.sh base | final | fb_warm | fb_cold
#   SMOKE=1  -> run id <which>_smoke, 1 round of 2 goals (checks the start, then delete logs/*_<which>_smoke*)
#   RESUME=1 -> continue a crashed run from logs/lifelong_checkpoint_reattack_<which>.pkl
set -e
cd "$(dirname "$0")"
WHICH=${1:?base, final, fb_warm or fb_cold}
BASE=~/.lmstudio/models/CohereLabs/tiny-aya-global-GGUF/tiny-aya-global-q4_k_m.gguf
EXTRA=()
SRC=reduced_seedoff
case "$WHICH" in
  base) ;;
  final) EXTRA=(--target_lora ../Multilingual_JB/checkpoints/hf/joint_all_v3_mtopd_s0/checkpoint-50/adapter.gguf
                --target_lora_scale 1.0) ;;
  fb_warm) EXTRA=(--target_lora ../Multilingual_JB/data/release_stage2/_gguf/f_grpo_comet_s0/adapter.gguf
                  --target_lora_scale 1.0); SRC=reattack_base ;;
  fb_cold) EXTRA=(--target_lora ../Multilingual_JB/data/release_stage2/_gguf/f_grpo_comet_s0/adapter.gguf
                  --target_lora_scale 1.0) ;;
  *) echo "unknown target $WHICH"; exit 1 ;;
esac
RUN=reattack_$WHICH
ITERS=(--lifelong_iterations 8)
if [ "${SMOKE:-0}" = 1 ]; then RUN=${RUN}_smoke; ITERS=(--lifelong_iterations 1 --max_lifelong 2); fi
[ "${RESUME:-0}" = 1 ] && EXTRA+=(--resume)
for f in "${EXTRA[@]}"; do [[ "$f" == *.gguf && ! -f "$f" ]] && { echo "missing $f"; exit 1; }; done
# --hot_lifelong loads library, attack log and summarizer log by run suffix; the copy is the starting state (never the
# source run's checkpoint, so the run starts at round 0). The start's attack-log length is saved so analysis can
# take only this run's entries.
for f in lifelong_strategy_library.pkl lifelong_strategy_library.json lifelong_attack_log.json lifelong_summarizer_log.json; do
  src=logs/${f%.*}_$SRC.${f##*.}; dst=logs/${f%.*}_$RUN.${f##*.}
  [ -f "$dst" ] || cp "$src" "$dst"
done
[ -f logs/$RUN.start.json ] || ../Multilingual_JB/.venv/bin/python -c "
import json, pickle
print(json.dumps({'source': '$SRC', 'attack_log_entries': len(json.load(open('logs/lifelong_attack_log_$SRC.json'))),
                  'strategies': len(pickle.load(open('logs/lifelong_strategy_library_$SRC.pkl', 'rb')))}))" > logs/$RUN.start.json
git diff > logs/$RUN.code.diff 2>/dev/null || true; git rev-parse HEAD > logs/$RUN.code.head 2>/dev/null || true
~/.lmstudio/bin/lms unload --all >/dev/null 2>&1 || true
~/.lmstudio/bin/lms load qwen/qwen3.6-35b-a3b --context-length 32768 -y >/dev/null 2>&1
WANDB_MODE=disabled caffeinate -i ../Multilingual_JB/.venv/bin/python main.py --modality text \
  --api_base http://localhost:1234/v1 --api_model qwen/qwen3.6-35b-a3b \
  --target_gguf "$BASE" "${EXTRA[@]}" \
  --target_profile tiny-aya-global-q4 --data data/reattack_goals.json --hot_lifelong --phase train \
  "${ITERS[@]}" --epochs 2 --max_turns 3 --retrieval_scope strict --script_gate_retries 4 \
  --run_id $RUN >> logs/$RUN.stdout.log 2>&1
