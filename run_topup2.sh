#!/bin/bash
# Overnight top-up (11-12 Nov): 120 fresh TRAINING goals x 5 surfaces (the ones that cap the conditions).
# Test phase (frozen library, no summarizer): the train-phase summarizer fails with HTTP 400 and is 4-5x slower.
# Starts only after the Mac sweep evaluation has finished (one heavy GPU job at a time).
set -e
cd "$(dirname "$0")"
while pgrep -f eval_sweep.sh >/dev/null || pgrep -f eval_adapter.py >/dev/null || pgrep -f score_stance_decider.py >/dev/null; do sleep 30; done
~/.lmstudio/bin/lms unload --all >/dev/null 2>&1 || true
~/.lmstudio/bin/lms load qwen/qwen3.6-35b-a3b --context-length 32768 -y >/dev/null 2>&1
git diff > logs/topup2t.code.diff; git rev-parse HEAD > logs/topup2t.code.head
echo "topup2t start $(date +%H:%M)"
WANDB_MODE=disabled caffeinate -i ../Multilingual_JB/.venv/bin/python main.py --modality text \
  --api_base http://localhost:1234/v1 --api_model qwen/qwen3.6-35b-a3b \
  --target_gguf ~/.lmstudio/models/CohereLabs/tiny-aya-global-GGUF/tiny-aya-global-q4_k_m.gguf \
  --target_profile tiny-aya-global-q4 --data data/topup2_goals.json --hot_lifelong --phase test \
  --lifelong_iterations 5 --epochs 2 --max_turns 3 --tactic_rotation \
  --linguistic_profile en,te,hi,hinglish,hi_en_mix --script_gate_retries 4 --retrieval_scope strict \
  --run_id topup2t > logs/topup2t.stdout.log 2>&1
echo "topup2t exit=$? $(date +%H:%M)"
