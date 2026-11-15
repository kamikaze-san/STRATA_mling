# STRATA (multilingual text mode): lifelong multi-turn jailbreak search across languages and code-mixing

> **Content warning and dual-use notice.** This is red-teaming code: it searches for multi-turn jailbreaks
> against a target language model. It is released for safety research (finding and repairing multilingual
> safety gaps). The files that carry ready-made attack material are **withheld** from this public release
> (see `WITHHELD.md`); goal lists, attack logs and the learned strategy library are not included.

## Lineage

| Stage | When | What |
|---|---|---|
| [AutoDAN-Turbo](https://github.com/SaFoLab-WISC/AutoDAN-Turbo) | upstream (MIT, see `LICENSE.txt`) | lifelong strategy-learning jailbreak agent: attacker, scorer, summarizer, strategy library with embedding retrieval |
| STRATA, multimodal | before the hackathon (Feb–Aug 2026) | our adaptation to multi-turn, multimodal attacks (image canvas and PDF modalities, Flux / Kolors image generation) |
| STRATA, multilingual text mode | **during the Lexsi hackathon (Nov 2026)** | the text-mode, multilingual and code-mixed extension described below |

## Added during the hackathon

- **Eight linguistic surfaces** (`framework/linguistic.py`, `configs/linguistic_profiles.yaml`): `en`, `hi`,
  `hi_en_mix`, `hinglish`, `te`, `te_en_mix`, `ja`, `ja_en_mix`, with per-surface script checks
  (`--script_gate_retries`) so the attacker really writes in the requested script and mix.
- **Multi-turn text mode** (`--modality text`, `--max_turns`): attacker follow-up turns, per-turn scoring,
  tactic rotation (`configs/text_mode/method_rotation.yaml`).
- **Local models through OpenAI-compatible servers** (`llm/openai_compat_model.py`, e.g. LM Studio) for the
  attacker, scorer and summarizer, and a **llama.cpp GGUF target** with optional GGUF LoRA
  (`llm/llamacpp_model.py`, `--target_gguf`, `--target_lora`), so attacks run against the deployed Q4_K_M model.
- **Held-out goal splits** (`scripts/make_splits.py`, `--retrieval_scope strict`) so evaluation goals are never
  seen during strategy learning.
- **Run analysis without exposure** (`scripts/summarize_run_log.py`, `scripts/inspect_trajectories.py`,
  `scripts/compare_labelers.py`): counts and per-surface success rates, never attack text.
- **Checkpoint/resume** for long lifelong runs, plus tests (`tests/`).

In the private history these changes are 45 files, about +7,200 / −500 lines of code and configs since the
last pre-hackathon commit (6 Aug 2026).

## Running

Python 3.12 (`pip install -r requirements.txt`), an OpenAI-compatible server for the attacker/scorer/summarizer
model, and the target GGUF. Example (held-out test phase against tiny-aya-global Q4_K_M):

```bash
python main.py --modality text \
  --api_base http://localhost:1234/v1 --api_model <attacker/scorer model id> \
  --target_gguf <path>/tiny-aya-global-q4_k_m.gguf --target_profile tiny-aya-global-q4 \
  --data <goals.json> --hot_lifelong --phase test \
  --lifelong_iterations 8 --epochs 2 --max_turns 3 --retrieval_scope strict --run_id heldout_test
```

`python main.py --help` lists every option. Text mode needs the withheld prompt files in
`configs/text_mode/` (see `WITHHELD.md`) and a goals file you provide. `run_topup2.sh`, `run_topup3.sh` and
`run_heldout_test.sh` are the exact commands used for the hackathon runs.

## How it connects to the rest of the project

STRATA's attack logs feed the preference-pair pipeline and the held-out evaluation set in
[Multilingual_JB](https://github.com/kamikaze-san/Multilingual_JB) (labelling, CuratorKIT, DPO + LoRA on
Hugging Face Jobs, evaluation of the repaired model).

## License

MIT (`LICENSE.txt`, inherited from AutoDAN-Turbo); our changes are released under the same license.
