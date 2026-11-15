# Files withheld from the public release

These exist in the private repository and are needed for a full run, but they contain ready-made attack
material, harmful goal lists, or attack outputs, so they are not published here.

| Withheld | Why | Needed for |
|---|---|---|
| `configs/text_mode/few_shot_examples.md`, `few_shot_examples_followup.md` | worked attack examples for the attacker model | text mode |
| `configs/text_mode/capability_menu.md`, `capability_menu_compact.md`, `phase_guidance.md`, `HANDOFF_ATTACKER.md` | attacker tactic guidance | text mode |
| `configs/inspection/labeler_prompt.md` | trajectory labelling prompt | `scripts/probe_labeler.py`, `scripts/inspect_trajectories.py` |
| `data/` (goal lists, held-out and training splits, jailbreak templates, target profiles) | harmful-request goals and templates | every run |
| `logs/` (attack logs, strategy libraries) | successful attacks and learned strategies | warm-up / lifelong resume, downstream DPO pairs |
| `sft/`, `assets/`, `all_v1.jsonl` | pre-hackathon multimodal training data and attack assets | multimodal mode |

Researchers who need these for safety work can request them from the authors; access is reviewed case by case.
