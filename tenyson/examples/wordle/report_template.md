# Wordle Research Experiment Report

## Stage 1: SFT + Baseline Eval
- SFT status: `{sft_status}`
- SFT WandB: `{sft_wandb_link}`
- Baseline mixed eval status: `{baseline_eval_status}`
- Baseline constraint_accuracy: `{baseline_eval_constraint_accuracy}`
- Baseline dict_accuracy: `{baseline_eval_dict_accuracy}`
- Baseline format_accuracy: `{baseline_eval_format_accuracy}`

## Branch A: Mixed RL (turns 1..5)
- Mixed RL status: `{mixed_rl_status}`
- Mixed RL WandB: `{mixed_rl_wandb_link}`
- Mixed final mixed-eval status: `{mixed_final_eval_status}`
- Mixed final constraint_accuracy: `{mixed_final_constraint_accuracy}`
- Mixed final dict_accuracy: `{mixed_final_dict_accuracy}`
- Mixed final format_accuracy: `{mixed_final_format_accuracy}`

## Branch B: Curriculum RL (2 -> 3 -> 4 -> 5)
- RL turn-2 status: `{curr_rl_t2_status}` | WandB: `{curr_rl_t2_wandb_link}`
- RL turn-3 status: `{curr_rl_t3_status}` | WandB: `{curr_rl_t3_wandb_link}`
- RL turn-4 status: `{curr_rl_t4_status}` | WandB: `{curr_rl_t4_wandb_link}`
- RL turn-5 status: `{curr_rl_t5_status}` | WandB: `{curr_rl_t5_wandb_link}`

### Curriculum Stage Evals
- After RL2, eval turn-2:
  - status: `{curr_eval_after_t2_turn2_status}`
  - constraint/dict/format: `{curr_eval_after_t2_turn2_constraint_accuracy}` / `{curr_eval_after_t2_turn2_dict_accuracy}` / `{curr_eval_after_t2_turn2_format_accuracy}`
- After RL3, eval turn-2:
  - status: `{curr_eval_after_t3_turn2_status}`
  - constraint/dict/format: `{curr_eval_after_t3_turn2_constraint_accuracy}` / `{curr_eval_after_t3_turn2_dict_accuracy}` / `{curr_eval_after_t3_turn2_format_accuracy}`
- After RL3, eval turn-3:
  - status: `{curr_eval_after_t3_turn3_status}`
  - constraint/dict/format: `{curr_eval_after_t3_turn3_constraint_accuracy}` / `{curr_eval_after_t3_turn3_dict_accuracy}` / `{curr_eval_after_t3_turn3_format_accuracy}`
- After RL4, eval turn-3:
  - status: `{curr_eval_after_t4_turn3_status}`
  - constraint/dict/format: `{curr_eval_after_t4_turn3_constraint_accuracy}` / `{curr_eval_after_t4_turn3_dict_accuracy}` / `{curr_eval_after_t4_turn3_format_accuracy}`
- After RL4, eval turn-4:
  - status: `{curr_eval_after_t4_turn4_status}`
  - constraint/dict/format: `{curr_eval_after_t4_turn4_constraint_accuracy}` / `{curr_eval_after_t4_turn4_dict_accuracy}` / `{curr_eval_after_t4_turn4_format_accuracy}`
- After RL5, eval turn-4:
  - status: `{curr_eval_after_t5_turn4_status}`
  - constraint/dict/format: `{curr_eval_after_t5_turn4_constraint_accuracy}` / `{curr_eval_after_t5_turn4_dict_accuracy}` / `{curr_eval_after_t5_turn4_format_accuracy}`
- After RL5, eval turn-5:
  - status: `{curr_eval_after_t5_turn5_status}`
  - constraint/dict/format: `{curr_eval_after_t5_turn5_constraint_accuracy}` / `{curr_eval_after_t5_turn5_dict_accuracy}` / `{curr_eval_after_t5_turn5_format_accuracy}`

## Final Mixed Eval Comparison (turns 1..5)
- Curriculum final mixed-eval status: `{curr_final_eval_status}`
- Curriculum final constraint_accuracy: `{curr_final_constraint_accuracy}`
- Curriculum final dict_accuracy: `{curr_final_dict_accuracy}`
- Curriculum final format_accuracy: `{curr_final_format_accuracy}`

### Mixed - Curriculum Delta
- constraint_accuracy delta: `{delta_final_constraint_accuracy}`
- dict_accuracy delta: `{delta_final_dict_accuracy}`
- format_accuracy delta: `{delta_final_format_accuracy}`
