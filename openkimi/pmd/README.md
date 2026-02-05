# Policy Mirror Descent (PMD)

Last updated: 02/04/2026.

Implementation of online policy mirror descent (PMD) used in Kimi K1.5 [arXiv:2501.12599](https://arxiv.org/pdf/2501.12599) and K2 [arXiv:2507.20534](https://arxiv.org/pdf/2507.20534).

## Variants and key components

1. PMD-partition (`adv_estimator=ploo`)
   - Registered in `openkimi/pmd/core_algos.py` as `@register_adv_est("ploo")`.
   - Groups rollouts by prompt id, computes a leave-one-out partition baseline, and returns token-masked advantages.

2. PMD-mean (`adv_estimator=rloo`)
   - Uses VERL's registered `rloo` advantage estimator.
   - Treat this as the mean-baseline PMD variant.

3. Online PMD policy loss (`opmd`)
   - Registered in `openkimi/pmd/core_algos.py` as `@register_policy_loss("opmd")`.
   - Uses sequence-level log-prob ratios and sequence-level advantages with a PMD regularization parameter `pmd_tau`.
   - Supports optional per-sample weighting through `partition_weights`.

4. Weighted PMD helper path in the Ray trainer
   - Implemented in `openkimi/pmd/pmd_ray_trainer.py` via `compute_wpmd_weight`.
   - Produces `partition_weights` and optional prompt filtering that keeps the groups with average reward in open interval `(lb, ub)` using:
     - `algorithm.partition_tau`
     - `algorithm.partition_reward_lb`
     - `algorithm.partition_reward_ub`

5. PMD trainer entrypoint
   - `openkimi/pmd/main_pmd.py` launches `RayPMDTrainer` with Hydra config.
   - Imports `openkimi.pmd.core_algos` to ensure PMD algorithms are registered in all worker processes.

## Launch example

```bash
python3 -m openkimi.pmd.main_pmd \
  --config-path /path/to/verl/trainer/config \
  --config-name ppo_trainer \
  algorithm.adv_estimator=ploo \
  +algorithm.partition_tau=0.01 \
  actor_rollout_ref.actor.policy_loss.loss_mode=opmd \
  +actor_rollout_ref.actor.policy_loss.pmd_tau=0.01
```

Switch `algorithm.adv_estimator` to choose PMD variant:
- `ploo` -> PMD-partition
- `rloo` -> PMD-mean

For full multi-node examples, see:
- `examples/math/run_pmd_dapo17k_qwen25-7b.sh`
- `examples/math/run_pmd_dapo17k_qwen3-30b-a3b.sh`

## Results

[30B PMD results (PDF)](../../asset/30b_results.pdf)

![30B PMD results](../../asset/30b_results.png)

## Key configs

| Config key | Typical value | Effect |
|---|---:|---|
| `algorithm.adv_estimator` | `ploo` or `rloo` | Select PMD variant: `ploo` (PMD-partition, fitting ideal KL subproblem solution), `rloo` (PMD-mean, Kimi K1.5/K2 approximation) |
| `algorithm.partition_tau` | `0.01` | `tau` for partition baseline/weights |
| `algorithm.partition_reward_lb` | `null` | Optional prompt-level lower bound filter |
| `algorithm.partition_reward_ub` | `null` | Optional prompt-level upper bound filter |
| `actor_rollout_ref.actor.policy_loss.loss_mode` | `opmd` | Enable PMD policy loss |
| `actor_rollout_ref.actor.policy_loss.pmd_tau` | `0.01` | PMD regularization parameter, typically set equal to `partition_tau` |
| `actor_rollout_ref.actor.loss_agg_mode` | `seq-mean-token-mean` | Sequence loss normalization mode |

## File map

- `openkimi/pmd/core_algos.py`: PMD advantage and policy loss registrations and implementations.
- `openkimi/pmd/pmd_ray_trainer.py`: PMD trainer loop extension and weighted PMD helper.
- `openkimi/pmd/main_pmd.py`: Hydra and Ray entrypoint for PMD training.
