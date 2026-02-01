# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
# Copyright 2025 Horizon RL Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO-like algorithms.
"""
from collections import defaultdict
from typing import Any, Callable, Optional

import numpy as np
import torch
from omegaconf import DictConfig

from verl.trainer.ppo.core_algos import (
    agg_loss, 
    register_adv_est,
    register_policy_loss,
)
from verl.trainer.config import AlgoConfig
from verl.workers.config import ActorConfig


@register_adv_est("ploo")
def compute_partition_loo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    tau: float = 0.01,
    config: Optional[AlgoConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage with partition function (leave-one-out) as baseline.

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length)
        index: `(np.ndarray)`
            index array for grouping
        tau: `(float)`
            temperature in partition function (default: 0.01)
        config: `(Optional[AlgoConfig])`
            algorithm configuration object. If provided, reads tau from config.partition_tau

    Returns:
        advantages: `(torch.Tensor)`
            shape is (bs, response_length)
        Returns: `(torch.Tensor)`
            shape is (bs, response_length)
    """
    # Get tau from config if available
    if config is not None:
        tau = config.get("partition_tau", tau)
    
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2partition = {}
    id2max = {}
    baseline = torch.zeros_like(scores)

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2partition[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2max[idx] = torch.max(scores_tensor)
                id2partition[idx] = torch.sum(
                    torch.exp((scores_tensor - id2max[idx]) / tau)
                )
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            response_num = len(id2score[index[i]])
            if response_num > 1:
                partition = id2partition[index[i]] - torch.exp((scores[i] - id2max[index[i]]) / tau)
                # Baseline is the leave-one-out log-mean-exp of other responses in the group.
                denom = torch.clamp(partition / (response_num - 1), min=1e-8)
                baseline[i] = id2max[index[i]] + tau * torch.log(denom)
        adv = (scores - baseline).unsqueeze(-1) * response_mask

    return adv, adv


@register_policy_loss("opmd")
def compute_policy_loss_opmd(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "seq-mean-token-mean",
    config: Optional[DictConfig | AlgoConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
    extra_loss_kwargs: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Online Policy Mirror Descent.
    
    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "seq-mean-token-mean".
        config: Actor configuration containing pmd_tau parameter
            
    Returns:
        tuple: (pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower)
            pg_clipfrac and pg_clipfrac_lower are set to 0.0 as PMD doesn't use clipping
    """
    assert config is not None
    assert not isinstance(config, AlgoConfig), "passing AlgoConfig not supported yet"
    
    # Get PMD-specific hyperparameters
    pmd_tau = config.policy_loss.get("pmd_tau", 0.01) if hasattr(config, "policy_loss") and config.policy_loss is not None else 0.01
    # Compute sequence-level quantities
    # For PMD, we work at sequence level: sum over tokens, then mean over batch
    response_lengths = torch.sum(response_mask, dim=-1).clamp(min=1)  # (batch_size,)
    
    # Sequence-level log probabilities: sum over tokens
    seq_log_prob = torch.sum(log_prob * response_mask, dim=-1)  # (batch_size,)
    seq_old_log_prob = torch.sum(old_log_prob * response_mask, dim=-1)  # (batch_size,)
    seq_log_prob_ratio = seq_log_prob - seq_old_log_prob
    
    # Sequence-level advantages and weights
    seq_advantages = torch.sum(advantages * response_mask, dim=-1) / response_lengths  # (batch_size,)
    if extra_loss_kwargs is not None and "partition_weights" in extra_loss_kwargs:
        partition_weights = extra_loss_kwargs["partition_weights"]
        seq_partition_weights = torch.sum(partition_weights * response_mask, dim=-1) / response_lengths
    else:
        seq_partition_weights = torch.ones_like(seq_advantages)
    
    # MSE term in Eq. (3), https://arxiv.org/pdf/2501.12599
    # Scale by length to normalize loss magnitude
    max_response_length = response_mask.shape[1]
    if loss_agg_mode == "seq-mean-token-mean":  # default
        weighted_loss = torch.mean(
            0.5 * pmd_tau * seq_partition_weights * ((seq_advantages / pmd_tau - seq_log_prob_ratio)**2) / response_lengths
        )
        pg_loss = weighted_loss
    elif loss_agg_mode == "seq-mean-token-sum-norm": 
        weighted_loss = torch.mean(
            0.5 * pmd_tau * seq_partition_weights * ((seq_advantages / pmd_tau - seq_log_prob_ratio)**2)
        )
        pg_loss = weighted_loss / max_response_length
    else: 
        raise NotImplementedError
    
    if rollout_is_weights is not None:
        pg_loss = pg_loss * rollout_is_weights
    
    # Compute KL(pi_old || pi_theta) for monitoring (sequence-level to match loss)
    seq_kl = -seq_log_prob_ratio / response_lengths  # Normalize by length
    ppo_kl = torch.mean(seq_kl)
    
    # PMD doesn't use clipping, so clipfracs are zero
    pg_clipfrac = torch.tensor(0.0, device=pg_loss.device)
    pg_clipfrac_lower = torch.tensor(0.0, device=pg_loss.device)

    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }
    
    return pg_loss, pg_metrics
