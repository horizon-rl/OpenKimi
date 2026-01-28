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


@register_adv_est("partition")
def compute_partition_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor, 
    index: np.ndarray,
    tau: float = 0.01,
    config: Optional[AlgoConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage with partition function as baseline.
    original implementation: 
    https://github.com/zhenghaoxu-gatech/verl/blob/c8d65a6c9678ca58ae899b24ded1d1cb80f073f2/verl/trainer/ppo/core_algos.py#L729

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

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2partition[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                max_score = torch.max(scores_tensor)
                shifted = scores_tensor - max_score
                id2partition[idx] = tau * (torch.log(torch.mean(torch.exp(shifted / tau))) + max_score / tau)
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2partition[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores

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

@register_policy_loss("pmd")
def compute_policy_loss_pmd(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "seq-mean-token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute Policy Mirror Descent (PMD) loss.
    
    PMD uses KL divergence from the sampling policy as a regularizer instead of clipping.
    Loss = -E[A(s,a) * log π(a|s)] + τ * KL(π || π_old)
    
    Based on the mirror descent framework for policy optimization.
    References:
    - "Trust Region Policy Optimization" (Schulman et al., 2015)
    - "Relative Entropy Policy Search" (Peters et al., 2010)
    
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
        rollout_is_weights: Not used in PMD
            
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
    
    # Sequence-level advantages: mean over tokens (or sum, depending on your preference)
    seq_advantages = torch.sum(advantages * response_mask, dim=-1) / response_lengths  # (batch_size,)
    
    # MSE term in Eq. (3), https://arxiv.org/pdf/2501.12599
    # Scale by max response length to normalize loss magnitude
    max_response_length = response_mask.shape[1]
    pg_loss = torch.mean((seq_advantages - pmd_tau * (seq_log_prob - seq_old_log_prob)) ** 2) / max_response_length / pmd_tau
    
    # Compute KL for monitoring (sequence-level to match loss)
    seq_kl = -(seq_log_prob - seq_old_log_prob) / response_lengths  # Normalize by length
    ppo_kl = torch.mean(seq_kl)
    
    # PMD doesn't use clipping, so clipfracs are zero
    pg_clipfrac = torch.tensor(0.0, device=pg_loss.device)
    pg_clipfrac_lower = torch.tensor(0.0, device=pg_loss.device)

    # TODO: @zhenghaoxu-gatech add rollout correction in loss  
    
    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower

@register_policy_loss("pmd_token")
def compute_policy_loss_pmd_token(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "seq-mean-token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute Policy Mirror Descent (PMD) token-level loss.
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
    
    normalized_advs = advantages / response_lengths.unsqueeze(1)
    token_loss = ((normalized_advs - pmd_tau * (log_prob - old_log_prob))**2) / pmd_tau
    pg_loss = agg_loss(loss_mat=token_loss, loss_mask=response_mask, loss_agg_mode="seq-mean-token-mean")
    
    # Compute KL for monitoring (sequence-level to match loss)
    seq_kl = -(seq_log_prob - seq_old_log_prob) / response_lengths  # Normalize by length
    ppo_kl = torch.mean(seq_kl)
    
    # PMD doesn't use clipping, so clipfracs are zero
    pg_clipfrac = torch.tensor(0.0, device=pg_loss.device)
    pg_clipfrac_lower = torch.tensor(0.0, device=pg_loss.device)
    
    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower

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
    No weight PMD.
    
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
        rollout_is_weights: Not used in PMD
            
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
    
    # Sequence-level advantages and weights
    seq_advantages = torch.sum(advantages * response_mask, dim=-1) / response_lengths  # (batch_size,)
    # Try to get partition_weights from extra_loss_kwargs first, then from config
    if extra_loss_kwargs is not None and "partition_weights" in extra_loss_kwargs:
        partition_weights = extra_loss_kwargs["partition_weights"]
    elif hasattr(config, 'extra_loss_data') and "partition_weights" in config.extra_loss_data:
        partition_weights = config.extra_loss_data["partition_weights"]
    else:
        partition_weights = None

    if partition_weights is not None:
        seq_partition_weights = torch.sum(partition_weights * response_mask, dim=-1) / response_lengths
    else:
        seq_partition_weights = torch.ones_like(seq_advantages)
    
    # MSE term in Eq. (3), https://arxiv.org/pdf/2501.12599
    # Scale by max response length to normalize loss magnitude
    max_response_length = response_mask.shape[1]
    if loss_agg_mode == "seq-mean-token-mean":
        weighted_loss = torch.mean(
            pmd_tau * seq_partition_weights * ((seq_advantages / pmd_tau - (seq_log_prob - seq_old_log_prob))**2) / response_lengths
        )
        pg_loss = weighted_loss
    else: # loss_agg_mode == "seq-mean-token-sum-norm"
        weighted_loss = torch.mean(
            pmd_tau * seq_partition_weights * (seq_advantages / pmd_tau - (seq_log_prob - seq_old_log_prob))**2
        )
        pg_loss = weighted_loss / max_response_length
    
    # Compute KL for monitoring (sequence-level to match loss)
    seq_kl = -(seq_log_prob - seq_old_log_prob) / response_lengths  # Normalize by length
    ppo_kl = torch.mean(seq_kl)
    
    # PMD doesn't use clipping, so clipfracs are zero
    pg_clipfrac = torch.tensor(0.0, device=pg_loss.device)
    pg_clipfrac_lower = torch.tensor(0.0, device=pg_loss.device)
    
    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower

@register_policy_loss("apmd")
def compute_policy_loss_apmd(
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
    alpha-divergence PMD. Minimize alpha-divergence between current policy and target solution.
    
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
        rollout_is_weights: Not used in PMD
            
    Returns:
        tuple: (pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower)
            pg_clipfrac and pg_clipfrac_lower are set to 0.0 as PMD doesn't use clipping
    """
    assert config is not None
    assert not isinstance(config, AlgoConfig), "passing AlgoConfig not supported yet"
    
    # Get PMD-specific hyperparameters
    pmd_tau = config.policy_loss.get("pmd_tau", 0.01) if hasattr(config, "policy_loss") and config.policy_loss is not None else 0.01
    pmd_alpha = config.policy_loss.get("pmd_alpha", 0.01) if hasattr(config, "policy_loss") and config.policy_loss is not None else 1.0
    
    # Compute sequence-level quantities
    # For PMD, we work at sequence level: sum over tokens, then mean over batch
    response_lengths = torch.sum(response_mask, dim=-1).clamp(min=1)  # (batch_size,)
    
    # Sequence-level log probabilities: sum over tokens
    seq_log_prob = torch.sum(log_prob * response_mask, dim=-1)  # (batch_size,)
    seq_old_log_prob = torch.sum(old_log_prob * response_mask, dim=-1)  # (batch_size,)
    
    # Sequence-level advantages and weights
    seq_advantages = torch.sum(advantages * response_mask, dim=-1) / response_lengths  # (batch_size,)
    # Try to get partition_weights from extra_loss_kwargs first, then from config
    if extra_loss_kwargs is not None and "partition_weights" in extra_loss_kwargs:
        partition_weights = extra_loss_kwargs["partition_weights"]
    elif hasattr(config, 'extra_loss_data') and "partition_weights" in config.extra_loss_data:
        partition_weights = config.extra_loss_data["partition_weights"]
    else:
        raise ValueError("Weighted PMD loss requires 'partition_weights' provided via extra_loss_kwargs or config.extra_loss_data.")
    seq_partition_weights = torch.sum(partition_weights * response_mask, dim=-1) / response_lengths  # (batch_size,)
    
    # MSE term in Eq. (3), https://arxiv.org/pdf/2501.12599
    # Scale by max response length to normalize loss magnitude
    max_response_length = response_mask.shape[1]
    if np.isclose(pmd_alpha, 1.0):
        log_ratio = torch.clamp(seq_log_prob - seq_old_log_prob, min=-20.0, max=20.0)
        ratio = torch.exp(log_ratio)
        
        gradient_scaler = torch.clamp(ratio, max=10.0) / ratio
        gradient_scaler = gradient_scaler.detach() 
        
        weighted_loss = torch.mean(gradient_scaler * (ratio * (log_ratio - seq_advantages / pmd_tau)))
    elif np.isclose(pmd_alpha, 0.0):
        weighted_loss = torch.mean(-seq_partition_weights * (seq_log_prob - seq_old_log_prob - seq_advantages / pmd_tau))
    else:
        ratio = torch.exp(seq_log_prob - seq_old_log_prob - seq_advantages / pmd_tau)
        weighted_loss = torch.mean(seq_partition_weights * ((ratio**pmd_alpha) - pmd_alpha * ratio - (1-pmd_alpha)) / (pmd_alpha * (pmd_alpha-1)))
    pg_loss = weighted_loss / max_response_length
    
    # Compute KL for monitoring (sequence-level to match loss)
    seq_kl = -(seq_log_prob - seq_old_log_prob) / response_lengths  # Normalize by length
    ppo_kl = torch.mean(seq_kl)
    
    # PMD doesn't use clipping, so clipfracs are zero
    pg_clipfrac = torch.tensor(0.0, device=pg_loss.device)
    pg_clipfrac_lower = torch.tensor(0.0, device=pg_loss.device)
    
    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower

@register_policy_loss("cmdpo")
def compute_policy_loss_cmdpo(
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
    CMDPO, https://openreview.net/pdf?id=OaijL8iG5G.
    
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
        rollout_is_weights: Not used
            
    Returns:
        tuple: (pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower)
            pg_clipfrac and pg_clipfrac_lower are set to 0.0 as PMD doesn't use clipping
    """
    assert config is not None
    assert not isinstance(config, AlgoConfig), "passing AlgoConfig not supported yet"
    
    # Get PMD-specific hyperparameters
    pmd_tau = config.policy_loss.get("pmd_tau", 0.01) if hasattr(config, "policy_loss") and config.policy_loss is not None else 0.01
    # Try to get uid from extra_loss_kwargs first, then from config
    if extra_loss_kwargs is not None and "uid" in extra_loss_kwargs:
        uid = extra_loss_kwargs["uid"]
    elif hasattr(config, 'extra_loss_data') and "uid" in config.extra_loss_data:
        uid = config.extra_loss_data["uid"]
    else:
        raise ValueError("CMDPO loss requires 'uid' provided via extra_loss_kwargs or config.extra_loss_data.")
    
    # Compute sequence-level quantities
    response_lengths = torch.sum(response_mask, dim=-1).clamp(min=1)  # (batch_size,)
    seq_log_prob = torch.sum(log_prob * response_mask, dim=-1)  # (batch_size,)
    seq_old_log_prob = torch.sum(old_log_prob * response_mask, dim=-1)  # (batch_size,)
    seq_advantages = torch.sum(advantages * response_mask, dim=-1) / response_lengths  # (batch_size,)

    ratio = torch.clamp(seq_log_prob - seq_old_log_prob, min=-10.0, max=10.0)
    
    # CMDPO loss starts
    uid_list = list(uid)
    assert len(uid_list) > 0, "CMDPO expects non-empty uid list."
    if len(uid_list) == 1:
        raise ValueError("CMDPO requires at least two responses per prompt.")
    group_size = None
    first_uid = uid_list[0]
    for idx in range(1, len(uid_list)):
        if uid_list[idx] != first_uid:
            group_size = idx
            break
    group_size = group_size or len(uid_list)
    assert group_size > 1, "CMDPO requires each prompt to have at least two responses."
    assert len(uid_list) % group_size == 0, "Batch size must be divisible by CMDPO group size."

    num_groups = len(uid_list) // group_size
    ratio_group = ratio.contiguous().view(num_groups, group_size)
    sum_ratio = ratio_group.sum(dim=1, keepdim=True)
    loo_ratio = (sum_ratio - ratio_group) / (group_size - 1)
    centered_ratio = (ratio_group - loo_ratio).reshape(-1)

    max_response_length = response_mask.shape[1]
    cmdpo_residual = pmd_tau * centered_ratio - seq_advantages
    pg_loss = torch.mean(cmdpo_residual**2) / max_response_length / pmd_tau
    # CMDPO loss ends
    
    # Compute KL for monitoring (sequence-level to match loss)
    seq_kl = -(seq_log_prob - seq_old_log_prob) / response_lengths  # Normalize by length
    ppo_kl = torch.mean(seq_kl)
    
    # PMD doesn't use clipping, so clipfracs are zero
    pg_clipfrac = torch.tensor(0.0, device=pg_loss.device)
    pg_clipfrac_lower = torch.tensor(0.0, device=pg_loss.device)
    
    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower