# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
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
implement PPO
"""

import numpy as np
import torch
from collections import defaultdict

import verl.utils.torch_functional as verl_F


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        pass


def get_kl_controller(config):
    if config.critic.kl_ctrl.type == 'fixed':
        kl_ctrl = FixedKLController(kl_coef=config.critic.kl_ctrl.kl_coef)
    elif config.critic.kl_ctrl.type == 'adaptive':
        assert config.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
        kl_ctrl = AdaptiveKLController(init_kl_coef=config.critic.kl_ctrl.kl_coef,
                                       target_kl=config.critic.kl_ctrl.target_kl,
                                       horizon=config.critic.kl_ctrl.horizon)
    else:
        raise ValueError('Unknown kl_ctrl type')

    return kl_ctrl


def compute_gae_advantage_return(token_level_rewards: torch.Tensor, values: torch.Tensor, eos_mask: torch.Tensor,
                                 gamma: torch.Tensor, lam: torch.Tensor):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma: `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, eos_mask)
    return advantages, returns


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
def compute_grpo_outcome_advantage(token_level_rewards: torch.Tensor,
                                   eos_mask: torch.Tensor,
                                   index: torch.Tensor,
                                   epsilon: float = 1e-6):
    """
    Compute advantage for GRPO, operating only on Outcome reward 
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return scores, scores


def compute_grpo_weighted_advantage_add(token_level_rewards: torch.Tensor,
                                        eos_mask: torch.Tensor,
                                        index: torch.Tensor,
                                        group_index: torch.Tensor,
                                        weight_power: float = 0.7,
                                        epsilon: float = 1e-6):
    """
    Compute advantage for GRPO with diversity weighting based on solution grouping.
    This combines GRPO advantage with a diversity weight based on solution frequency.
    Uses ADDITION operation: combined_advantages = grpo_advantages + weights
    
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        index: `(torch.Tensor)`
            shape: (bs,) - prompt group indices for GRPO
        group_index: `(torch.Tensor)`
            shape: (bs,) - solution group indices within each prompt group
        weight_power: `(float)`
            power for frequency weighting, default 0.7
        epsilon: `(float)`
            small value to avoid division by zero
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    # First compute GRPO advantages
    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        
        # Compute GRPO advantages (same as original GRPO)
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        
        grpo_advantages = torch.zeros_like(scores)
        for i in range(bsz):
            grpo_advantages[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        
        # Compute diversity weights based on solution grouping
        # Count frequency of each solution group within each prompt group
        group_freq = defaultdict(lambda: defaultdict(int))
        for i in range(bsz):
            prompt_idx = index[i]  # index is numpy array of strings/objects
            group_idx = group_index[i].item()  # group_index is tensor, need .item()
            group_freq[prompt_idx][group_idx] += 1
        
        # Compute weights: weight[i] = 1.0 / (freq[i] ** weight_power)
        weights = torch.zeros_like(scores)
        for i in range(bsz):
            prompt_idx = index[i]  # index is numpy array of strings/objects
            group_idx = group_index[i].item()  # group_index is tensor, need .item()
            freq = group_freq[prompt_idx][group_idx]
            weights[i] = 1.0 / (freq ** weight_power)
        
        # Combine GRPO advantages with diversity weights (ADDITION)
        combined_advantages = grpo_advantages + weights
        combined_advantages = combined_advantages.unsqueeze(-1).tile([1, response_length]) * eos_mask
        
        # Also return the individual components for metrics
        grpo_advantages_expanded = grpo_advantages.unsqueeze(-1).tile([1, response_length]) * eos_mask
        weights_expanded = weights.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return combined_advantages, combined_advantages, {
        'grpo_advantages': grpo_advantages_expanded,
        'diversity_weights': weights_expanded,
        'grpo_advantages_scalar': grpo_advantages,  # scalar version for easier metrics
        'diversity_weights_scalar': weights         # scalar version for easier metrics
    }


def compute_grpo_weighted_advantage_mul(token_level_rewards: torch.Tensor,
                                        eos_mask: torch.Tensor,
                                        index: torch.Tensor,
                                        group_index: torch.Tensor,
                                        weight_power: float = 0.7,
                                        epsilon: float = 1e-6):
    """
    Compute advantage for GRPO with diversity weighting based on solution grouping.
    This combines GRPO advantage with a diversity weight based on solution frequency.
    Uses MULTIPLICATION operation: combined_advantages = grpo_advantages * weights
    
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        index: `(torch.Tensor)`
            shape: (bs,) - prompt group indices for GRPO
        group_index: `(torch.Tensor)`
            shape: (bs,) - solution group indices within each prompt group
        weight_power: `(float)`
            power for frequency weighting, default 0.7
        epsilon: `(float)`
            small value to avoid division by zero
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    # First compute GRPO advantages
    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        
        # Compute GRPO advantages (same as original GRPO)
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        
        grpo_advantages = torch.zeros_like(scores)
        for i in range(bsz):
            grpo_advantages[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        
        # Compute diversity weights based on solution grouping
        # Count frequency of each solution group within each prompt group
        group_freq = defaultdict(lambda: defaultdict(int))
        for i in range(bsz):
            prompt_idx = index[i]  # index is numpy array of strings/objects
            group_idx = group_index[i].item()  # group_index is tensor, need .item()
            group_freq[prompt_idx][group_idx] += 1
        
        # Compute weights: weight[i] = 1.0 / (freq[i] ** weight_power)
        weights = torch.zeros_like(scores)
        for i in range(bsz):
            prompt_idx = index[i]  # index is numpy array of strings/objects
            group_idx = group_index[i].item()  # group_index is tensor, need .item()
            freq = group_freq[prompt_idx][group_idx]
            weights[i] = 1.0 / (freq ** weight_power)
        
        # Combine GRPO advantages with diversity weights (MULTIPLICATION)
        combined_advantages = grpo_advantages * weights
        combined_advantages = combined_advantages.unsqueeze(-1).tile([1, response_length]) * eos_mask
        
        # Also return the individual components for metrics
        grpo_advantages_expanded = grpo_advantages.unsqueeze(-1).tile([1, response_length]) * eos_mask
        weights_expanded = weights.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return combined_advantages, combined_advantages, {
        'grpo_advantages': grpo_advantages_expanded,
        'diversity_weights': weights_expanded,
        'grpo_advantages_scalar': grpo_advantages,  # scalar version for easier metrics
        'diversity_weights_scalar': weights         # scalar version for easier metrics
    }


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def compute_policy_loss(old_log_prob, log_prob, advantages, eos_mask, cliprange):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange: (float)
            The clip range used in PPO. See https://arxiv.org/abs/1707.06347

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac: (float)
            a float number indicating the fraction of policy gradient loss being clipped

    """
    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, eos_mask)

    pg_losses = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)

    pg_loss = verl_F.masked_mean(torch.max(pg_losses, pg_losses2), eos_mask)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses).float(), eos_mask)
    return pg_loss, pg_clipfrac, ppo_kl


def compute_entropy_loss(logits, eos_mask):
    """Compute Categorical entropy loss

    Args:
        logits: `(torch.Tensor)`
            shape: (bs, response_length, vocab_size)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = verl_F.masked_mean(entropy, mask=eos_mask)
    return entropy_loss


def compute_value_loss(vpreds, returns, values, eos_mask, cliprange_value):
    """Compute the value loss. Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (`torch.FloatTensor`):
            Predicted values of the value head, shape (`batch_size`, `response_length`)
        values (`torch.FloatTensor`):
            Old values of value head, shape (`batch_size`, `response_length`)
        returns: (`torch.FloatTensor`):
            Ground truth returns, shape (`batch_size`, `response_length`)

    Returns:
        vf_loss: a scalar (`torch.FloatTensor`):
            value function loss
        vf_clipfrac: a float
            The ratio of vf being clipped

    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns)**2
    vf_losses2 = (vpredclipped - returns)**2
    vf_loss = 0.5 * verl_F.masked_mean(torch.max(vf_losses1, vf_losses2), eos_mask)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), eos_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104

    Args:
        logprob:
        ref_logprob:

    Returns:

    """
    if kl_penalty == "kl":
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty == "mse":
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty == 'low_var_kl':
        kl = ref_logprob - logprob
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError
