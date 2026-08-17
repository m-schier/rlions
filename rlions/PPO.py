#  RLIonS - Reinforcement Learning Ion Shuttling Compiler
#  Copyright (C) 2026 Maximilian Schier, Lea Richtmann
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as published
#  by the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

import jax
import jax.numpy as jnp
import chex
from typing import Any, Callable
import flax.linen as nn
import flax

from typing import Tuple
from functools import partial

import optax

from rlions.JaxRolloutMetrics import RolloutMetricState, rm_init, rm_update, rm_get_episodic_return, rm_get_episode_length, rm_get_episode_duration


@chex.dataclass(frozen=True)
class PpoState:
    env_state: Any
    variables: Any
    opt_state: Any
    rollout_metrics: RolloutMetricState
    train_steps: Any


@chex.dataclass(frozen=True)
class PpoImpl:
    init: Callable
    act_deterministic: Callable
    act_stochastic: Callable
    train_step: Callable


def calculate_gae(rewards, values, terminations, durations, gamma, gae_lambda, truncations=None):
    # "High Dimensional Continuous Control using Generalized Advantage Estimation", Schulman et al.

    # TODO: Not supporting environment truncations for now
    chex.assert_equal(truncations, None)

    chex.assert_shape(gamma, ())
    chex.assert_shape(gae_lambda, ())
    chex.assert_equal(rewards.shape[0] + 1, values.shape[0])
    chex.assert_equal(rewards.shape[1:], values.shape[1:])
    chex.assert_equal_shape([rewards, terminations, durations])

    n = rewards.shape[0]

    def _body_fun(t, arg):
        t = n - t - 1  # Iterate in reverse

        _advantages, _last_advantage = arg

        next_value = values[t + 1]

        delta = jax.lax.select(
            terminations[t],
            rewards[t] - values[t],
            rewards[t] - values[t] + (gamma ** durations[t]) * next_value,
        )
        curr_advantage = jax.lax.select(
            terminations[t],
            delta,
            delta + ((gamma * gae_lambda) ** durations[t]) * _last_advantage,
        )

        return _advantages.at[t].set(curr_advantage), curr_advantage

    advantages, _ = jax.lax.fori_loop(
        0, n, _body_fun, (jnp.empty_like(rewards), jnp.zeros_like(rewards[0]))
    )

    return advantages


@chex.dataclass(frozen=True)
class RolloutResult:
    states: Any
    observations: Any
    actions: jax.Array
    log_probs: jax.Array
    rewards: jax.Array
    terminated: jax.Array
    truncated: jax.Array
    durations: jax.Array
    values: jax.Array
    weight: Any


def rollout(env, obs_fn, policy, rng, state, steps=100, n_envs=1, truncations_are_terminations: bool = False) -> RolloutResult:
    """
    Rollout a trajectory
    :param env: Non-vectorized environment implementation
    :param policy: Policy function taking unvectorized (rng, state) and returning unvectorized
    (action, log_prob, value of given state). Note that the implementation is unaware regarding masking and obs shaping.
    :param rng: RNG key
    :param state: Vectorized state which must agree with `n_envs`
    :param steps: Number of steps to unroll
    :param n_envs: Number of parallel envs already contained in the state
    :param truncations_are_terminations: If True, handle truncations as terminations. If False, on truncated (and not
    terminated), collect the final obs and reset on the next step.
    :return:
    """

    from .Util import broadcast_left

    vec_policy = jax.vmap(policy)
    vec_reset = jax.vmap(env.reset)
    vec_step = jax.vmap(env.step)

    # Pre-flight to determine shapes
    dummy_obs = obs_fn(state)
    dummy_logits, dummy_value = vec_policy(state, dummy_obs)

    # TODO: Correctly unrolling with truncations is a bit more tricky as we need to wait one step before resetting such
    # TODO: that the value can be used, so we are just ignoring this problem for now
    # result_truncated = jnp.empty((steps, n_envs), jnp.bool)

    init_val = (
        rng,
        RolloutResult(
            states=jax.tree.map(lambda x: jnp.empty((steps + 1,) + x.shape, x.dtype).at[0].set(x), state),
            observations=jax.tree.map(lambda x: jnp.empty((steps + 1,) + x.shape, x.dtype), dummy_obs),
            actions=jnp.empty((steps,) + dummy_logits.shape[:-1], jnp.int32),
            log_probs=jnp.empty((steps,) + dummy_logits.shape[:-1], dummy_logits.dtype),
            rewards=jnp.empty((steps, n_envs)),
            terminated=jnp.empty((steps, n_envs), jnp.bool),
            truncated=jnp.empty((steps, n_envs), jnp.bool),
            durations=jnp.empty((steps, n_envs), jnp.float32),
            weight=jnp.empty((steps, n_envs), jnp.float32),
            values=jnp.empty((steps + 1,) + dummy_value.shape, dummy_value.dtype),
        ),
    )

    def _body(i, val):
        iter_rng, rr = val

        iter_rng, policy_rng, reset_rng = jax.random.split(iter_rng, 3)

        current_state = jax.tree.map(lambda x: x[i], rr.states)
        current_obs = obs_fn(current_state)
        logits, value = vec_policy(current_state, current_obs)
        action = jax.random.categorical(policy_rng, logits)
        log_prob = jax.nn.log_softmax(logits)[tuple(jnp.indices(action.shape)) + (action,)]

        rr = rr.replace(
            observations=jax.tree.map(lambda o, el: o.at[i].set(el), rr.observations, current_obs),
            actions=rr.actions.at[i].set(action),
            log_probs=rr.log_probs.at[i].set(log_prob),
            values=rr.values.at[i].set(value),
        )

        stepped_state, reward, terminated, truncated, duration = vec_step(current_state, action)

        # TODO: Think about whether duration handling is correct in truncation handling
        if truncations_are_terminations:
            wants_reset = terminated | truncated
            recorded_terminated = terminated | truncated
            recorded_truncated = jnp.zeros_like(truncated)
            recorded_weight = jnp.ones(n_envs)
            recorded_reward = reward
        else:
            last_step_truncated = jax.lax.select(
                i > 0,
                rr.truncated[i - 1],
                jnp.zeros((n_envs,), jnp.bool),
            )

            # Reset on termination or following a truncation on the next step
            wants_reset = terminated | last_step_truncated
            # We manually insert a terminate signal on the invalid transition to force cutting traces
            recorded_terminated = terminated | last_step_truncated
            # In case an environment keeps returning a truncate signal do not record it to prevent spurious resets
            recorded_truncated = truncated & (~last_step_truncated)
            # Record the reward normally, unless the last step was a truncation. If the last step was a truncation,
            # this step is recorded as a termination, and on termination GAE calculates advantage as reward - value.
            # Since we want to propagate a 0 back in time as this step should be ignored, both must be equal.
            # (And we can't change the value because that is actually required)
            recorded_reward = jax.lax.select(last_step_truncated, value, reward)
            recorded_weight = 1.0 - last_step_truncated

        reset_state = vec_reset(jax.random.split(reset_rng, n_envs))

        selected_state = jax.tree.map(lambda r, s: jnp.where(broadcast_left(wants_reset, r.shape), r, s), reset_state, stepped_state)

        rr = rr.replace(
            states=jax.tree.map(lambda x, y: x.at[i + 1].set(y), rr.states, selected_state),
            rewards=rr.rewards.at[i].set(recorded_reward),
            terminated=rr.terminated.at[i].set(recorded_terminated),
            truncated=rr.truncated.at[i].set(recorded_truncated),
            durations=rr.durations.at[i].set(duration),
            weight=rr.weight.at[i].set(recorded_weight),
        )

        return iter_rng, rr

    rng, result = jax.lax.fori_loop(0, steps, _body, init_val)

    # Fill ultimate value
    last_state = jax.tree.map(lambda x: x[-1], result.states)
    last_obs = obs_fn(last_state)
    _, value = vec_policy(last_state, last_obs)
    result = result.replace(
        values=result.values.at[-1].set(value),
        observations=jax.tree.map(lambda o, el: o.at[-1].set(el), result.observations, last_obs),
    )

    return result


def mul_exp(x: jax.Array, logp: jax.Array) -> jax.Array:
    p = jnp.exp(logp)
    x = jnp.where(p == 0, 0.0, x)
    return x * p


def make_ppo(obs_fn, network: nn.Module, mask_fn, env, n_envs=80, n_steps=100, gamma=0.99, gae_lambda=0.96, mini_batch_size=1024,
             norm_psi=True, ent_coef=.0, epsilon=.2, clip_v_loss=True, vf_coef=.5, max_epochs=10, stop_kl=None,
             lr=2.5e-4):
    assert 0. < gae_lambda < 1.

    opt = optax.inject_hyperparams(partial(optax.adam, eps=1e-5, b1=.9))(learning_rate=lr if not callable(lr) else lr(0))

    def init(rng):
        network_rng, env_rng = jax.random.split(rng)

        env_state = jax.vmap(env.reset)(jax.random.split(env_rng, n_envs))
        variables = network.init(network_rng, obs_fn(env_state))

        return PpoState(
            env_state=env_state,
            variables=variables,
            opt_state=opt.init(variables['params']),
            rollout_metrics=rm_init(n_envs),
            train_steps=0,
        )

    def act_stochastic(state, rng, env_state, temperature=1.0):
        logits, _ = network.apply(state.variables, obs_fn(env_state))
        logits = logits * temperature

        if mask_fn is not None:
            valid_mask = mask_fn(env_state)
            chex.assert_equal_shape([valid_mask, logits])
            logits = jnp.where(valid_mask, logits, jnp.full_like(logits, -jnp.inf))

        return jax.random.categorical(rng, logits)

    def act_deterministic(state, env_state):
        logits, _ = network.apply(state.variables, obs_fn(env_state))

        if mask_fn is not None:
            valid_mask = mask_fn(env_state)
            chex.assert_equal_shape([valid_mask, logits])
            logits = jnp.where(valid_mask, logits, jnp.full_like(logits, -jnp.inf))

        return jnp.argmax(logits, axis=-1)

    def _train_epoch(state, rng, rr: RolloutResult, action_masks):
        obs = rr.observations
        # 1. Calculate Psis
        psis = calculate_gae(rewards=rr.rewards, values=rr.values, terminations=rr.terminated, durations=rr.durations,
                             gamma=gamma, gae_lambda=gae_lambda)
        # Compare https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/ppo2/runner.py#L50
        returns = psis + rr.values[:-1]

        chex.assert_equal_shape([returns, rr.rewards, psis])

        rng, loop_rng, shuffle_rng = jax.random.split(rng, 3)

        # Shuffle mini batched data
        learn_t = n_steps
        n_samples = learn_t * rr.rewards.shape[1]
        shuffle_idxs = jax.random.permutation(shuffle_rng, n_samples)

        # Flatten num_envs x horizon for mini-batching
        # mb_obs = obs[:learn_t].reshape((-1,) + obs.shape[2:])[shuffle_idxs]
        mb_obs = jax.tree_util.tree_map(lambda o: o[:learn_t].reshape((-1,) + o.shape[2:])[shuffle_idxs], obs)
        mb_actions = rr.actions[:learn_t].reshape((-1,) + rr.actions.shape[2:])[shuffle_idxs]
        mb_action_masks = None if action_masks is None else action_masks[:learn_t].reshape((-1,) + action_masks.shape[2:])[shuffle_idxs]
        mb_returns = returns[:learn_t].reshape((-1,) + returns.shape[2:])[shuffle_idxs]
        mb_psis = psis[:learn_t].reshape((-1,) + psis.shape[2:])[shuffle_idxs]
        mb_log_probs_old = rr.log_probs[:learn_t].reshape((-1,) + rr.log_probs.shape[2:])[shuffle_idxs]
        mb_values_old = rr.values[:learn_t].reshape((-1,) + rr.values.shape[2:])[shuffle_idxs]
        mb_weights = rr.weight[:learn_t].reshape((-1,) + rr.weight.shape[2:])[shuffle_idxs]

        chex.assert_equal_shape([mb_returns, mb_psis, mb_log_probs_old, mb_values_old])
        chex.assert_rank(mb_returns, 1)

        n_batches = n_samples // mini_batch_size

        if n_batches == 0:
            raise ValueError(f"Have empty batches with {n_samples = }, {mini_batch_size = }")

        def _mb_loop_body(i, args):
            _rng, _state, _loss, _log_probs, _pi_clip_ratio, _vf_clip_ratio, _loss_actor, _loss_vn = args
            _rng, _step_rng = jax.random.split(_rng)

            start = i * mini_batch_size

            # Select the next slice from the prepared mini-batch arrays
            slice_obs = jax.tree_util.tree_map(lambda o: jax.lax.dynamic_slice_in_dim(o, start, mini_batch_size), mb_obs)
            slice_actions = jax.lax.dynamic_slice_in_dim(mb_actions, start, mini_batch_size)
            slice_action_masks = None if mb_action_masks is None else jax.lax.dynamic_slice_in_dim(mb_action_masks, start, mini_batch_size)
            slice_psis = jax.lax.dynamic_slice_in_dim(mb_psis, start, mini_batch_size)
            slice_lp_old = jax.lax.dynamic_slice_in_dim(mb_log_probs_old, start, mini_batch_size)
            slice_returns = jax.lax.dynamic_slice_in_dim(mb_returns, start, mini_batch_size)
            slice_values_old = jax.lax.dynamic_slice_in_dim(mb_values_old, start, mini_batch_size)
            slice_weights = jax.lax.dynamic_slice_in_dim(mb_weights, start, mini_batch_size)

            if norm_psi:
                slice_psis = (slice_psis - slice_psis.mean()) / (
                            slice_psis.std() + 1e-8)

            variables, params_init = flax.core.pop(state.variables, 'params')

            def objective(p):
                logits, v_pred = network.apply({**variables, 'params': p}, slice_obs)

                assert (mask_fn is None) == (slice_action_masks is None)

                if mask_fn is not None:
                    chex.assert_equal_shape([logits, slice_action_masks])
                    logits = jnp.where(slice_action_masks, logits, jnp.full_like(logits, -jnp.inf))

                all_log_probs = jax.nn.log_softmax(logits)
                log_probs = all_log_probs[tuple(jnp.indices(slice_actions.shape)) + (slice_actions,)]

                ratio = jnp.exp(log_probs - slice_lp_old)  # = (pi() / pi_old())

                # Policy loss
                pg_loss1 = -slice_psis * ratio
                pg_loss2 = -slice_psis * jnp.clip(ratio, 1 - epsilon, 1 + epsilon)
                pg_loss = jnp.mean(jnp.maximum(pg_loss1, pg_loss2) * slice_weights)

                # This is slightly more complicated than PPO because PPO calculates entropy in closed form, which
                # limits actor distributions. We estimate entropy, which requires sampling according to the new policy
                chex.assert_scalar_non_negative(ent_coef)
                if ent_coef > 0:
                    entropy_loss = jnp.mean(-jnp.sum(mul_exp(all_log_probs, all_log_probs), axis=-1) * slice_weights)
                else:
                    entropy_loss = 0.

                is_pi_clipped = jnp.abs(ratio - 1.) > epsilon

                chex.assert_equal_shape([log_probs, slice_lp_old, slice_psis, ratio, is_pi_clipped])

                # Value function loss
                if clip_v_loss:
                    v_loss_unclipped = (v_pred - slice_returns) ** 2
                    v_clipped = slice_values_old + jnp.clip(
                        v_pred - slice_values_old,
                        -epsilon,
                        epsilon,
                    )
                    v_loss_clipped = (v_clipped - slice_returns) ** 2
                    v_loss_max = jnp.maximum(v_loss_unclipped, v_loss_clipped)
                    is_vf_clipped = v_loss_clipped > v_loss_unclipped
                    __loss_vn = 0.5 * jnp.mean(v_loss_max * slice_weights)
                else:
                    is_vf_clipped = jnp.zeros_like(v_pred)
                    __loss_vn = 0.5 * jnp.mean((slice_returns - v_pred) ** 2 * slice_weights)

                return pg_loss - ent_coef * entropy_loss + vf_coef * __loss_vn, (log_probs, is_pi_clipped, is_vf_clipped, pg_loss, __loss_vn)

            (new_loss, (new_log_probs, new_pi_clipped, new_vf_clipped, new_loss_actor, new_loss_vn)), grads \
                = jax.value_and_grad(objective, has_aux=True)(params_init)
            updates, new_opt_state = opt.update(grads, _state.opt_state, params_init)
            new_params = optax.apply_updates(params_init, updates)
            new_variables = {**variables, 'params': new_params}

            return _rng, _state.replace(
                opt_state=new_opt_state,
                variables=new_variables,
            ), _loss + new_loss, _log_probs + jnp.mean(new_log_probs), _pi_clip_ratio + jnp.sum(new_pi_clipped), \
                   _vf_clip_ratio + jnp.sum(new_vf_clipped), _loss_actor + new_loss_actor, _loss_vn + new_loss_vn

        _, state, epoch_loss, aux_log_probs, pi_clip_ratio, vf_clip_ratio, loss_actor, loss_vn = \
            jax.lax.fori_loop(
                0, n_batches,
                _mb_loop_body,
                (loop_rng, state, 0., 0., 0., 0., 0., 0.)
            )

        epoch_loss = epoch_loss / n_batches
        loss_actor = loss_actor / n_batches
        loss_vn = loss_vn / n_batches
        entropy = -aux_log_probs / n_batches
        pi_clip_ratio = pi_clip_ratio / (n_batches * mini_batch_size)
        vf_clip_ratio = vf_clip_ratio / (n_batches * mini_batch_size)

        return state, epoch_loss, entropy, pi_clip_ratio, vf_clip_ratio, loss_actor, loss_vn

    @jax.jit
    def train_step(rng, state: PpoState) -> Tuple[PpoState, dict]:
        from .Util import broadcast_left

        def _wrapped_policy(_s, _o):
            logits, values = network.apply(state.variables, _o)

            if mask_fn is not None:
                valid_mask = mask_fn(_s)
                chex.assert_equal_shape([valid_mask, logits])
                logits = jnp.where(valid_mask, logits, jnp.full_like(logits, -jnp.inf))

            return logits, values

        # Maybe update LR
        if callable(lr):
            state.opt_state.hyperparams['learning_rate'] = lr(state.train_steps)

        rollout_rng, reset_rng, pi_rng = jax.random.split(rng, 3)
        rr = rollout(env, obs_fn, _wrapped_policy, rollout_rng, state.env_state, steps=n_steps, n_envs=n_envs)

        # Update rollout metrics
        new_rm = jax.lax.fori_loop(0, n_steps, lambda i, s: rm_update(s, rr.rewards[i], rr.terminated[i] | rr.truncated[i], rr.weight[i] > 0, duration=rr.durations[i]), state.rollout_metrics)

        psis = calculate_gae(rr.rewards, rr.values, rr.terminated, rr.durations, gamma, gae_lambda)
        chex.assert_equal_shape([psis, rr.rewards])

        explained_variance = 1 - jnp.var(psis) / jnp.var(rr.values[:-1] + psis)

        entropy_old = -jnp.mean(rr.log_probs)

        full_action_masks = None if mask_fn is None else jax.vmap(jax.vmap(mask_fn))(rr.states)  # Two additional axis

        # Train new policy
        def _epoch_body(arg):
            _state, _rng, _loss_pi, _loss_vn, _entropy, _pi_clip_ratio, _vf_clip_ratio, _kl, _i = arg

            _rng, _step_rng = jax.random.split(_rng)

            _state, _, new_entropy, new_pi_clip_ratio, new_vf_clip_ratio, new_loss_actor, new_loss_vn = \
                _train_epoch(_state, _step_rng, rr, full_action_masks)

            new_kl = 0.0

            return _state, _rng, _loss_pi + new_loss_actor, _loss_vn + new_loss_vn, _entropy + new_entropy, \
                   _pi_clip_ratio + new_pi_clip_ratio, _vf_clip_ratio + new_vf_clip_ratio, jnp.maximum(_kl, new_kl), _i + 1

        state, _, loss_pi, loss_vn, entropy, pi_clip_ratio, vf_clip_ratio, max_kl, pi_iter = jax.lax.while_loop(
            lambda arg: (arg[-1] < max_epochs) & ((stop_kl is None) or (arg[-2] < stop_kl)),
            _epoch_body,
            (state, pi_rng, 0., 0., 0., 0., 0., 0., 0)
        )

        loss_pi = loss_pi / pi_iter
        loss_vn = loss_vn / pi_iter
        pi_clip_ratio = pi_clip_ratio / pi_iter
        vf_clip_ratio = vf_clip_ratio / pi_iter

        # Save new initial states for next iteration. If a truncation happened exactly at the end, reset now
        # as the rollout function resets on the next transition and therefore has not in that case
        wants_reset = rr.truncated[-1]
        last_states = jax.tree.map(lambda x: x[-1], rr.states)
        reset_states = jax.vmap(env.reset)(jax.random.split(reset_rng, n_envs))
        used_next_states = jax.tree.map(lambda r, s: jnp.where(broadcast_left(wants_reset, r.shape), r, s), reset_states, last_states)

        state = state.replace(
            env_state=used_next_states,
            rollout_metrics=new_rm,
            train_steps=state.train_steps + 1,
        )

        metrics = {
            'avg_episodic_return': rm_get_episodic_return(state.rollout_metrics),
            'avg_episode_duration': rm_get_episode_duration(state.rollout_metrics),
            'avg_episode_length': rm_get_episode_length(state.rollout_metrics),
            'avg_truncation_prob': jnp.mean(rr.truncated),
            'avg_advantage': jnp.mean(psis),
            'entropy': entropy_old,
            'explained_variance': explained_variance,
            'pi_clip_ratio': pi_clip_ratio,
            'vf_clip_ratio': vf_clip_ratio,
            'max_kl': max_kl,
            'epochs': pi_iter,
            'learning_rate': state.opt_state.hyperparams['learning_rate'],
            'loss_pi': loss_pi,
            'loss_vn': loss_vn,
        }

        return state, metrics

    return PpoImpl(
        init=init,
        act_deterministic=act_deterministic,
        act_stochastic=act_stochastic,
        train_step=train_step,
    )


class SequentialPpoModule(nn.Module):
    n_act: int
    features: Tuple[int, ...] = (128, 128)
    shared: bool = True

    @nn.compact
    def __call__(self, obs):
        def _make_hidden():
            x = obs

            for f in self.features:
                x = nn.relu(nn.Dense(f)(x))

            return x

        val_nn = nn.Dense(1)
        logit_nn = nn.Dense(self.n_act, kernel_init=jax.nn.initializers.orthogonal(scale=0.01))

        if self.shared:
            hidden = _make_hidden()
            logits = logit_nn(hidden)
            values = val_nn(hidden).squeeze(-1)
        else:
            logits = logit_nn(_make_hidden())
            values = val_nn(_make_hidden()).squeeze(-1)

        return logits, values


class ResidualBlock(nn.Module):
    @nn.compact
    def __call__(self, x):
        *_, f = x.shape
        orig = x
        x = nn.relu(x)
        x = nn.Dense(f)(x)
        x = nn.relu(x)
        x = nn.Dense(f)(x)

        return orig + x


class ResidualPpoModule(nn.Module):
    n_act: int
    width: int
    blocks: int
    shared: bool = True

    @nn.compact
    def __call__(self, obs):
        def _make_hidden():
            x = nn.Dense(self.width)(obs)

            for _ in range(self.blocks):
                x = ResidualBlock()(x)

            return nn.relu(x)

        val_nn = nn.Dense(1)
        logit_nn = nn.Dense(self.n_act, kernel_init=jax.nn.initializers.orthogonal(scale=0.01))

        if self.shared:
            hidden = _make_hidden()
            logits = logit_nn(hidden)
            values = val_nn(hidden).squeeze(-1)
        else:
            logits = logit_nn(_make_hidden())
            values = val_nn(_make_hidden()).squeeze(-1)

        return logits, values
