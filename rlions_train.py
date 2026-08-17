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

import os.path
import pickle
import sys

import jax

from rlions.PPO import make_ppo, SequentialPpoModule, ResidualPpoModule


def make_obs_fn(args, adapter):
    if args.obs_encoder == "basic":
        from rlions.ObservationsImproved import make_encoder_basic

        return make_encoder_basic(
            adapter,
            lookahead=args.lookahead,
            max_ion_gate_count=args.obs_max_gate_count,
            observe_steps=args.obs_steps,
            op_bands=args.obs_op_bands,
            ion_gate_bands=args.obs_ion_gate_bands,
            encode_ion_gate_counts=False,
            number_encoder=args.obs_number_encoder,
        )
    elif args.obs_encoder.startswith('encoder_b'):
        fixed_depth = '_fd' in args.obs_encoder

        from rlions.ObservationsImproved import make_encoder3

        number_encoder = args.obs_number_encoder

        return make_encoder3(
            adapter,
            lookahead=args.lookahead,
            max_ion_gate_count=args.obs_max_gate_count,
            observe_steps=args.obs_steps,
            op_bands=args.obs_op_bands,
            ion_gate_bands=args.obs_ion_gate_bands,
            encode_ion_gate_counts=False,
            fixed_depth=fixed_depth,
            number_encoder=number_encoder,
        )
    else:
        raise ValueError(f"{args.obs_encoder = }")


def make_network(args, n_actions):
    if args.net_type == 'sequential':
        return SequentialPpoModule(n_act=n_actions, features=(args.net_width,) * args.net_depth, shared=args.net_shared)
    elif args.net_type == 'residual':
        return ResidualPpoModule(n_act=n_actions, width=args.net_width, blocks=args.net_depth, shared=args.net_shared)
    else:
        raise ValueError(f"{args.net_type = }")


def make_configured_ppo(args, env, chip):
    adapter = chip.make_observation_adapter()
    network = make_network(args, chip.n_actions)
    obs_fn = make_obs_fn(args, adapter)

    if env.reset:
        # Dump observation info if we can obtain a state
        n_features, = obs_fn(env.reset(jax.random.key(0))).shape
        print(f"Observation space features = {n_features}", file=sys.stderr)

    if args.mask_invalid:
        mask_fn = env.valid_mask
    else:
        mask_fn = None

    if args.decay_lr:
        lr = lambda s: args.lr * (1. - s / args.steps * 0.9)
    else:
        lr = args.lr

    gae_lambda = getattr(args, "gae_lambda", 0.96)
    vf_coef = getattr(args, "vf_coef", 0.5)
    clip_v_loss = getattr(args, "clip_v_loss", True)

    return make_ppo(obs_fn, network, mask_fn, env, n_envs=args.n_envs, n_steps=args.n_steps, gamma=args.gamma,
                    ent_coef=args.ent_coef, max_epochs=args.max_epochs, mini_batch_size=args.mini_batch_size,
                    epsilon=args.epsilon, lr=lr, gae_lambda=gae_lambda, vf_coef=vf_coef, clip_v_loss=clip_v_loss)


def main():
    from rlions.Logging.MlFlowLogger import MlFlowLogger
    from rlions.Logging.SmoothedMetrics import SmoothedMetrics
    from rlions.Chips import get_chip_by_name
    from argparse import ArgumentParser
    import numpy as np

    parser = ArgumentParser()
    # Strictly part of MDP but often considered PPO parameter
    parser.add_argument('--gamma', type=float, default=0.9995)
    # PPO parameters
    parser.add_argument('--gae_lambda', type=float, default=0.96)
    parser.add_argument('--ent_coef', type=float, default=1e-4)
    parser.add_argument('--epsilon', type=float, default=0.1)
    parser.add_argument('--n_envs', type=int, default=250)
    parser.add_argument('--n_steps', type=int, default=40)
    parser.add_argument('--max_epochs', type=int, default=4)
    parser.add_argument('--mini_batch_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=2.5e-4)
    parser.add_argument('--vf_coef', type=float, default=0.5)
    parser.add_argument('--no_clip_v_loss', action='store_false', dest='clip_v_loss')
    parser.add_argument('--no_decay_lr', action='store_false', dest='decay_lr')
    # Network architecture
    parser.add_argument('--net_type', choices=['residual', 'sequential'], default='residual')
    parser.add_argument('--net_depth', type=int, default=3)
    parser.add_argument('--net_width', type=int, default=512)
    parser.add_argument('--net_shared', action='store_true', help="Share parameters between policy and value heads")
    parser.add_argument('--no_mask', action='store_false', dest='mask_invalid', help="Do not mask invalid actions, instead they are no-ops. Not recommended.")
    # Observation encoding
    parser.add_argument('--no_obs_sort', action='store_false', dest='obs_sort_stack')
    parser.add_argument('--obs_encoder', choices=['encoder_b', 'encoder_b_fd', 'basic'], default='encoder_b_fd')
    parser.add_argument('--obs_number_encoder', choices=['linear', 'sinusoidal2'], default='sinusoidal2')
    parser.add_argument('--obs_max_op_count', type=int, default=20)
    parser.add_argument('--obs_op_bands', type=int, default=6)
    parser.add_argument('--obs_ion_gate_bands', type=int, default=7)
    parser.add_argument('--obs_max_gate_count', type=int, default=1250)
    parser.add_argument('--no_obs_clip', action='store_false', dest='obs_clip')
    parser.add_argument('--obs_steps', action='store_true', dest='obs_steps')
    # This incredibly confusing sounding option should only be temporary and we should pick one
    parser.add_argument('--no_obs_op_for_gate_count', action='store_false', dest='obs_op_for_gate_count')
    parser.add_argument('--lookahead', type=int, default=4)
    # Reward shaping
    parser.add_argument('--factor_completion', type=float, default=1.0)
    # Problem/environment settings
    parser.add_argument('--steps', type=int, default=1_000_000)
    parser.add_argument('--min_ion_count', type=int, default=None)
    parser.add_argument('--ion_count', type=int, default=50)
    parser.add_argument('--max_op_count', type=int, default=1700)
    parser.add_argument('--timeout', choices=['terminate_penalty', 'truncate'], default='truncate')
    # Chip configuration
    parser.add_argument('--chip', type=str, default='qvls_x_50')
    # A different shaped reward gamma than the MDP gamma modifies greediness of the agent. Higher shaped gamma causes
    # more immediate rewards to become overvalued, while lower shaped gamma causes more immediate rewards to be
    # undervalued.
    parser.add_argument('--shaped_gamma', type=float, default=1.0)
    parser.add_argument('--step_reward', type=float, default=-0.1)
    parser.add_argument('--tag', default=None, help="Arbitrary tag, not used by program but included in logs")

    args = parser.parse_args()
    args.improved_shape = True

    # Increment on breaking change
    args.version = 1

    if args.timeout != 'truncate' and args.shaped_gamma is not None:
        raise ValueError("Shaped gamma requires timeout mode of truncate to not have additional side effects")

    chip = get_chip_by_name(args.chip)
    env = chip.make_train_env(
        max_op_count=args.max_op_count,
        min_ion_count=args.min_ion_count,
        max_ion_count=args.ion_count,
        smdp_gamma=args.gamma,
        shaped_gamma=args.shaped_gamma,
        step_reward=args.step_reward,
        timeout=args.timeout,
        shape_factor_gates=args.factor_completion,
        shape_dynamic_factor=False,
    )

    args.seed = np.random.randint(0, 2 ** 31 - 1)
    rng = jax.random.key(args.seed)

    ppo = make_configured_ppo(args, env, chip)

    rng, init_rng = jax.random.split(rng)
    ppo_state = ppo.init(init_rng)

    # Dump some agent info
    from operator import mul, add
    from functools import reduce
    param_count = jax.tree.reduce(add, jax.tree.map(lambda p: reduce(mul, p.shape, 1), ppo_state.variables['params']), 0)
    print(f"Agent parameter count = {param_count}", file=sys.stderr)

    sm = SmoothedMetrics()

    with MlFlowLogger(config=vars(args)) as logger:
        def do_store():
            store_path = os.path.join(logger.local_storage, "agent.pckl")

            with open(store_path, "wb") as fp:
                pickle.dump({
                    "args": args,
                    "state": ppo_state,
                }, fp)

            print("Saved to:", os.path.abspath(store_path), file=sys.stderr)

        try:
            for i in range(args.steps):
                rng, step_rng = jax.random.split(rng)
                ppo_state, metrics = ppo.train_step(step_rng, ppo_state)
                sm.update(metrics)

                if i % 50 == 0:
                    metrics = {**sm.report()}

                    logger.log_dict(metrics, i)

                # Some intermediate saving
                if i > 0 and i % 10_000 == 0:
                    do_store()
        finally:
            do_store()


if __name__ == '__main__':
    main()
