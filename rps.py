"""A simple multi-agent env with two agents play rock paper scissors.

This demonstrates running two learning policies in competition, both using the same
RLlib algorithm (PPO by default).

The combined reward as well as individual rewards should roughly remain at 0.0 as no
policy should - in the long run - be able to learn a better strategy than chosing
actions at random. However, it could be possible that - for some time - one or the other
policy can exploit a "stochastic weakness" of the opponent policy. For example a policy
`A` learns that its opponent `B` has learnt to choose "paper" more often, which in
return makes `A` choose "scissors" more often as a countermeasure.
"""

import re

from pettingzoo.classic import rps_v2

from ray.rllib.connectors.env_to_module import FlattenObservations
from ray.rllib.core.rl_module.marl_module import MultiAgentRLModuleSpec
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.utils.test_utils import (
    add_rllib_example_script_args,
    run_rllib_example_script_experiment,
)

from ray.rllib.examples.rl_modules.classes import (
    AlwaysSameHeuristicRLM,
    BeatLastHeuristicRLM,
)
import gymnasium as gym
import random
from ray.tune.registry import get_trainable_cls, register_env


parser = add_rllib_example_script_args(
    default_iters=50,
    default_timesteps=200000,
    default_reward=6.0,
)
parser.add_argument(
    "--use-lstm",
    action="store_true",
    help="Whether to use an LSTM wrapped module instead of a simple MLP one. With LSTM "
    "the reward diff can reach 7.0, without only 5.0.",
)


register_env(
    "RockPaperScissors",
    lambda _: ParallelPettingZooEnv(rps_v2.parallel_env()),
)


if __name__ == "__main__":
    args = parser.parse_args()

    assert args.num_agents == 2, "Must set --num-agents=2 when running this script!"
    assert (
        args.enable_new_api_stack
    ), "Must set --enable-new-api-stack when running this script!"

#     base_config = (
#         get_trainable_cls(args.algo)
#         .get_default_config()
#         .environment("RockPaperScissors")
#         .env_runners(
#             env_to_module_connector=lambda env: (
#                 # `agent_ids=...`: Only flatten obs for the learning RLModule.
#                 FlattenObservations(multi_agent=True, agent_ids={"player_0"}),
#             ),
#         )
#         .multi_agent(
#             policies={"always_same", "beat_last", "learned"},
#             # Let learning Policy always play against either heuristic one:
#             # `always_same` or `beat_last`.
#             policy_mapping_fn=lambda aid, episode: (
#                 "learned"
#                 if aid == "player_0"
#                 else random.choice(["always_same", "beat_last"])
#             ),
#             # Must define this as both heuristic RLMs will throw an error, if their
#             # `forward_train` is called.
#             policies_to_train=["learned"],
#         )
#         .training(
#             vf_loss_coeff=0.005,
#         )
#         .rl_module(
#             model_config_dict={
#                 "use_lstm": args.use_lstm,
#                 # Use a simpler FCNet when we also have an LSTM.
#                 "fcnet_hiddens": [32] if args.use_lstm else [256, 256],
#                 "lstm_cell_size": 256,
#                 "max_seq_len": 15,
#                 "vf_share_layers": True,
#             },
#             rl_module_spec=MultiAgentRLModuleSpec(
#                 module_specs={
#                     "always_same": SingleAgentRLModuleSpec(
#                         module_class=AlwaysSameHeuristicRLM,
#                         observation_space=gym.spaces.Discrete(4),
#                         action_space=gym.spaces.Discrete(3),
#                     ),
#                     "beat_last": SingleAgentRLModuleSpec(
#                         module_class=BeatLastHeuristicRLM,
#                         observation_space=gym.spaces.Discrete(4),
#                         action_space=gym.spaces.Discrete(3),
#                     ),
#                     "learned": SingleAgentRLModuleSpec(),
#                 }
#             ),
#         )
#     )
#   # Make `args.stop_reward` "point" to the reward of the learned policy.
#     stop = {
#         "training_iteration": args.stop_iters,
#         "env_runner_results/module_episode_returns_mean/learned": args.stop_reward,
#         "num_env_steps_sampled_lifetime": args.stop_timesteps,
#     }

#     run_rllib_example_script_experiment(
#         base_config,
#         args,
#         stop=stop,
#         success_metric={
#             "env_runner_results/module_episode_returns_mean/learned": args.stop_reward,
#         },
#     )
  

    ###########################################################

    base_config = (
        get_trainable_cls(args.algo)
        .get_default_config()
        .environment("RockPaperScissors")
        .env_runners(
            env_to_module_connector=lambda env: FlattenObservations(multi_agent=True),
        )
        .multi_agent(
            policies={"p0", "p1"},
            # `player_0` uses `p0`, `player_1` uses `p1`.
            policy_mapping_fn=lambda aid, episode: re.sub("^player_", "p", aid),
        )
        .training(
            vf_loss_coeff=0.005,
        )
        .rl_module(
            model_config_dict={
                "use_lstm": args.use_lstm,
                # Use a simpler FCNet when we also have an LSTM.
                "fcnet_hiddens": [32] if args.use_lstm else [256, 256],
                "lstm_cell_size": 256,
                "max_seq_len": 15,
                "vf_share_layers": True,
            },
            rl_module_spec=MultiAgentRLModuleSpec(
                module_specs={
                    "p0": SingleAgentRLModuleSpec(),
                    "p1": SingleAgentRLModuleSpec(),
                }
            ),
        )
    )

    run_rllib_example_script_experiment(base_config, args)