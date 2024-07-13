# rl_social_dilemma

Zainstaluj pipem:
- ray
- pettingzoo
- gymnasium
- numpy 1.26.0 (nie 2.0)

Odpal:
- dla prisoner dillema:
  python run.py --enable-new-api-stack --num-agents=2
- dla rock paper scissor:
  python rps.py --enable-new-api-stack --num-agents=2

Przydatne źródła:
- https://docs.ray.io/en/latest/rllib/rllib-env.html#pettingzoo-multi-agent-environments
- https://github.com/ray-project/ray/blob/master/rllib/examples/multi_agent/rock_paper_scissors_heuristic_vs_learned.py
- https://github.com/ray-project/ray/blob/master/rllib/examples/multi_agent/rock_paper_scissors_learned_vs_learned.py
- https://github.com/tianyu-z/pettingzoo_dilemma_envs
