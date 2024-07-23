# rl_social_dilemma

Reinforcement learning project with MeltingPot.

## Getting Started

### Choosing Conda environment

Pick a specification from `./environments/` folder. `environment-forge` is recommended, although prioritizes heavy use of Forge, while `environment-anaconda` moves bulk of package installation onto in-environment `pip`, although still requires access to Forge to complete.

```shell
conda env create -f environment-{variant}.yml
```
(Notice: environment specifications in `./environments/swap-pre-meltingpot/` are experimental optimizations and do not contain all dependencies or MeltingPot, use at your own risk.)

### ~~Running tests notebook~~

### Running code notebook

Open `init.ipynb` in your preferred editor of choice, make sure you have your chosen `RL` environment activated. 