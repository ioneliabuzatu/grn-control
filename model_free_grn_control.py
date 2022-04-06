import sys
import experiment_buddy
import wandb
from stable_baselines3.common.env_util import make_vec_env
from wandb.integration.sb3 import WandbCallback
import time
import gym_gene_control  # noqa

from src.rl_agents import train_ppo
from src.rl_agents import train_ddpg
from src.rl_agents import train_td3
from src.rl_agents import train_sac
import os

agent_to_train = [train_sac, train_ppo, train_ddpg, train_td3]
env_name = "gene_control-v0"  # "gene_control-simple-v0"

config = {
    "policy_type": "MlpPolicy",
    "total_time_steps": 25000,
    "env_name": env_name,
}

experiment_buddy.register_defaults(config)

env = make_vec_env(env_name, n_envs=2)  # Parallel environments

for agent in agent_to_train:
    buddy = experiment_buddy.deploy(
        host="", disabled=False,
        wandb_kwargs={
            'sync_tensorboard': True, 'monitor_gym': True, 'save_code': True, 'entity': 'control-grn', 'project': 'RL'
        },
        experiment_id=agent.__name__
    )
    run = buddy.run

    agent(run)

    print("Done training", agent.__name__)
