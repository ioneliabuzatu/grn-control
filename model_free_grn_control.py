import experiment_buddy
from stable_baselines3.common.env_util import make_vec_env

import gym_gene_control  # noqa
from src.rl_agents import train_a2c
from src.rl_agents import train_ddpg
from src.rl_agents import train_ppo
from src.rl_agents import train_sac
from src.rl_agents import train_td3
from src.zoo_functions import open_datasets_json

rl_train_agents_json_params = open_datasets_json(filepath="data/rl_agents_params.json")

agent_to_train = [train_a2c]  # train_td3, train_sac, train_ppo    # train_ddpg,

config = {
    "policy_type": "MlpPolicy",
}

experiment_buddy.register_defaults(config)

for agent in agent_to_train:
    agent_params_kwargs = rl_train_agents_json_params[agent.__name__]
    # agent_params_kwargs["env_name"] = "gene_control-simple-v0"
    buddy = experiment_buddy.deploy(
        host="", disabled=False,
        wandb_kwargs={
            'sync_tensorboard': True,
            'monitor_gym': True,
            'save_code': True,
            'entity': 'control-grn',
            'project': 'RL',
            'reinit': True
        },
        wandb_run_name=agent.__name__
    )
    run = buddy.run

    agent(run, agent_params_kwargs)

    print("Done training", agent.__name__)
