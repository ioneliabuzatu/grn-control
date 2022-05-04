import experiment_buddy
from stable_baselines3.common.env_util import make_vec_env

import gym_gene_control  # noqa
from src.rl_agents import train_a2c
from src.rl_agents import train_ddpg
from src.rl_agents import train_ppo
from src.rl_agents import train_sac
from src.rl_agents import train_td3
from src.zoo_functions import open_datasets_json

import stable_baselines3.common.utils
from stable_baselines3.common.callbacks import BaseCallback
import stable_baselines3.common.on_policy_algorithm

rl_train_agents_json_params = open_datasets_json(filepath="data/rl_agents_params.json")

agent_to_train = [train_sac, train_ppo, train_ddpg, train_td3, train_a2c]

config = {
    "policy_type": "MlpPolicy",
}

experiment_buddy.register_defaults(config)


class SensitivityCallback(BaseCallback):
    def _on_step(self) -> bool:
        obs = self.locals["new_obs"]
        obs_tensor = stable_baselines3.common.utils.obs_as_tensor(obs, self.model.device).reshape(-1, *obs.shape[-2:])
        obs_tensor = obs_tensor.requires_grad_(True)
        if isinstance(self.model, stable_baselines3.common.on_policy_algorithm.OnPolicyAlgorithm):
            actions, values, log_probs = self.model.policy(obs_tensor)
        else:
            actions = self.model.policy(obs_tensor)
            values = self.model.critic(obs_tensor, actions)
            if isinstance(values, tuple):
                values = (values[0] + values[1]) / 2  # for SAC

        # actions.abs().mean().backward()
        values.abs().mean().backward()
        for gene_idx, gene in enumerate(obs_tensor.grad.mean(0)):
            print(f"Gene:{gene_idx}", gene[0])
        return True


for agent in agent_to_train:
    agent_params_kwargs = rl_train_agents_json_params[agent.__name__]
    agent_params_kwargs["model_learn_kwargs"]["callback"] = SensitivityCallback()
    agent_params_kwargs["env_name"] = "gene_control-simple-v0"
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
