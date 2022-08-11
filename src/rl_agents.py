"""RL agents taken from: https://stable-baselines3.readthedocs.io/en/master/index.html"""
import seaborn as sns
import gym
import jax.numpy as jnp
import numpy as np
import stable_baselines3.common.callbacks
import stable_baselines3.common.on_policy_algorithm
import stable_baselines3.common.utils
import wandb
from stable_baselines3 import A2C
from stable_baselines3 import DDPG
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3 import TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import StopTrainingOnMaxEpisodes  # noqa
from stable_baselines3.common.callbacks import CallbackList
import matplotlib.pyplot as plt
import gym_gene_control  # noqa

__all__ = [
    "predict",
    "train_a2c",
    "train_ddpg",
    "train_ppo",
    "train_sac",
    "train_td3",
]

genes_names = ['Cdkn2a', 'Xist', 'Sox2', 'Nanog', 'Tdgf1', 'Zfp42', 'Fmr1nb', 'Ooep', 'Tcl1',
               'Obox6', 'Klf4', 'Esrrb', 'Dppa4', 'Myc', 'Lncenc1', 'Sohlh2', 'Pou5f1', 'Gdf9']

{'Cdkn2a': 0,
 'Xist': 1,
 'Sox2': 2,
 'Nanog': 3,
 'Tdgf1': 4,
 'Zfp42': 5,
 'Fmr1nb': 6,
 'Ooep': 7,
 'Tcl1': 8,
 'Obox6': 9,
 'Klf4': 10,
 'Esrrb': 11,
 'Dppa4': 12,
 'Myc': 13,
 'Lncenc1': 14,
 'Sohlh2': 15,
 'Pou5f1': 16,
 'Gdf9': 17}


def plot_sensitivity(model, obs):
    obs_tensor = stable_baselines3.common.utils.obs_as_tensor(obs, model.device).reshape(-1, *obs.shape[-2:])
    obs_tensor = obs_tensor.requires_grad_(True)
    if isinstance(model, stable_baselines3.common.on_policy_algorithm.OnPolicyAlgorithm):
        actions, mean_value, log_probs = model.policy(obs_tensor)
    else:
        actions = model.policy(obs_tensor)
        values = model.critic(obs_tensor, actions)
        if not isinstance(values, tuple):
            values = (values,)
        mean_value = 0.
        for value in values:
            mean_value += value
        mean_value /= len(values)

    mean_value.abs().mean().backward()
    batch_sensitivity = obs_tensor.grad.mean(0).detach().cpu().numpy()
    batch_sensitivity /= (np.linalg.norm(batch_sensitivity))  # should account for the lr ~3e-4
    # batch_sensitivity *= 10  # just because
    # cell_type = batch_sensitivity[target_cell_type]
    return batch_sensitivity


def predict(env, model, run, time_steps: int = 10):
    env.count_resets = 0
    obs = env.reset()
    batch_sensitivity = plot_sensitivity(model, np.array(obs))

    for step in range(0, time_steps):
        actions, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(actions)

        batch_sensitivity += plot_sensitivity(model, np.array(obs))

        print("predict step", step, "reward", rewards, "action", actions.round(3))
        if rewards.dtype == jnp.float32:
            run.log({"reward/prediction reward": rewards}, step=step)
        else:
            run.log({"reward/prediction reward": rewards[0]}, step=step)

    batch_sensitivity /= time_steps
    batch_sensitivity = np.abs(batch_sensitivity)
    run.log({f'sensitivity_analysis/{model.__class__.__name__}': wandb.plots.HeatMap(
        genes_names,
        ['D0', 'iPSC'],
        batch_sensitivity.T, show_text=False)})

    heatmap_kwargs = {
        'linewidth': 5, 'xticklabels': ['D0', 'iPSC'], 'cbar_kws': {"shrink": .7}, 'square': True, 'cmap': 'viridis'}
    heatmap_actions = sns.heatmap(batch_sensitivity, **heatmap_kwargs)
    run.log({"+/heatmap_actions": wandb.Image(heatmap_actions)}, step=0)
    plt.close()


def train_a2c(run, agent_kwargs, generals_kwargs):
    env = make_vec_env(
        agent_kwargs['env_name'], n_envs=agent_kwargs['n_envs'],
        env_kwargs={'wandb_writer': run, **generals_kwargs}
    )
    model = A2C(env=env, tensorboard_log=f"runs/{run.id}", **agent_kwargs['model_load_kwargs'])
    model.learn(**agent_kwargs['model_learn_kwargs'], callback=CallbackList([
        WandbCallback(**agent_kwargs['model_learn_wandb_callback_kwargs'], model_save_path=f"models/wandb/{run.id}"),
    ])
                )
    print("END LEARN.")
    model.save(**agent_kwargs['model_save_kwargs'])
    predict(env, model, run, **agent_kwargs['model_predict_kwargs'])


def train_ddpg(run, agent_kwargs, general_kwargs):
    env = gym.make(agent_kwargs['env_name'], **general_kwargs, wandb_writer=run)
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    model = DDPG(env=env, action_noise=action_noise, tensorboard_log=f"runs/{run.id}",
                 **agent_kwargs['model_load_kwargs'])
    model.learn(
        **agent_kwargs['model_learn_kwargs'],
        callback=WandbCallback(**agent_kwargs['model_learn_wandb_callback_kwargs'],
                               model_save_path=f"models/wandb/{run.id}")
    )
    model.save(**agent_kwargs['model_save_kwargs'])
    env = model.get_env()
    predict(env, model, run, **agent_kwargs['model_predict_kwargs'])


def train_ppo(run, agent_kwargs, generals_kwargs):
    env = make_vec_env(
        agent_kwargs['env_name'], n_envs=agent_kwargs['n_envs'],
        env_kwargs={'wandb_writer': run, **generals_kwargs}
    )
    model = PPO(env=env, **agent_kwargs['model_load_kwargs'], tensorboard_log=f"runs/{run.id}")
    model.learn(
        **agent_kwargs['model_learn_kwargs'],
        callback=WandbCallback(
            **agent_kwargs['model_learn_wandb_callback_kwargs'],
            model_save_path=f"models/wandb/{run.id}"
        )
    )
    model.save(**agent_kwargs['model_save_kwargs'])
    predict(env, model, run, **agent_kwargs['model_predict_kwargs'])


def train_sac(run, kwargs, general_kwargs):
    env = gym.make(kwargs['env_name'], **general_kwargs, wandb_writer=run)
    model = SAC(env=env, tensorboard_log=f"runs/{run.id}", **kwargs['model_load_kwargs'])
    model.learn(
        **kwargs['model_learn_kwargs'],
        callback=WandbCallback(**kwargs['model_learn_wandb_callback_kwargs'], model_save_path=f"models/wandb/{run.id}")
    )
    model.save(**kwargs['model_save_kwargs'])
    predict(env, model, run, **kwargs['model_predict_kwargs'])


def train_td3(run, kwargs, general_kwargs):
    env = gym.make(kwargs['env_name'], **general_kwargs, wandb_writer=run)
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    model = TD3(env=env, action_noise=action_noise, tensorboard_log=f"runs/{run.id}", **kwargs['model_load_kwargs'])
    model.learn(
        **kwargs['model_learn_kwargs'],
        callback=WandbCallback(**kwargs['model_learn_wandb_callback_kwargs'], model_save_path=f"models/wandb/{run.id}")
    )
    model.save(**kwargs['model_save_kwargs'])
    env = model.get_env()
    predict(env, model, run, **kwargs['model_predict_kwargs'])