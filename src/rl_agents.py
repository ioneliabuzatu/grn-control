"""RL agents taken from: https://stable-baselines3.readthedocs.io/en/master/index.html"""
import gym
import numpy as np
import stable_baselines3.common.callbacks

import gym_gene_control  # noqa
import jax.numpy as jnp

from stable_baselines3 import A2C
from stable_baselines3 import DDPG
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3 import TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise
from wandb.integration.sb3 import WandbCallback

__all__ = [
    "predict",
    "train_a2c",
    "train_ddpg",
    "train_ppo",
    "train_sac",
    "train_td3",
]


def predict(env, model, run, time_steps: int = 10):
    obs = env.reset()
    for step in range(time_steps):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)

        print("reward", rewards, "action", action.round(3))
        if rewards.dtype == jnp.float32:
            run.log({"prediction reward": rewards}, step=step)
        else:
            run.log({"prediction reward": rewards[0]}, step=step)


def train_a2c(run, kwargs):
    env = make_vec_env(kwargs['env_name'], n_envs=kwargs['n_envs'])
    model = A2C(env=env, tensorboard_log=f"runs/{run.id}", **kwargs['model_load_kwargs'])
    model.learn(
        callback=prepare_callbacks(kwargs, run),
        **kwargs['model_learn_kwargs'],
    )
    model.save(**kwargs['model_save_kwargs'])
    predict(env, model, run, **kwargs['model_predict_kwargs'])


def train_ddpg(run, kwargs):
    env = gym.make(kwargs['env_name'])
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    model = DDPG(env=env, action_noise=action_noise, tensorboard_log=f"runs/{run.id}", **kwargs['model_load_kwargs'])
    model.learn(
        callback=prepare_callbacks(kwargs, run),
        **kwargs['model_learn_kwargs'],
    )
    model.save(**kwargs['model_save_kwargs'])
    env = model.get_env()
    predict(env, model, run, **kwargs['model_predict_kwargs'])


def train_ppo(run, kwargs):
    env = make_vec_env(kwargs['env_name'], n_envs=kwargs['n_envs'])
    model = PPO(env=env, **kwargs['model_load_kwargs'], tensorboard_log=f"runs/{run.id}")
    model.learn(
        callback=prepare_callbacks(kwargs, run),
        **kwargs['model_learn_kwargs'],
    )
    model.save(**kwargs['model_save_kwargs'])
    predict(env, model, run, **kwargs['model_predict_kwargs'])


def train_sac(run, kwargs):
    env = gym.make(kwargs['env_name'])
    model = SAC(env=env, tensorboard_log=f"runs/{run.id}", **kwargs['model_load_kwargs'])

    model.learn(
        callback=prepare_callbacks(kwargs, run),
        **kwargs['model_learn_kwargs'],
    )
    model.save(**kwargs['model_save_kwargs'])
    predict(env, model, run, **kwargs['model_predict_kwargs'])


def prepare_callbacks(kwargs, run):
    callbacks = [WandbCallback(**kwargs['model_learn_wandb_callback_kwargs'], model_save_path=f"models/wandb/{run.id}")]
    if 'callback' in kwargs['model_learn_kwargs']:
        callbacks.append(kwargs['model_learn_kwargs'].pop('callback'))
    a = stable_baselines3.common.callbacks.CallbackList(callbacks)
    return a


def train_td3(run, kwargs):
    env = gym.make(kwargs['env_name'])
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    model = TD3(env=env, action_noise=action_noise, tensorboard_log=f"runs/{run.id}", **kwargs['model_load_kwargs'])
    model.learn(
        callback=prepare_callbacks(kwargs, run),  # Be careful, order matters, prepare_callbacks removes the callback from kwargs
        **kwargs['model_learn_kwargs'],
    )
    model.save(**kwargs['model_save_kwargs'])
    env = model.get_env()
    predict(env, model, run, **kwargs['model_predict_kwargs'])
