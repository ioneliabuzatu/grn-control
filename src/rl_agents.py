"""RL agents taken from: https://stable-baselines3.readthedocs.io/en/master/index.html"""
import gym
import numpy as np
import gym_gene_control  # noqa

from stable_baselines3 import A2C
from stable_baselines3 import DDPG
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3 import TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise
from wandb.integration.sb3 import WandbCallback

__all__ = [
    "train_a2c",
    "train_ddpg",
    "train_ppo",
    "train_sac",
    "train_td3",
]


def train_a2c(run, how_many_time_steps_for_prediction: int = 10):
    env = make_vec_env("gene_control-v0", n_envs=4)

    model = A2C("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=25000)
    model.save("a2c_cartpole")

    del model  # remove to demonstrate saving and loading

    model = A2C.load("a2c_cartpole")

    obs = env.reset()
    for step in range(how_many_time_steps_for_prediction):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        # env.render()


def train_ddpg(run, how_many_time_steps_for_prediction: int=10):
    env = gym.make("gene_control-v0")

    # The noise objects for DDPG
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1)
    model.learn(total_timesteps=10000, log_interval=10)
    model.save("ddpg_pendulum")
    env = model.get_env()

    del model  # remove to demonstrate saving and loading

    model = DDPG.load("ddpg_pendulum")

    obs = env.reset()
    for step in range(how_many_time_steps_for_prediction):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)


def train_ppo(run, how_many_time_steps_for_prediction: int=10):
    env = make_vec_env("gene_control-v0", n_envs=2)  # Parallel environments

    model = PPO("MlpPolicy", env, verbose=2, device="cpu", n_steps=1, tensorboard_log=f"runs/{run.id}")
    model.learn(
        total_timesteps=5,
        callback=WandbCallback(
            gradient_save_freq=100,
            model_save_path=f"models/{run.id}",
            verbose=2,
        )
    )

    model.save("models/gene_control_simple_ppo")
    del model
    model = PPO.load("models/gene_control_simple_ppo")

    print("Predict")
    obs = env.reset()
    for step in range(how_many_time_steps_for_prediction):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        run.log({"prediction reward": rewards[0]})
        print("reward", rewards, "action", action.round(3))


def train_sac(run, how_many_time_steps_for_prediction: int=10):
    env = gym.make("gene_control-v0")

    model = SAC("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000, log_interval=4)
    model.save("sac_pendulum")

    del model  # remove to demonstrate saving and loading

    model = SAC.load("sac_pendulum")

    obs = env.reset()
    for step in range(how_many_time_steps_for_prediction):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        if done:
            obs = env.reset()


def train_td3(run, how_many_time_steps_for_prediction: int=10):
    env = gym.make("gene_control-v0")

    # The noise objects for TD3
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1)
    model.learn(total_timesteps=10000, log_interval=10)
    model.save("td3_pendulum")
    env = model.get_env()

    del model  # remove to demonstrate saving and loading

    model = TD3.load("td3_pendulum")

    obs = env.reset()
    for step in range(how_many_time_steps_for_prediction):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
