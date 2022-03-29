from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from wandb.integration.sb3 import WandbCallback
import experiment_buddy


import gym_gene_control  # noqa


params = {'todo': 'todo'}
experiment_buddy.register_defaults(params)
buddy = experiment_buddy.deploy(host="", disabled=False)
run = buddy.run

# Parallel environments
env = make_vec_env("gene_control-simple-v0", n_envs=2)

model = PPO("MlpPolicy", env, verbose=2, device="cpu", n_steps=1)
model.learn(total_timesteps=50, callback=WandbCallback())
model.save("gene_control_simple_ppo")

del model  # remove to demonstrate saving and loading

model = PPO.load("gene_control_simple_ppo")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    print("reward", rewards, "action", action.round(3))
