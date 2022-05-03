import stable_baselines3.common.utils
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback

import gym_gene_control  # noqa

# Parallel environments
env = make_vec_env("gene_control-simple-v0", n_envs=2)

model = PPO("MlpPolicy", env, verbose=2, device="cpu", n_steps=1)


class SensitivityCallback(BaseCallback):
    def on_rollout_end(self) -> None:
        obs = self.model.rollout_buffer.observations
        obs_tensor = stable_baselines3.common.utils.obs_as_tensor(obs, self.model.device).reshape(-1, *obs.shape[-2:])
        obs_tensor = obs_tensor.requires_grad_(True)
        actions, values, log_probs = self.model.policy(obs_tensor)
        actions.abs().mean().backward()
        for gene_idx, gene in enumerate(obs_tensor.grad.mean(0)):
            print(f"Gene:{gene_idx}", gene[0])

    def _on_step(self) -> bool:
        pass


model.learn(total_timesteps=50, callback=SensitivityCallback())
model.save("gene_control_simple_ppo")

del model  # remove to demonstrate saving and loading

model = PPO.load("gene_control_simple_ppo")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    print("reward", rewards, "action", action.round(3))
