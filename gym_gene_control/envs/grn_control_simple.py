import gym
import jax.numpy as jnp
import numpy as np

import jax_simulator
import src.models.expert.classfier_cell_state
import src.techinical_noise
import src.zoo_functions


class GRNControlSimpleEnv(gym.Env):
    target_gene_type = 2
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.noise_amplitude = 0.1
        self.num_cells_to_sim = 3

        dataset_dict = src.zoo_functions.open_datasets_json(return_specific_dataset='Dummy')
        dataset = src.zoo_functions.dataset_namedtuple(*dataset_dict.values())
        self.sim = jax_simulator.Sim(
            num_genes=dataset.tot_genes,
            num_cells_types=dataset.tot_cell_types,
            simulation_num_steps=self.num_cells_to_sim,
            interactions_filepath=dataset.interactions,
            regulators_filepath=dataset.regulators,
            noise_amplitude=self.noise_amplitude,
        )
        self.sim.build()
        self.action_space = gym.spaces.Box(low=np.zeros(self.sim.num_genes), high=np.ones(self.sim.num_genes))
        self.observation_space = gym.spaces.Box(low=np.zeros(self.sim.num_genes), high=np.ones(self.sim.num_genes))
        self.initial_state = None

    def step(self, action):
        x_T = self.sim.run_one_rollout(action)
        x_T = self.to_array(x_T)

        desired_concentration = self.initial_state[:, :, self.target_gene_type].mean(axis=0)  # TODO: worry about outliers, zeros and negatives
        dist = jnp.linalg.norm(np.tile(desired_concentration.reshape(1, -1, 1), (1, 1, x_T.shape[2])) - x_T)

        reward = -dist
        done = False

        # TODO: return bad reward if there is no convergence
        return x_T, reward, done, {}

    def to_array(self, x):
        expr_clean = jnp.stack(tuple([x[gene] for gene in range(self.sim.num_genes)])).swapaxes(0, 1)
        return expr_clean

    def reset(self):
        if self.initial_state is None:
            action = jnp.zeros(self.sim.num_genes)
            self.initial_state = self.to_array(self.sim.run_one_rollout(action))
        return self.initial_state

    def render(self, mode='human'):
        raise NotImplemented

    def close(self):
        raise NotImplemented


if __name__ == '__main__':
    env = GRNControlSimpleEnv()
    x = env.reset()
    print(x)
    for _ in range(100):
        action = env.action_space.sample()
        x, _, _, _ = env.step(action)
        print(x)
