import gym
import jax.numpy as jnp
import numpy as np

import jax_simulator
import src.models.expert.classfier_cell_state
import src.techinical_noise
import src.zoo_functions


class GRNControlSimpleEnv(gym.Env):
    target_gene_type = 0
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.noise_amplitude = 0.8
        self.num_cells_to_sim = 10
        self.MAX_ACTIONS_VALUE = 10

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
        action_size = len(self.sim.layers[0])
        self.action_space = gym.spaces.Box(
            low=np.zeros(action_size) + 1e-6,
            high=np.ones(action_size) * self.MAX_ACTIONS_VALUE
        )

        self.observation_space = gym.spaces.Box(
            low=np.zeros((self.sim.num_genes, self.sim.num_cell_types)),
            high=np.ones((self.sim.num_genes, self.sim.num_cell_types)) * np.inf,
        )
        self.initial_state = None

    def step(self, action):
        rollout_states = self.sim.run_one_rollout(action)
        x = self.dict_to_array(rollout_states)
        x_T = x[-1, :, :]

        desired_concentration = self.initial_state[:, self.target_gene_type]
        dist = jnp.linalg.norm(np.tile(desired_concentration.reshape(1, -1, 1), (1, 1, x_T.shape[1])) - x_T)

        reward = -dist
        done = True

        # TODO: return bad reward if there is no convergence
        extra_info = {}
        return x_T, reward, done, extra_info

    def dict_to_array(self, x):
        expr_clean = jnp.stack(tuple([x[gene] for gene in range(self.sim.num_genes)])).swapaxes(0, 1)
        return expr_clean

    def reset(self):
        if self.initial_state is None:
            action = np.ones(self.action_space.shape)  # TODO: make this a random action instead of all ones
            trajectory = self.dict_to_array(self.sim.run_one_rollout(action))
            self.initial_state = trajectory[-1, :, :]  # last_time_step of the simulation trajectory
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
