import gym
import jax
import jax.numpy as jnp
import numpy as np
from gym import spaces

import jax_simulator
import src.models.expert.classfier_cell_state
import src.techinical_noise
import src.zoo_functions


class GRNControlSimpleEnv(gym.Env):
    target_gene_type = 2
    metadata = {'render.modes': ['human']}

    def __init__(self):
        params = {'num_genes': 100, 'NUM_SIM_CELLS': 100}

        dataset_dict = src.zoo_functions.open_datasets_json(return_specific_dataset='Dummy')
        dataset = src.zoo_functions.dataset_namedtuple(*dataset_dict.values())
        self.sim = jax_simulator.Sim(
            num_genes=params["num_genes"],
            num_cells_types=params["NUM_SIM_CELLS"],
            simulation_num_steps=params['NUM_SIM_CELLS'],
            interactions_filepath=dataset.interactions,
            regulators_filepath=dataset.regulators,
            noise_amplitude=0.5
        )
        self.sim.build()
        self.action_space = spaces.Box(low=np.zeros(self.sim.num_genes), high=np.ones(self.sim.num_genes))
        self.observation_space = spaces.Box(low=np.zeros(self.sim.num_genes), high=np.ones(self.sim.num_genes))
        self.initial_state = None

    def step(self, action):
        final_gene_expression_dict = self.sim.run_one_rollout(action)
        final_gene_expression = self.to_numpy(final_gene_expression_dict)

        target_gene_expression = self.initial_state[:, :, self.target_gene_type]
        dist = np.linalg.norm(final_gene_expression - target_gene_expression[self.target_gene_type])

        reward = -dist
        done = False

        # TODO: return bad reward if there is no convergence
        return final_gene_expression, reward, done, {}

    def to_numpy(self, final_gene_expression):
        final_gene_expression = jnp.stack(tuple([final_gene_expression[gene] for gene in range(self.sim.num_genes)])).swapaxes(0, 1)
        final_gene_expression = jnp.concatenate(final_gene_expression, axis=1).T
        return final_gene_expression

    def reset(self):
        if self.initial_state is None:
            action = jnp.zeros(self.sim.num_genes)
            self.initial_state = self.to_numpy(self.sim.run_one_rollout(action))
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
