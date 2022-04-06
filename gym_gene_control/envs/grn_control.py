import gym
import jax
import torch
import numpy as np
import jax.numpy as jnp
from gym import error, spaces, utils
from gym.utils import seeding

import src.zoo_functions
import src.models.expert.classfier_cell_state
import jax_simulator
import src.techinical_noise


class GRNControlEnv(gym.Env):
    target_gene_type = 0
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.noise_amplitude = 0.8
        self.num_cells_to_sim = 10
        self.MAX_ACTIONS_VALUE = 10

        params = {'num_genes': 100, 'NUM_SIM_CELLS': 100}
        dataset_dict = src.zoo_functions.open_datasets_json(return_specific_key='DS4')
        dataset = src.zoo_functions.dataset_namedtuple(*dataset_dict.values())

        expert_checkpoint_filepath = "src/models/expert/checkpoints/classifier_ds4.pth"
        classifier = src.models.expert.classfier_cell_state.CellStateClassifier(num_genes=dataset.tot_genes,
                                                                                num_cell_types=dataset.tot_cell_types).to(
            "cpu")
        loaded_checkpoint = torch.load(expert_checkpoint_filepath, map_location=lambda storage, loc: storage)
        classifier.load_state_dict(loaded_checkpoint)
        classifier.eval()
        self.expert = src.zoo_functions.torch_to_jax(classifier)

        self.sim = jax_simulator.Sim(
            num_genes=dataset.tot_genes, num_cells_types=dataset.tot_cell_types,
            simulation_num_steps=params['NUM_SIM_CELLS'],
            interactions_filepath=dataset.interactions, regulators_filepath=dataset.regulators, noise_amplitude=0.5
        )
        self.sim.build()
        self.add_technical_noise_function = src.techinical_noise.AddTechnicalNoiseJax(
            dataset.tot_genes, dataset.tot_cell_types, params['NUM_SIM_CELLS'],
            dataset.params_outliers_genes_noise, dataset.params_library_size_noise, dataset.params_dropout_noise
        )

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
        # @jax.jit
        def loss_fn(actions):
            gene_expression = self.sim.run_one_rollout(actions)
            gene_expression = self.dict_to_array(gene_expression)
            x_T = gene_expression[-1, :, :]

            if self.add_technical_noise_function is not None:
                gene_expression = self.add_technical_noise_function.get_noisy_technical_concentration(
                    gene_expression.T).T
            else:
                gene_expression = jnp.concatenate(gene_expression, axis=1).T

            expert_log_probs = self.expert(gene_expression)
            cross_entropy = jnp.mean(expert_log_probs[:, self.target_gene_type])
            return -cross_entropy, x_T

        reward, x_T = loss_fn(action)
        done = True
        extra_info = {}

        return x_T, reward, done, extra_info

    def reset(self):
        if self.initial_state is None:
            action = np.ones(self.action_space.shape)  # TODO: make this a random action instead of all ones
            trajectory = self.dict_to_array(self.sim.run_one_rollout(action))
            self.initial_state = trajectory[-1, :, :]  # last_time_step of the simulation trajectory
        return self.initial_state

    def dict_to_array(self, x):
        expr_clean = jnp.stack(tuple([x[gene] for gene in range(self.sim.num_genes)])).swapaxes(0, 1)
        return expr_clean

    def render(self, mode='human'):
        raise NotImplemented

    def close(self):
        raise NotImplemented


if __name__ == '__main__':
    env = GRNControlEnv()
    x = env.reset()
    print(x)
    for _ in range(100):
        action = env.action_space.sample()
        x, reward, done, extra_info = env.step(action)
        print("reward...    ", reward)
