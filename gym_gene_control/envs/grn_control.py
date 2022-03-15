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
    metadata = {'render.modes': ['human']}

    def __init__(self):
        ds4_ground_truth_initial_dist = np.load("data/ds4_10k_each_type.npy")
        params = {'num_genes': 100, 'NUM_SIM_CELLS': 100}

        # manual calculation of some distance matrix
        mean_samples_wise_t0 = ds4_ground_truth_initial_dist[:10000].mean(axis=0)
        mean_samples_wise_t1 = ds4_ground_truth_initial_dist[10000:20000].mean(axis=0)
        mean_samples_wise_t2 = ds4_ground_truth_initial_dist[20000:].mean(axis=0)
        d01 = mean_samples_wise_t0 - mean_samples_wise_t1
        d02 = mean_samples_wise_t0 - mean_samples_wise_t2
        d12 = mean_samples_wise_t1 - mean_samples_wise_t2
        print(f"distances: \n 0 <-> 1 {abs(d01.sum()):3f} \n 0 <-> 2 {abs(d02.sum()):3f} \n 1 <-> 2 {abs(d12.sum()):3f}")

        dataset_dict = src.zoo_functions.open_datasets_json(return_specific_dataset='DS4')
        dataset = src.zoo_functions.dataset_namedtuple(*dataset_dict.values())

        expert_checkpoint_filepath = "src/models/expert/checkpoints/classifier_ds4.pth"
        classifier = src.models.expert.classfier_cell_state.CellStateClassifier(num_genes=dataset.tot_genes, num_cell_types=dataset.tot_cell_types).to("cpu")
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
        self.action_space = spaces.Box(low=np.zeros(self.sim.num_genes), high=np.ones(self.sim.num_genes))
        self.observation_space = spaces.Box(low=np.zeros(self.sim.num_genes), high=np.ones(self.sim.num_genes))

    def step(self, action):
        @jax.jit
        def loss_fn(actions):
            gene_expression = self.sim.run_one_rollout(actions)
            gene_expression = jnp.stack(tuple([gene_expression[gene] for gene in range(
                self.sim.num_genes)])).swapaxes(0, 1)

            if self.add_technical_noise_function is not None:
                gene_expression = self.add_technical_noise_function.get_noisy_technical_concentration(
                    gene_expression.T).T
            else:
                gene_expression = jnp.concatenate(gene_expression, axis=1).T

            expert_logprobs = self.expert(gene_expression)
            desired_celltype = 2
            cross_entropy = jnp.mean(expert_logprobs[:, desired_celltype])
            return -cross_entropy, gene_expression

    def reset(self):
        return self.step(np.zeros(self.sim.num_genes))

    def render(self, mode='human'):
        raise NotImplemented

    def close(self):
        raise NotImplemented
