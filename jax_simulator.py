import functools
from copy import deepcopy
from time import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_ind

from src.load_utils import load_grn_jax, topo_sort_graph_layers, get_basal_production_rate
from src.zoo_functions import is_debugger_active, plot_three_genes


class Sim:

    def __init__(self, num_genes: int, num_cells_types: int,
                 interactions_filepath: str, regulators_filepath: str,
                 simulation_num_steps: int, num_samples_from_trajectory: int = None,
                 noise_amplitude: float = 1.,
                 add_technical_noise: bool = True,
                 **kwargs):

        self.num_genes = num_genes
        self.num_cell_types = num_cells_types
        self.interactions_filename = interactions_filepath
        self.regulators_filename = regulators_filepath
        self.simulation_num_steps = simulation_num_steps
        self.num_samples_from_trajectory = num_samples_from_trajectory  # use in future when trajectory is long 1M steps
        self.noise_parameters_genes = np.repeat(noise_amplitude, num_genes)
        self.add_technical_noise = add_technical_noise

        self.adjacency = jnp.zeros(shape=(self.num_genes, self.num_genes))
        self.regulators_dict = dict()
        self.repressive_dict = dict()
        self.decay_lambda = 0.8
        self.mean_expression = -1 * jnp.ones((num_genes, num_cells_types))
        self.is_regulator = None
        print("simulation num step per trajectories: ", self.simulation_num_steps)
        self.half_response = jnp.zeros(num_genes)
        self.hill_coefficient = 2
        self.dt = 0.01

        self.layers = None
        self.basal_production_rates = None

    def build(self):
        adjacency, graph = load_grn_jax(self.interactions_filename, self.adjacency)
        self.adjacency = jnp.array(adjacency)
        regulators, genes = np.where(self.adjacency)

        # TODO: there is a loop too much below
        self.regulators_dict = dict(zip(genes, [np.array(np.where(self.adjacency[:, g])[0]) for g in genes]))
        self.repressive_dict = dict(
            zip(genes, [self.adjacency[:, g, None].repeat(self.num_cell_types, axis=1) < 0 for g in genes]))

        self.layers = topo_sort_graph_layers(graph)

    def run_one_rollout(self, actions=None):
        """return the gene expression of shape [samples, genes]"""
        if actions is not None:
            basal_production_rates = jnp.zeros((self.num_genes, self.num_cell_types))
            for action_gene, master_id in zip(actions, self.layers[0]):
                basal_production_rates = basal_production_rates.at[master_id].set(action_gene)
        else:
            basal_production_rates = jnp.array(
                get_basal_production_rate(self.regulators_filename, self.num_genes, self.num_cell_types))

        x = self.simulate_expression_layer_wise(basal_production_rates)
        return x

    def simulate_expression_layer_wise(self, basal_production_rates):
        key = jax.random.PRNGKey(0)
        key, subkey = jax.random.split(key)
        subkeys = jax.random.split(subkey, self.num_cell_types)

        layers_copy = [np.array(l) for l in deepcopy(self.layers)]  # TODO: remove deepcopy
        master_layer = np.array(layers_copy[0])
        x_0 = dict(zip(master_layer, (basal_production_rates[master_layer] / self.decay_lambda)))
        curr_genes_expression = x_0

        d_genes = jax.vmap(self.simulate_master_layer, in_axes=(1, 0, None, 0))(
            basal_production_rates, curr_genes_expression, master_layer, subkeys
        )
        x = {master_layer[i]: jnp.vstack((x_0[master_layer[i]].reshape(1, -1), d_genes.T[:, i, :])) for i in range(len(
            master_layer))}
        mean_expression = {idx: jnp.mean(x[idx], axis=0) for idx in master_layer}
        # self.mean_expression = self.mean_expression.at[master_layer].set([jnp.mean(x[master_gene], axis=0) for
        #                                                                   master_gene in
        #                                                                   master_layer])

        for num_layer, layer in enumerate(layers_copy[1:], start=1):
            key, subkey = jax.random.split(key)
            subkeys = jax.random.split(subkey, self.num_cell_types)

            half_response = dict(zip(layer, self.calculate_half_response(tuple(layer), mean_expression)))
            # self.half_response = self.half_response.at[layer].set([half_response[layer_gene] for layer_gene in layer])

            x_0.update(dict(zip(layer, self.init_concentration(tuple(layer), half_response, mean_expression, x))))

            curr_genes_expression = x_0
            production_rates = jnp.array(
                [self.calculate_production_rate(gene, half_response, mean_expression) for gene in layer])
            trajectories = jax.vmap(self.simulate_targets, in_axes=(1, 0, None, 0))(
                production_rates, curr_genes_expression, layer, subkeys
            )
            x.update({layer[i]: jnp.vstack((x_0[layer[i]].reshape(1, -1), trajectories.T[:, i, :])) for i in
                      range(len(layer))})
            mean_expression.update({idx: jnp.mean(x[idx], axis=0) for idx in layer})
            # self.mean_expression = self.mean_expression.at[layer].set([jnp.mean(self.x[layer_gene], axis=0) for
            # layer_gene in layer])

        return x  # TODO: add variable that need to be carried over (half_response, mean_expression, etc.)

    def simulate_master_layer(self, basal_production_rate, curr_genes_expression, layer, key):
        subkeys = jax.random.split(key, len(layer))
        dx = jax.vmap(self.euler_maruyama_master, in_axes=(0, 0, 0, 0))(
            jnp.array([curr_genes_expression[gene] for gene in layer]),
            basal_production_rate.take(jnp.array(layer)),
            self.noise_parameters_genes.take(jnp.array(layer)),
            subkeys
        )
        return dx

    def simulate_targets(self, production_rates, curr_genes_expression, layer, key):
        subkeys = jax.random.split(key, len(layer))
        dx = jax.vmap(self.euler_maruyama_targets, in_axes=(0, 0, 0, None, 0))(
            jnp.array([curr_genes_expression[gene] for gene in layer]),
            self.noise_parameters_genes.take(jnp.array(layer)),
            production_rates,
            tuple(layer),
            subkeys
        )
        return dx

    @functools.partial(jax.jit, static_argnums=(0, 1))
    def calculate_half_response(self, layer, mean_expression):
        half_responses = []
        for gene in layer:
            regulators = self.regulators_dict[gene]
            mean_expression_per_cells_regulators_wise = jnp.vstack(tuple([mean_expression[reg] for reg in regulators]))
            half_response = jnp.mean(mean_expression_per_cells_regulators_wise)
            half_responses.append(half_response)
        return half_responses

    @functools.partial(jax.jit, static_argnums=(0, 1))
    def init_concentration(self, layer: list, half_response, mean_expression, x):
        """ Init concentration genes; Note: calculate_half_response should be run before this method """
        rates = jnp.array([self.calculate_production_rate(gene, half_response, mean_expression) for gene in layer])
        rates = rates / self.decay_lambda
        return rates

    @functools.partial(jax.jit, static_argnums=(0, 1))
    def calculate_production_rate(self, gene, half_response, mean_expression):
        regulators = self.regulators_dict[gene]
        mean_expression = jnp.vstack(tuple([mean_expression[reg] for reg in regulators]))
        absolute_k = jnp.abs(self.adjacency[regulators][:, gene])
        half_response = half_response[gene]
        is_repressive = self.repressive_dict[gene]
        is_repressive_ = is_repressive[regulators]
        hill_function = self.hill_function(mean_expression, half_response, is_repressive_)
        rate = jnp.einsum("r,rt->t", absolute_k, hill_function)
        return rate

    def hill_function(self, regulators_concentration, half_response, is_repressive):
        rate = (
                jnp.power(regulators_concentration, self.hill_coefficient) /
                (jnp.power(half_response, self.hill_coefficient) + jnp.power(regulators_concentration,
                                                                             self.hill_coefficient))
        )
        rate2 = jnp.where(is_repressive, 1 - rate, rate)
        return rate2

    @functools.partial(jax.jit, static_argnums=(0,))
    def euler_maruyama_master(self, curr_genes_expression, basal_production_rate, q, key: jax.random.PRNGKey):
        production_rates = basal_production_rate
        key, subkey = jax.random.split(key)
        dw_p = jax.random.normal(subkey, shape=(self.simulation_num_steps - 1,))
        key, subkey = jax.random.split(key)
        dw_d = jax.random.normal(subkey, shape=(self.simulation_num_steps - 1,))

        def concentration_forward(curr_concentration, state):
            dw_production, dw_decay = state
            decay = jnp.multiply(0.8, curr_concentration)
            amplitude_p = q * jnp.power(production_rates, 0.5)
            amplitude_d = q * jnp.power(decay, 0.5)
            noise = jnp.multiply(amplitude_p, dw_production) + jnp.multiply(amplitude_d, dw_decay)
            next_gene_conc = curr_concentration + (self.dt * jnp.subtract(production_rates, decay)) + jnp.power(self.dt,
                                                                                                                0.5) * noise  # shape=( # #genes,#types)
            next_gene_conc = jnp.clip(next_gene_conc, a_min=0)
            return next_gene_conc, next_gene_conc

        all_states = dw_p, dw_d
        gene_trajectory_concentration = jax.lax.scan(concentration_forward, curr_genes_expression, all_states)[1]
        return gene_trajectory_concentration

    @functools.partial(jax.jit, static_argnums=(0,))
    def euler_maruyama_targets(self, curr_genes_expression, q, production_rates, layer, key):
        key, subkey = jax.random.split(key)
        dw_p = jax.random.normal(key, shape=(self.simulation_num_steps - 1,))
        key, subkey = jax.random.split(key)
        dw_d = jax.random.normal(key, shape=(self.simulation_num_steps - 1,))

        def step(carry, state):
            curr_x = carry
            dw_d, dw_p = state
            decay = jnp.multiply(0.8, curr_x)
            amplitude_p = q * jnp.power(production_rates, 0.5)
            amplitude_d = q * jnp.power(decay, 0.5)
            noise = jnp.multiply(amplitude_p, dw_p) + jnp.multiply(amplitude_d, dw_d)
            next_x = curr_x + (self.dt * jnp.subtract(production_rates, decay)) + jnp.power(self.dt,
                                                                                            0.5) * noise  # # shape=(#genes,#types)
            next_x = jnp.clip(next_x, a_min=0)
            return next_x, next_x

        all_state = dw_d, dw_p
        gene_trajectory = jax.lax.scan(step, curr_genes_expression, all_state)[1]
        return gene_trajectory

    def technical_noise(self):
        """ sequencing noise """
        raise NotImplementedError


if __name__ == '__main__':
    start = time()

    sim = Sim(num_genes=100, num_cells_types=9,
              interactions_filepath="data/Interaction_cID_4.txt",
              regulators_filepath="data/Regs_cID_4.txt",
              simulation_num_steps=10000,
              )
    sim.build()

    if is_debugger_active():
        with jax.disable_jit():
            x = sim.run_one_rollout()
    else:
        x = sim.run_one_rollout()

    expr_clean = jnp.stack(tuple([x[gene] for gene in range(sim.num_genes)])).swapaxes(0, 1)

    print(expr_clean.shape)
    print(f"time: {time() - start}")

    plot_three_genes(expr_clean.T[0, 44], expr_clean.T[0, 1], expr_clean.T[0, 99], hlines=None)
