import functools
from copy import deepcopy
from time import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_ind

from src.load_utils import load_grn, topo_sort_graph_layers, get_basal_production_rate
from src.zoo_functions import is_debugger_active, plot_three_genes


class Sim:

    def __init__(self, num_genes: int, num_cells_types: int, num_cells_to_simulate: int, noise_amplitude: float = 1.,
                 **kwargs):
        self.interactions_filename = 'data/Interaction_cID_4.txt'
        self.regulators_filename = 'data/Regs_cID_4.txt'
        self.num_genes = num_genes
        self.num_cell_types = num_cells_types
        self.num_cells_to_simulate = num_cells_to_simulate
        self.adjacency = np.zeros(shape=(self.num_genes, self.num_genes))
        self.regulators_dict = dict()
        self.repressive_dict = dict()
        self.decay_lambda = 0.8
        self.mean_expression = -1 * jnp.ones((num_genes, num_cells_types))
        self.sampling_state = 50
        self.is_regulator = None
        self.simulation_time_steps = self.sampling_state * self.num_cells_to_simulate
        print("sampling time steps: ", self.simulation_time_steps)
        self.x = jnp.zeros(shape=(self.simulation_time_steps, num_genes, num_cells_types))
        self.half_response = jnp.zeros(num_genes)
        self.hill_coefficient = 2
        self.p_value_for_convergence = 1e-3
        self.window_len = 100
        self.noise_parameters_genes = np.repeat(noise_amplitude, num_genes)

    def run(self):
        adjacency, graph = load_grn(self.interactions_filename, self.adjacency)
        self.adjacency = jnp.array(adjacency)
        regulators, genes = np.where(self.adjacency)

        # TODO: there is a loop too much below
        self.regulators_dict = dict(zip(genes, [np.array(np.where(self.adjacency[:, g])[0]) for g in genes]))
        self.repressive_dict = dict(
            zip(genes, [self.adjacency[:, g, None].repeat(self.num_cell_types, axis=1) < 0 for g in genes]))
        # is_repressive = np.expand_dims(self.adjacency[regulators][:, genes] < 0, -1).repeat(self.num_cell_types, axis=-1)

        layers = topo_sort_graph_layers(graph)
        # self.is_regulator = self.adjacency != 0

        basal_production_rate = get_basal_production_rate(self.regulators_filename, self.num_genes, self.num_cell_types)
        self.simulate_expression_layer_wise(layers, basal_production_rate)

    def simulate_expression_layer_wise(self, layers, basal_production_rate):
        f, ax = plt.subplots(1, 3, figsize=(10, 10))

        key = jax.random.PRNGKey(0)
        key, subkey = jax.random.split(key)
        subkeys = jax.random.split(subkey, self.num_cell_types)

        layers_copy = [np.array(l) for l in deepcopy(layers)]  # TODO: remove deepcopy
        layer = np.array(layers_copy[0])
        self.x = self.x.at[0, layer].set(basal_production_rate[np.array(layer)] / self.decay_lambda)

        curr_genes_expression = self.x[0]
        curr_genes_expression = {k: curr_genes_expression[k] for k in
                                 range(len(curr_genes_expression))}  # TODO remove this
        d_genes = jax.vmap(self.simulate_master_layer, in_axes=(1, 0, None, 0))(
            basal_production_rate, curr_genes_expression, layer, subkeys
        )
        self.x = self.x.at[1:, layer].set(d_genes.T)
        self.mean_expression = self.mean_expression.at[layer].set(np.mean(self.x[:, layer], axis=0))

        ax[0].bar([str(g) for g in layers[0]], self.x[0, layer, 0])

        for num_layer, layer in enumerate(layers_copy[1:], start=1):
            key, subkey = jax.random.split(key)
            subkeys = jax.random.split(subkey, self.num_cell_types)

            half_responses = self.calculate_half_response(tuple(layer), self.mean_expression)
            self.half_response = self.half_response.at[layer].set(half_responses)
            self.x = self.init_concentration(tuple(layer), self.half_response, self.mean_expression, self.x)
            
            ax[num_layer].bar([str(g) for g in layer], self.x[0, layer, 0])

            curr_genes_expression = self.x[0]
            curr_genes_expression = {k: curr_genes_expression[k] for k in
                                     range(len(curr_genes_expression))}  # TODO remove this
            production_rates = jnp.array([self.calculate_production_rate(gene, self.half_response,
                                                                         self.mean_expression) for gene in
                                          layer])
            trajectories = jax.vmap(self.simulate_targets, in_axes=(1, 0, None, None, 0))(
                production_rates, curr_genes_expression, self.noise_parameters_genes, layer, subkeys
            )
            self.x = self.x.at[1:, layer].set(trajectories.T)
            self.mean_expression = self.mean_expression.at[layer].set(np.mean(self.x[:, layer], axis=0))

        plt.show()

    def simulate_master_layer(self, basal_production_rate, curr_genes_expression, layer, key):
        subkeys = jax.random.split(key, len(layer))
        dx = jax.vmap(self.euler_maruyama_master, in_axes=(0, 0, 0, 0))(
            jnp.array([curr_genes_expression[gene] for gene in layer]),
            basal_production_rate.take(jnp.array(layer)),
            self.noise_parameters_genes.take(jnp.array(layer)),
            subkeys
        )
        return dx

    def simulate_targets(self, production_rates, curr_genes_expression, q, layer, key):
        subkeys = jax.random.split(key, len(layer))
        dx = jax.vmap(self.euler_maruyama_targets, in_axes=(0, 0, 0, None, 0))(
            jnp.array([curr_genes_expression[gene] for gene in layer]),
            q.take(jnp.array(layer)),
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
            mean_expression_per_cells_regulators_wise = mean_expression[regulators]
            half_response = np.mean(mean_expression_per_cells_regulators_wise)
            half_responses.append(half_response)
        return half_responses

    @functools.partial(jax.jit, static_argnums=(0, 1))
    def init_concentration(self, layer: list, half_response: np.ndarray, mean_expression: np.ndarray, x):
        """ Init concentration genes; Note: calculate_half_response should be run before this method """
        rates = [self.calculate_production_rate(gene, half_response, mean_expression) for gene in layer]
        return x.at[0, layer].set(rates)

    @functools.partial(jax.jit, static_argnums=(0, 1))
    def calculate_production_rate(self, gene, half_response, mean_expression):
        # regulators = np.where(self.adjacency[:, gene] != 0)
        # regulators = self.adjacency[:, gene] != 0
        regulators = self.regulators_dict[gene]

        mean_expression = mean_expression[regulators]

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
        rate2 = jnp.where(is_repressive, rate, 1 - rate)
        # rate = rate.at[is_repressive].set(1 - rate[is_repressive])
        return rate2

    @functools.partial(jax.jit, static_argnums=(0,))
    def euler_maruyama_master(self, curr_genes_expression, basal_production_rate, q, key: jax.random.PRNGKey):

        production_rates = basal_production_rate
        decays = jnp.multiply(self.decay_lambda, curr_genes_expression)
        key, subkey = jax.random.split(key)
        dw_p = jax.random.normal(subkey, shape=(self.simulation_time_steps - 1,))
        key, subkey = jax.random.split(key)
        dw_d = jax.random.normal(subkey, shape=(self.simulation_time_steps - 1,))

        def concentration_forward(curr_concentration, state):
            dw_production, dw_decay = state
            amplitude_p = q * jnp.power(production_rates, 0.5)
            amplitude_d = q * jnp.power(decays, 0.5)
            noise = jnp.multiply(amplitude_p, dw_production) + jnp.multiply(amplitude_d, dw_decay)
            next_gene_conc = curr_concentration + (0.01 * jnp.subtract(production_rates, decays)) + jnp.power(
                0.01, 0.5) # * noise  # shape=( # #genes,#types)
            next_gene_conc = jnp.clip(next_gene_conc, a_min=0)
            return next_gene_conc, next_gene_conc

        all_states = dw_p, dw_d
        gene_trajectory_concentration = jax.lax.scan(concentration_forward, curr_genes_expression, all_states)[1]
        return gene_trajectory_concentration

    @functools.partial(jax.jit, static_argnums=(0,))
    def euler_maruyama_targets(self, curr_genes_expression, q, production_rates, layer, key):
        key, subkey = jax.random.split(key)
        dw_p = jax.random.normal(key, shape=(self.simulation_time_steps - 1,))
        key, subkey = jax.random.split(key)
        dw_d = jax.random.normal(key, shape=(self.simulation_time_steps - 1,))

        def step(carry, state):
            curr_x = carry
            dw_d, dw_p = state
            decays = jnp.multiply(self.decay_lambda, curr_genes_expression)
            amplitude_p = q * jnp.power(production_rates, 0.5)
            amplitude_d = q * jnp.power(decays, 0.5)
            noise = jnp.multiply(amplitude_p, dw_p) + jnp.multiply(amplitude_d, dw_d)
            next_x = curr_x + 0.01 * jnp.subtract(production_rates, decays) + jnp.power(0.01,
                                                                                        0.5)  # * noise  # shape=(#genes,#types)
            next_x = jnp.clip(next_x, a_min=0)
            return next_x, next_x

        all_state = dw_d, dw_p
        gene_trajectory = jax.lax.scan(step, curr_genes_expression, all_state)[1]
        return gene_trajectory

    def check_for_convergence(self, gene_concentration, concentration_criteria='np_all_close'):
        converged = False

        if concentration_criteria == 't_test':
            sample1 = gene_concentration[-2 * self.window_len:-1 * self.window_len]
            sample2 = gene_concentration[-1 * self.window_len:]
            _, p = ttest_ind(sample1, sample2)
            if p >= self.p_value_for_convergence:
                converged = True

        elif concentration_criteria == 'mean':
            abs_mean_gene = np.abs(np.mean(gene_concentration[-self.window_len:]))
            if abs_mean_gene <= self.p_value_for_convergence:
                converged = True

        elif concentration_criteria == 'np_all_close':
            converged = np.allclose(gene_concentration[-2 * self.window_len:-1 * self.window_len],
                                    gene_concentration[-self.window_len:],
                                    atol=self.p_value_for_convergence)

        return converged

    def add_technical_noise(self):
        """ sequencing noise """
        raise NotImplementedError


if __name__ == '__main__':
    start = time()

    sim = Sim(num_genes=100, num_cells_types=9, num_cells_to_simulate=5)

    if is_debugger_active():
        with jax.disable_jit():
            sim.run()
    else:
        sim.run()

    expr_clean = sim.x
    print(expr_clean.shape)
    print(f"time: {time() - start}")

    plot_three_genes(expr_clean.T[0, 44], expr_clean.T[0, 1], expr_clean.T[0, 99])
