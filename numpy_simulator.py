import csv
from copy import deepcopy
from itertools import repeat

import jax.numpy as jnp
import networkx as nx
import numpy as np
from scipy.stats import ttest_ind

from load_utils import load_grn, topo_sort_graph_layers, get_basal_production_rate


class Sim:

    def __init__(self, num_genes, num_cells_types, num_cells_to_simulate, **kwargs):
        self.interactions_filename = 'SERGIO/data_sets/De-noised_100G_9T_300cPerT_4_DS1/Interaction_cID_4.txt'
        self.regulators_filename = 'SERGIO/data_sets/De-noised_100G_9T_300cPerT_4_DS1/Regs_cID_4.txt'
        self.num_genes = num_genes
        self.num_cell_types = num_cells_types
        self.num_cells_to_simulate = num_cells_to_simulate
        self.adjacency = np.zeros(shape=(self.num_genes, self.num_genes))
        self.decay_lambda = 0.8
        self.mean_expression = -1 * np.ones((num_genes, num_cells_types))
        self.sampling_state = 50
        self.simulation_time_steps = self.sampling_state * self.num_cells_to_simulate
        print("sampling time steps: ", self.simulation_time_steps)
        self.x = np.zeros(shape=(self.simulation_time_steps, num_genes, num_cells_types))
        self.half_response = np.zeros(num_genes)
        self.hill_coefficient = 2

        self.p_value_for_convergence = 1e-3
        self.window_len = 100
        self.noise_parameters_genes = np.ones(num_genes)

    def run(self):
        self.adjacency, graph = load_grn(self.interactions_filename, self.adjacency)
        layers = topo_sort_graph_layers(graph)
        basal_production_rate = get_basal_production_rate(self.regulators_filename, self.num_genes, self.num_cell_types)
        self.simulate_expression_layer_wise(layers, basal_production_rate)

    def simulate_expression_layer_wise(self, layers, basal_production_rate):
        layers_copy = deepcopy(layers)
        for num_layer, layer in enumerate(layers_copy):
            if num_layer != 0 : # not the master layer
                self.calculate_half_response(layer)
            self.init_concentration(layer, basal_production_rate)

            step = 1
            while step < self.simulation_time_steps:
                for gene in layer:
                    curr_gene_expression = self.x[step - 1, gene]
                    production_rate = self.calculate_production_rate(gene, basal_production_rate)
                    decay = np.multiply(self.decay_lambda, curr_gene_expression)

                    dw_p = np.random.normal(size=len(curr_gene_expression))
                    dw_d = np.random.normal(size=len(curr_gene_expression))
                    amplitude_p = np.multiply(self.noise_parameters_genes[gene], np.power(production_rate, 0.5))
                    amplitude_d = np.multiply(self.noise_parameters_genes[gene], np.power(decay, 0.5))
                    noise = np.multiply(amplitude_p, dw_p) + np.multiply(amplitude_d, dw_d)

                    dx = 0.01 * (production_rate - decay) + np.power(0.01, 0.5) * noise

                    updated_concentration_gene = curr_gene_expression + dx
                    self.x[step, gene] = updated_concentration_gene.clip(0) # clipping is important!

                step += 1

            self.mean_expression[layer] = np.mean(self.x[:, layer], axis=0)

            # check_convergence = self.check_for_convergence(self.x[:, layer])  TODO: convergence layer-wise?
            # print(f'step: {step} | layer: {num_layer} | Did it converge? : {check_convergence}')

    def calculate_half_response(self, layer):
        for gene in layer:
            regulators = np.where(self.adjacency[:, gene] != 0)
            if regulators:  # TODO this is wrong, how to check for empty np.where?
                mean_expression_per_cells_regulators_wise = self.mean_expression[regulators]
                half_response = np.mean(mean_expression_per_cells_regulators_wise)
                self.half_response[gene] = half_response

    def init_concentration(self, layer: list, basal_production_rate):  # TODO missing basal_production_rate
        """
        Initializes the concentration of all genes in the input level
        Note: calculate_half_response_ should be run before this method
        """
        x0 = np.zeros(shape=(self.num_genes, self.num_cell_types))

        for gene in layer:
            rate = self.calculate_production_rate(gene, basal_production_rate)
            x0[gene] = rate / self.decay_lambda

        self.x[0, layer] = x0[layer]

    def hill_function(self, regulators_concentration, half_response, is_repressive):
        rate = np.power(regulators_concentration, self.hill_coefficient) / (
                np.power(half_response, self.hill_coefficient) + np.power(regulators_concentration,
                                                                          self.hill_coefficient))

        rate[is_repressive] = 1 - rate[is_repressive]
        return rate

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

    def calculate_production_rate(self, gene, basal_production_rate):
        gene_basal_production = basal_production_rate[gene]
        if (gene_basal_production != 0).all():
            return gene_basal_production

        regulators = np.where(self.adjacency[:, gene] != 0)
        mean_expression = self.mean_expression[regulators]
        absolute_k = np.abs(self.adjacency[regulators][:, gene])
        is_repressive = np.expand_dims(self.adjacency[regulators][:, gene] < 0, -1).repeat(self.num_cell_types,
                                                                                           axis=-1)
        half_response = self.half_response[gene]
        hill_function = self.hill_function(mean_expression, half_response, is_repressive)
        rate = np.einsum("r,rt->t", absolute_k, hill_function)

        return rate

    def get_selected_concentrations_time_steps(self):
        return np.random.randint(low=-self.simulation_time_steps, high=0, size=self.num_cells_to_simulate)

    def euler_maruyama(self, drift, diffusion, t_span, y0, num_points, key):  # TODO add it and fix it
        delta_t = (t_span[1] - t_span[0]) / num_points
        delta_W_α = jnp.sqrt(delta_t) * jax.random.normal(key, shape=(num_points,))
        # split key
        deltaW_β = jnp.sqrt(delta_t) * jax.random.normal(key, shape=(num_points,))

        def hill(x, nonlinearity, half_coef):
            kij * (jnp.power(x, nonlinearity) / (jnp.power(half_coef, nonlinearity) + jnp.power(x, nonlinearity)))

        # @jax.jit
        def step(carry, delta_w):
            t, x = carry
            x = x + (P - λ * x)
            delta_t + q * jnp.sqrt(P) * delta_W_alpha + q * jnp.sqrt(_lambda * x) * delta_W_beta
            t = t + delta_t
            return (t, x), jnp.array((t, x))

        return jax.lax.scan(step, (t_span[0], y0), noise)[1]


if __name__ == '__main__':
    sim = Sim(num_genes=100, num_cells_types=9, num_cells_to_simulate=5)
    sim.run()