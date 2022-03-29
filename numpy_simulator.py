import time
from copy import deepcopy

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import ttest_ind

from src.load_utils import load_grn, topo_sort_graph_layers, get_basal_production_rate
from src.zoo_functions import plot_three_genes
from src.techinical_noise import AddTechnicalNoise


# np.random.seed(123)


class Sim:

    def __init__(self, num_genes, num_cells_types, num_cells_to_simulate, interactions, regulators, noise_amplitude,
                 **kwargs):
        self.interactions_filename = interactions
        self.regulators_filename = regulators
        self.num_genes = num_genes
        self.num_cell_types = num_cells_types
        self.num_cells_to_simulate = num_cells_to_simulate
        self.adjacency = np.zeros(shape=(self.num_genes, self.num_genes))
        self.decay_lambda = 0.8
        self.mean_expression = -1 * np.ones((num_genes, num_cells_types))
        self.sampling_state = 10
        self.simulation_time_steps = self.sampling_state * self.num_cells_to_simulate
        print("sampling time steps: ", self.simulation_time_steps)
        self.x = np.zeros(shape=(self.simulation_time_steps, num_genes, num_cells_types))
        self.half_response = np.zeros(num_genes)
        self.hill_coefficient = 2

        self.p_value_for_convergence = 1e-3
        self.window_len = 100
        self.noise_parameters_genes = np.repeat(noise_amplitude, num_genes)
        self._x = np.zeros(shape=(num_cells_to_simulate, num_genes, num_cells_types))

        self.hlines = [44, 1, 99]

    def run(self):
        self.adjacency, graph = load_grn(self.interactions_filename, self.adjacency)
        layers = topo_sort_graph_layers(graph)
        basal_production_rate = get_basal_production_rate(self.regulators_filename, self.num_genes, self.num_cell_types)
        self.simulate_expression_layer_wise(layers, basal_production_rate)

    def simulate_expression_layer_wise(self, layers, basal_production_rate):
        random_sampling_state = np.random.randint(low=-self.simulation_time_steps, high=0,
                                                  size=self.num_cells_to_simulate)
        layers_copy = deepcopy(layers)
        for num_layer, layer in enumerate(layers_copy):
            if num_layer != 0:  # not the master layer
                self.calculate_half_response(layer)
            self.init_concentration(layer, basal_production_rate)
            print("layer: ", num_layer)

            production_rates = [self.calculate_production_rate(gene, basal_production_rate) for gene in layer]
            for step in range(1, self.simulation_time_steps):
                curr_genes_expression = self.x[step - 1, layer]
                dx = self.euler_maruyama(production_rates, curr_genes_expression, list(layer))

                updated_concentration_gene = curr_genes_expression + dx
                self.x[step, layer] = updated_concentration_gene.clip(0)  # clipping is important!

            # self.mean_expression[layer] = np.mean(self.x[random_sampling_state][:, layer], axis=0)
            # self._x[:, layer] = self.x[random_sampling_state][:, layer]
            self.mean_expression[list(layer)] = np.mean(self.x[:, list(layer)], axis=0)

    def calculate_half_response(self, layer):

        for gene in layer:
            regulators = np.where(self.adjacency[:, gene] != 0)
            mean_expression_per_cells_regulators_wise = self.mean_expression[regulators]
            half_response = np.mean(mean_expression_per_cells_regulators_wise)
            self.half_response[gene] = half_response

    def init_concentration(self, layer: list, basal_production_rate):
        """ Init concentration genes; Note: calculate_half_response should be run before this method """
        rates = np.array([self.calculate_production_rate(gene, basal_production_rate, init=True) for gene in layer])
        self.x[0, layer] = rates / self.decay_lambda

    def calculate_production_rate(self, gene, basal_production_rate, init=False):
        gene_basal_production = basal_production_rate[gene]
        if (gene_basal_production != 0).all():
            return gene_basal_production

        regulators = np.where(self.adjacency[:, gene] != 0)
        mean_expression = self.mean_expression[regulators]
        regulators_concentration = self.x[:, regulators[0]]
        absolute_k = np.abs(self.adjacency[regulators][:, gene])
        is_repressive = np.expand_dims(self.adjacency[regulators][:, gene] < 0, -1).repeat(self.num_cell_types,
                                                                                           axis=-1)
        half_response = self.half_response[gene]
        hill_function = self.hill_function(mean_expression, regulators_concentration, half_response, is_repressive,
                                           init)
        if init:
            rate = np.einsum("r,rs->s", absolute_k, hill_function)
        else:
            rate = np.einsum("r,trs->ts", absolute_k, hill_function)

        return rate

    def hill_function(self, mean_expression, regulators_concentration, half_response, is_repressive, init):
        if init:
            rate = np.power(mean_expression, self.hill_coefficient) / (
                np.power(half_response, self.hill_coefficient) + np.power(mean_expression,
                                                                          self.hill_coefficient))
            rate[is_repressive] = 1 - rate[is_repressive]
        else:
            rate = np.power(regulators_concentration, self.hill_coefficient) / (
                np.power(half_response, self.hill_coefficient) + np.power(regulators_concentration,
                                                                          self.hill_coefficient))

            rate[:, is_repressive] = 1 - rate[:, is_repressive]
        return rate

    def euler_maruyama(self, production_rates, curr_genes_expression, layer):
        if 44 in layer:
            self.hlines[0] = np.divide(production_rates, 0.8)[layer.index(44), 0]
        if 1 in layer:
            self.hlines[1] = np.divide(production_rates, 0.8)[layer.index(1), 0]
        if 99 in layer:
            self.hlines[2] = np.divide(production_rates, 0.8)[layer.index(99), 0]

        decays = np.multiply(self.decay_lambda, curr_genes_expression)
        dw_p = np.random.normal(size=curr_genes_expression.shape)  # np.random.normal(1)*jnp.ones_like(
        # curr_genes_expression)
        dw_d = np.random.normal(size=curr_genes_expression.shape)
        amplitude_p = np.einsum("g,gt->gt", self.noise_parameters_genes[layer], np.power(production_rates, 0.5))
        amplitude_d = np.einsum("g,gt->gt", self.noise_parameters_genes[layer], np.power(decays, 0.5))
        noise = np.multiply(amplitude_p, dw_p) + np.multiply(amplitude_d, dw_d)
        d_genes = 0.01 * np.subtract(production_rates, decays) + np.power(0.01, 0.5) * noise  # shape=(#genes,#types)
        return d_genes

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


if __name__ == '__main__':
    start = time.time()
    interactions_filename = 'SERGIO/data_sets/De-noised_100G_9T_300cPerT_4_DS1/Interaction_cID_4.txt'
    regulators_filename = 'SERGIO/data_sets/De-noised_100G_9T_300cPerT_4_DS1/Regs_cID_4.txt'

    sim = Sim(num_genes=100, num_cells_types=9, num_cells_to_simulate=30,
              interactions=interactions_filename, regulators=regulators_filename,
              noise_amplitude=0.5
              )
    sim.run()
    expr_clean = sim.x
    print(expr_clean.shape)
    print(f"took {time.time() - start} seconds")

    plot_three_genes([expr_clean.T[0, 44], expr_clean.T[0, 1], expr_clean.T[0, 99]], hlines=None)

    outlier_genes_noises = (0.01, 0.8, 1)
    library_size_noises = (4.8, 0.3)
    dropout_noises = (20, 82)

    expr = AddTechnicalNoise(100, 9, 300, outlier_genes_noises, library_size_noises,
                             dropout_noises).get_noisy_technical_concentration(expr_clean.T)

    print(f"shape clean data: {expr.shape}")
