from copy import deepcopy

import numpy as np

from src.load_utils import load_grn, topo_sort_graph_layers, get_basal_production_rate


class Sim:

    def __init__(self, num_genes, num_cells_types, num_cells_to_simulate, interactions, regulators, noise_amplitude,
                 seed=123, **kwargs):
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

        np.random.seed(seed)

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

            production_rates = np.array([self.calculate_production_rate(gene, basal_production_rate) for gene in layer])
            ndim_production_rates = production_rates.ndim
            for step in range(1, self.simulation_time_steps):
                curr_genes_expression = self.x[step - 1, layer]

                if ndim_production_rates == 2:
                    dx = self.euler_maruyama(production_rates, curr_genes_expression, list(layer))
                elif ndim_production_rates == 3:
                    dx = self.euler_maruyama(production_rates[:, step, :], curr_genes_expression, list(layer))

                updated_concentration_gene = curr_genes_expression + dx
                self.x[step, layer] = updated_concentration_gene.clip(0)

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
        rates = [self.calculate_production_rate(gene, basal_production_rate, at_initialization=True) for gene in layer]
        self.x[0, layer] = np.array(rates) / self.decay_lambda

    def calculate_production_rate(self, gene, basal_production_rate, at_initialization=False):
        """
        param at_initialization: if True, the mean expression is used in the hill function instead of the
        concentrations - note: there are no concentrations at the initializations yet!
        """
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
                                           at_initialization)
        if at_initialization:
            rate = np.einsum("r,rs->s", absolute_k, hill_function)
        else:
            rate = np.einsum("r,trs->ts", absolute_k, hill_function)

        return rate

    def hill_function(self, mean_expression, regulators_concentration, half_response, is_repressive, at_initialization):
        if at_initialization:
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
        decays = np.multiply(self.decay_lambda, curr_genes_expression)
        dw_p = np.random.normal(size=curr_genes_expression.shape)
        dw_d = np.random.normal(size=curr_genes_expression.shape)
        amplitude_p = np.einsum("g,gt->gt", self.noise_parameters_genes[layer], np.power(production_rates, 0.5))
        amplitude_d = np.einsum("g,gt->gt", self.noise_parameters_genes[layer], np.power(decays, 0.5))
        noise = np.multiply(amplitude_p, dw_p) + np.multiply(amplitude_d, dw_d)
        d_genes = 0.01 * np.subtract(production_rates, decays) + np.power(0.01, 0.5) * noise  # shape=(#genes,#types)
        return d_genes
