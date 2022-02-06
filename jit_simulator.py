import functools
from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import ttest_ind

from load_utils import load_grn, topo_sort_graph_layers, get_basal_production_rate


class Sim:

    def __init__(self, num_genes, num_cells_types, num_cells_to_simulate, **kwargs):
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
        self.x = np.zeros(shape=(self.simulation_time_steps, num_genes, num_cells_types))
        self.half_response = np.zeros(num_genes)
        self.hill_coefficient = 2

        self.p_value_for_convergence = 1e-3
        self.window_len = 100
        self.noise_parameters_genes = np.ones(num_genes)

    def run(self):
        adjacency, graph = load_grn(self.interactions_filename, self.adjacency)
        self.adjacency = jnp.array(adjacency)
        regulators, genes = np.where(self.adjacency)
        self.regulators_dict = dict(zip(genes, [np.array(np.where(self.adjacency[:, g])[0]) for g in genes]))
        self.repressive_dict = dict(zip(genes, [self.adjacency[:, g] < 0 for g in genes]))
        # is_repressive = np.expand_dims(self.adjacency[regulators][:, genes] < 0, -1).repeat(self.num_cell_types, axis=-1)

        layers = topo_sort_graph_layers(graph)
        # self.is_regulator = self.adjacency != 0

        basal_production_rate = get_basal_production_rate(self.regulators_filename, self.num_genes, self.num_cell_types)
        self.simulate_expression_layer_wise(layers, basal_production_rate)

    def simulate_expression_layer_wise(self, layers, basal_production_rate):
        layers_copy = [np.array(l) for l in deepcopy(layers)]  # TODO: remove deepcopy
        layer = np.array(layers_copy[0])
        self.x[0, layer] = basal_production_rate[np.array(layer)] / self.decay_lambda

        for step in range(1, self.simulation_time_steps):
            curr_genes_expression = self.x[step - 1, layer]
            dx = self.euler_maruyama_master(basal_production_rate, curr_genes_expression, layer)
            updated_concentration_gene = curr_genes_expression + dx
            self.x[step, layer] = updated_concentration_gene.clip(a_min=0)  # clipping is important!

        self.mean_expression = self.mean_expression.at[layer].set(np.mean(self.x[:, layer], axis=0))

        for num_layer, layer in enumerate(layers_copy[1:], start=1):
            half_responses = self.calculate_half_response(tuple(layer), self.mean_expression)
            self.half_response[layer] = half_responses

            self.init_concentration(tuple(layer), self.half_response, self.mean_expression)

            # TODO(Ioni) Make scan.
            for step in range(1, self.simulation_time_steps):
                curr_genes_expression = self.x[step - 1, layer]
                dx = self.euler_maruyama_targets_layer(curr_genes_expression, layer, self.half_response, self.mean_expression)
                updated_concentration_gene = curr_genes_expression + dx
                self.x[step, layer] = updated_concentration_gene.clip(0)  # clipping is important!

            self.mean_expression[layer] = np.mean(self.x[:, layer], axis=0)

    @functools.partial(jax.jit, static_argnums=(0, 1))  # Jax should ignore the class instance (self), and layer
    def calculate_half_response(self, layer, mean_expression):
        half_responses = []
        for gene in layer:
            regulators = self.regulators_dict[gene]
            mean_expression_per_cells_regulators_wise = mean_expression[regulators]
            half_response = np.mean(mean_expression_per_cells_regulators_wise)
            half_responses.append(half_response)
        return half_responses

    @functools.partial(jax.jit, static_argnums=(0, 1))  # Jax should ignore the class instance (self)
    def init_concentration(self, layer: list, half_response: np.ndarray, mean_expression: np.ndarray):
        """ Init concentration genes; Note: calculate_half_response should be run before this method """
        rates = [self.calculate_production_rate(gene, half_response, mean_expression) for gene in layer]
        rates = np.array(rates)
        self.x[0, layer] = rates / self.decay_lambda

    @functools.partial(jax.jit, static_argnums=(0, 1))  # Jax should ignore the class instance (self)
    def calculate_production_rate(self, gene, half_response, mean_expression):
        # regulators = np.where(self.adjacency[:, gene] != 0)
        # regulators = self.adjacency[:, gene] != 0
        regulators = self.regulators_dict[gene]

        mean_expression = mean_expression[regulators]

        absolute_k = jnp.abs(self.adjacency[regulators][:, gene])

        half_response = half_response[gene]
        is_repressive = self.repressive_dict[gene]
        is_repressive = is_repressive[regulators]

        hill_function = self.hill_function(mean_expression, half_response, is_repressive)
        rate = jnp.einsum("r,rt->t", absolute_k, hill_function)
        return rate

    def hill_function(self, regulators_concentration, half_response, is_repressive):
        rate = (
                jnp.power(regulators_concentration, self.hill_coefficient) /
                (jnp.power(half_response, self.hill_coefficient) + jnp.power(regulators_concentration, self.hill_coefficient))
        )
        rate = rate.at[is_repressive].set(1 - rate[is_repressive])
        return rate

    @functools.partial(jax.jit, static_argnums=(0,))  # Ignore the class instance
    def euler_maruyama_master(self, basal_production_rate, curr_genes_expression, layer):
        production_rates = basal_production_rate[layer]
        decays = jnp.multiply(self.decay_lambda, curr_genes_expression)
        dw_p = np.random.normal(size=curr_genes_expression.shape)  # TODO: use jax and noise control
        dw_d = np.random.normal(size=curr_genes_expression.shape)  # TODO: use jax and noise control
        # amplitude_p = jnp.einsum("g,gt->gt", self.noise_parameters_genes[layer], jnp.power(production_rates, 0.5))
        # amplitude_d = jnp.einsum("g,gt->gt", self.noise_parameters_genes[layer], jnp.power(decays, 0.5))
        amplitude_p = jnp.einsum(",gt->gt", self.noise_parameters_genes[0], jnp.power(production_rates, 0.5))
        amplitude_d = jnp.einsum(",gt->gt", self.noise_parameters_genes[0], jnp.power(decays, 0.5))
        noise = jnp.multiply(amplitude_p, dw_p) + jnp.multiply(amplitude_d, dw_d)
        d_genes = 0.01 * jnp.subtract(production_rates, decays) + jnp.power(0.01, 0.5)  # * noise  # shape=(#genes,#types)
        return d_genes

    @functools.partial(jax.jit, static_argnums=(0,))  # Ignore the class instance
    def euler_maruyama_targets_layer(self, curr_genes_expression, layer, half_response, mean_expression):
        production_rates = [self.calculate_production_rate(gene, half_response, mean_expression) for gene in layer]
        decays = np.multiply(self.decay_lambda, curr_genes_expression)
        dw_p = np.random.normal(size=curr_genes_expression.shape)
        dw_d = np.random.normal(size=curr_genes_expression.shape)
        amplitude_p = jnp.einsum("g,gt->gt", self.noise_parameters_genes, np.power(production_rates, 0.5))
        amplitude_d = jnp.einsum("g,gt->gt", self.noise_parameters_genes, np.power(decays, 0.5))
        noise = np.multiply(amplitude_p, dw_p) + np.multiply(amplitude_d, dw_d)
        d_genes = 0.01 * np.subtract(production_rates, decays) + np.power(0.01, 0.5)  # * noise  # shape=(#genes,#types)
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

    def add_external_noise(self):
        """ sequencing noise """
        raise NotImplementedError


if __name__ == '__main__':
    sim = Sim(num_genes=100, num_cells_types=9, num_cells_to_simulate=5)
    with jax.disable_jit():
        sim.run()
    sim.run()