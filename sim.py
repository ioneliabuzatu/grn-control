import csv
from itertools import repeat

import jax
import jax.numpy as jnp
import networkx as nx
import numpy as np
from copy import deepcopy


class Sim:

    def __init__(self, num_genes, num_cells_types, num_cells_to_simulate):
        self.interactions_filename = 'data/Sergio/De-noised_100G_9T_300cPerT_4_DS1/Interaction_cID_4.txt'
        self.regulators_filename = 'data/Sergio/De-noised_100G_9T_300cPerT_4_DS1/Regs_cID_4.txt'
        self.num_genes = num_genes
        self.num_cell_types = num_cells_types
        self.num_cells_to_simulate = num_cells_to_simulate
        self.adjacency = np.zeros(shape=(self.num_genes, self.num_genes))
        self.decay_lambda = 0.8
        self.mean_expression = -1 * jnp.ones((num_genes, num_cells_types))
        self.sampling_state = 10
        self.simulation_time_steps = self.sampling_state * self.num_cells_to_simulate
        self.x = np.zeros(shape=(self.simulation_time_steps, num_genes, num_cells_types))
        self.half_response = np.zeros(num_genes)
        self.hill_coefficient = 2
        self.noise_amplitude = jnp.ones(self.num_genes)  # TODO: Redefine according to article
        self.t_span = (0, 1)
        self.num_points = 100

    def run(self):
        self.adjacency, graph = self.load_grn()
        layers = self.topo_sort_graph_layers(graph)
        print(layers)
        basal_production_rate = self.get_basal_production_rate()  # TODO: Move to initialization?
        # Note: basal rate is zero for non-master regulator genes
        print(basal_production_rate.shape)
        # print(basal_production_rate[:, 0])
        print("above")
        print(self.adjacency.shape)
        # print(self.adjacency[67])

        # get half_response for all genes ahead of time, for all cell
        self.half_response = jax.vmap(jax.vmap(self.get_half_response, in_axes=(0, None)), in_axes=(None, 0))(
            self.adjacency, self.mean_expression.T).T  # TODO: compare to previous half_response recovery function
        print(self.half_response.shape)

        # Initialize concentration of master regulators across all cells
        init_concentration_dict = jax.vmap(self.init_master_reg_one_cell, in_axes=(1, None))(basal_production_rate, layers[0])
        # print(init_concentration_dict)
        # Then, initialize concentration of regulated genes layer per layer, across all genes
        init_concentration_dict = jax.vmap(self.init_regulated_one_cell, in_axes=(None, 1, 0, None))(
            self.adjacency, self.half_response, init_concentration_dict, layers)
        # print(init_concentration_dict.keys())

        # Finally, simulate concentration trajectories across all cells
        key = jax.random.PRNGKey(0)
        subkeys = jax.random.split(key, self.num_cell_types)
        complete_concentration_trajectories = \
            jax.vmap(self.simulate_single_cell, in_axes=(None, None, None, 1, 1, None, 0, None, 0))(
                self.t_span, self.num_points, self.adjacency, basal_production_rate, self.half_response,
                self.noise_amplitude, init_concentration_dict, layers, subkeys
            )

        print(complete_concentration_trajectories.keys())
        print(complete_concentration_trajectories[45].shape)  # should be (9x100)

        # self.simulate_expression_layer_wise(layers, basal_production_rate)

        return complete_concentration_trajectories

    def load_grn(self):
        topo_sort_graph = nx.DiGraph()

        with open(self.interactions_filename, 'r') as f:
            for row in csv.reader(f, delimiter=','):
                target_node_id, num_regulators = int(float(row.pop(0))), int(float(row.pop(0)))
                regulators_nodes_ids = [int(float(row.pop(0))) for x in range(num_regulators)]
                contributions = [float(row.pop(0)) for x in range(num_regulators)]
                coop_state = [float(row.pop(0)) for x in range(num_regulators)]  # TODO add it

                self.adjacency[regulators_nodes_ids, target_node_id] = contributions

                topo_sort_graph.add_weighted_edges_from(
                    zip(regulators_nodes_ids, repeat(target_node_id), contributions)
                )
        return jnp.array(self.adjacency), topo_sort_graph

    @staticmethod
    def topo_sort_graph_layers(graph: nx.DiGraph):
        layers = list(nx.topological_generations(graph))
        return layers

    def get_basal_production_rate(self):
        """this is a user defined parameter. set to 0 but the master regulators
        Example: regulator_id = g0 --> g1 --| g2, in three cell types: 0, 0.5, 1.5, 3
        """
        basal_production_rates = np.zeros((self.num_genes, self.num_cell_types))
        with open(self.regulators_filename, 'r') as f:
            for row in csv.reader(f, delimiter=','):
                master_regulator_node_id = int(float(row.pop(0)))
                b_for_cell_type = np.array(list(map(float, row)))
                basal_production_rates[master_regulator_node_id] = b_for_cell_type

        return jnp.array(basal_production_rates)

    def simulate_expression_layer_wise(self, layers, basal_production_rate):
        layers_copy = deepcopy(layers)
        for layer in layers_copy:
            self.calculate_half_response(layer)
            self.init_concentration(layer, basal_production_rate)

            step = 1
            while layer:
                for gene in layer:
                    curr_gene_expression = self.x[step-1, gene]
                    assert len(curr_gene_expression) == self.num_cell_types
                    production_rate = 1  # TODO self.calculate_production_rate()
                    decay = np.multiply(self.decay_lambda, curr_gene_expression)
                    noise = 1
                    dx = 0.01 * (production_rate - decay) + np.power(0.01, 0.5) * noise

                    updated_concentration_gene = curr_gene_expression + dx
                    self.x[step, gene] = updated_concentration_gene
                    step += 1

                    if step == self.simulation_time_steps:
                        # converged = self.check_for_convergence()
                        # print(f'Did it converged: {converged}')
                        layer.remove(gene)
                        step = 1

            # TODO em
            # TODO convergence

    # def calculate_half_response(self, layer):
    #     for gene in layer:
    #         regulators = np.where(self.adjacency[:, gene] != 0)
    #         if regulators:  # TODO this is wrong, how to check for empty np.where?
    #             mean_expression_per_cells_regulators_wise = self.mean_expression[regulators]
    #             half_response = np.mean(mean_expression_per_cells_regulators_wise)
    #             self.half_response[gene] = half_response

    @staticmethod
    def get_half_response(gene_adjacency, cell_mean_expression):
        regulators = (gene_adjacency != 0).astype(int)
        return jnp.mean(regulators*cell_mean_expression)

    # def init_concentration(self, layer: list, basal_production_rate):  # TODO missing basal_production_rate
    #     """
    #     Initializes the concentration of all genes in the input level
    #     Note: calculate_half_response_ should be run before this method
    #     """
    #
    #     x0 = np.zeros(shape=(self.num_genes, self.num_cell_types))
    #
    #     for gene in layer:
    #         rate = 0
    #         regulators = np.where(self.adjacency[:, gene] != 0)
    #         mean_expression = self.mean_expression[regulators]
    #         absolute_k = np.abs(self.adjacency[regulators][:, gene])
    #         is_repressive = np.expand_dims(self.adjacency[regulators][:, gene] < 0, -1).repeat(self.num_cell_types,
    #                                                                                            axis=-1)
    #         half_response = self.half_response[gene]
    #         hill_function = self.hill_function(mean_expression, half_response, is_repressive)
    #         rate += np.einsum("r,rt->t", absolute_k, hill_function)
    #
    #         x0[gene] = rate / self.decay_lambda
    #
    #     self.x[0, layer] = x0[layer]

    def init_master_reg_one_cell(self, cell_basal_rates, first_layer):
        layer_init_concentration = jax.vmap(self.init_master_regulator)(cell_basal_rates.take(jnp.array(first_layer)))
        return dict(zip(first_layer, layer_init_concentration))

    def init_master_regulator(self, gene_basal_rate):
        return gene_basal_rate/self.decay_lambda

    def init_regulated(self, gene_adjacency, half_response, concentration_dict):
        reduced_gene_adjacency, reduced_half_response, reduced_regulators_concentration =\
            self.reduce_ind_for_hill_fn(gene_adjacency, half_response, concentration_dict)
        return self.hill_functions_sum(reduced_gene_adjacency, reduced_half_response, reduced_regulators_concentration
                                       )/self.decay_lambda

    def init_regulated_one_cell(self, adjacency, half_response, init_concentration_dict, layers):
        for layer in layers[1:]:
            layer_init_concentration = jax.vmap(self.init_regulated, in_axes=(1, None, None))(
                adjacency.take(jnp.array(layer), axis=1), half_response, init_concentration_dict
            )
            init_concentration_dict.update(dict(zip(layer, layer_init_concentration)))
        return init_concentration_dict

    # def hill_function(self, regulators_concentration, half_response, is_repressive):
    #     rate = np.power(regulators_concentration, self.hill_coefficient) / (
    #                 np.power(half_response, self.hill_coefficient) + np.power(regulators_concentration,
    #                                                                           self.hill_coefficient))
    #
    #     rate[is_repressive] = 1 - rate[is_repressive]
    #     return rate

    @staticmethod
    def reduce_ind_for_hill_fn(gene_adjacency, half_response, concentration_dict):
        """Helper fn for production rate rate calculation, modeled as hill fn.
        Filter out multiple 0 entries from adjacency graph. Done outside main function in order to scan
        the main function without redoing this step each time."""

        # # Damn, can't filter out zero entries because of jax jit/vmap size compliance requirements
        # nonzero_indices = jnp.nonzero(gene_adjacency)
        # reduced_gene_adjacency = jnp.take(gene_adjacency, nonzero_indices)
        # reduced_half_response = jnp.take(half_response, nonzero_indices)
        # reduced_regulators_concentration = jnp.array([concentration_dict[i] for i in nonzero_indices])

        # Let's take regulators indices then
        kept_indices = [int(k) for k in concentration_dict]
        reduced_gene_adjacency = jnp.take(gene_adjacency, jnp.array(kept_indices))
        reduced_half_response = jnp.take(half_response, jnp.array(kept_indices))
        reduced_regulators_concentration = jnp.array([concentration_dict[i] for i in kept_indices])

        return reduced_gene_adjacency, reduced_half_response, reduced_regulators_concentration

    def hill_functions_sum(self, reduced_gene_adjacency, reduced_half_response, reduced_regulators_concentration):

        common = jnp.power(reduced_regulators_concentration, self.hill_coefficient)
        rate = common / (jnp.power(reduced_half_response, self.hill_coefficient) + common)

        activator_hill = reduced_gene_adjacency*rate
        repressor_hill = reduced_gene_adjacency * (1-rate)

        return jnp.sum(jnp.where(reduced_gene_adjacency >= 0, activator_hill, repressor_hill))

    def calculate_production_rate(self):
        return

    def get_selected_concentrations_time_steps(self):
        indices_ = np.random.randint(low=-self.simulation_time_steps, high=0, size=self.num_cells_to_simulate)
        return indices_

    def simulate_single_cell(self, t_span, num_points, gene_adjacency, basal_rate, half_response, noise_amplitude, concentration_dict, layers, key):
        full_concentration_dict = {}
        # initial layer
        key, subkey = jax.random.split(key, 2)
        subkeys = jax.random.split(subkey, len(layers[0]))

        master_regulators_full_concentration = \
            jax.vmap(self.euler_maruyama, in_axes=(None, None, 0, 1, 0, None, 0, None, 0, None))(
                t_span, num_points, jnp.array([concentration_dict[gene] for gene in layers[0]]), gene_adjacency.take(jnp.array(layers[0]), axis=1),
                basal_rate.take(jnp.array(layers[0])), half_response, noise_amplitude.take(jnp.array(layers[0])),
                concentration_dict, subkeys, True
            )
        full_concentration_dict.update(dict(zip(layers[0], master_regulators_full_concentration)))

        # subsequent layers
        for layer in layers:
            key, subkey = jax.random.split(key, 2)
            subkeys = jax.random.split(subkey, len(layer))
            layer_regulators_full_concentration = \
                jax.vmap(self.euler_maruyama, in_axes=(None, None, 0, 1, 0, None, 0, None, 0, None))(
                    t_span, num_points, jnp.array([concentration_dict[gene] for gene in layer]),
                    gene_adjacency.take(jnp.array(layer), axis=1),
                    basal_rate.take(jnp.array(layer)), half_response, noise_amplitude.take(jnp.array(layer)),
                    concentration_dict, subkeys, False
                )
            full_concentration_dict.update(dict(zip(layer, layer_regulators_full_concentration)))

        return full_concentration_dict


    def euler_maruyama(self, t_span, num_points, x0, gene_adjacency, basal_rate, half_response, noise_amplitude, concentration_dict, key, master=False):
        delta_t = (t_span[1] - t_span[0]) / num_points
        key, subkey = jax.random.split(key)
        W_alfa = jnp.sqrt(delta_t) * jax.random.normal(subkey, shape=(num_points,))
        key, subkey = jax.random.split(key)
        W_beta = jnp.sqrt(delta_t) * jax.random.normal(subkey, shape=(num_points,))

        reduced_gene_adjacency, reduced_half_response, reduced_regulators_concentration = \
            self.reduce_ind_for_hill_fn(gene_adjacency, half_response, concentration_dict)

        # @jax.jit
        def concentration_forward(carry, state):
            """ Forward concentration trajectory for a given gene"""
            current_x = carry
            if not master:
                current_reg_concentration, delta_W_alfa, delta_W_beta = state
                P = self.hill_functions_sum(reduced_gene_adjacency, reduced_half_response,
                                            current_reg_concentration) + basal_rate
            else:
                delta_W_alfa, delta_W_beta = state
                P = basal_rate
            next_x = current_x + (P - self.decay_lambda * current_x)*delta_t +\
                     noise_amplitude * jnp.sqrt(P) * delta_W_alfa +\
                     noise_amplitude * jnp.sqrt(self.decay_lambda * current_x) * delta_W_beta
            return next_x, next_x

        if not master:
            all_state = reduced_regulators_concentration, W_alfa, W_beta
        else:
            all_state = W_alfa, W_beta
        gene_full_concentration = jax.lax.scan(concentration_forward, x0, all_state)[1]
        return gene_full_concentration


if __name__ == '__main__':
    trajectories = Sim(num_genes=100, num_cells_types=9, num_cells_to_simulate=5).run()  # return complete
    # trajectories for all genes across all cells
