import functools
import time
from copy import deepcopy

import jax
import jax.numpy as jnp
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from src.load_utils import load_grn_jax, topo_sort_graph_layers, get_basal_production_rate

np.seterr(invalid="raise")


class Sim:
    hill_coefficient = 2

    def __init__(self, num_genes: int, num_cells_types: int,
                 interactions_filepath: str, regulators_filepath: str,
                 simulation_num_steps: int, num_samples_from_trajectory: int = None,
                 noise_amplitude: float = 1.,
                 seed=np.random.randint(1000000),
                 **kwargs):

        self.num_genes = num_genes
        self.num_cell_types = num_cells_types
        self.interactions_filename = interactions_filepath
        self.regulators_filename = regulators_filepath
        self.simulation_num_steps = simulation_num_steps
        self.num_samples_from_trajectory = num_samples_from_trajectory  # use in future when trajectory is long 1M steps
        self.noise_parameters_genes = jnp.repeat(noise_amplitude, num_genes)

        self.adjacency = jnp.zeros(shape=(self.num_genes, self.num_genes))
        self.decay_lambda = 0.8
        self.mean_expression = -1 * jnp.ones((num_genes, num_cells_types))
        print("simulation num step per trajectories: ", self.simulation_num_steps)
        self.half_response = jnp.zeros(num_genes)
        self.dt = 0.01

        self.layers = None
        self.seed = seed

    def next_seed(self):
        """Quick seed change for next rollout"""
        self.seed += 1

    def build(self):
        adjacency, graph = load_grn_jax(self.interactions_filename, self.adjacency)

        self.adjacency = jnp.array(adjacency)  # TODO change it to jnp
        self.regulators_dict = dict()
        self.is_repressive = list()

        for g in range(self.num_genes):
            is_regulator = self.adjacency[:, g]
            regulator_indices = np.where(is_regulator)[0]

            if len(regulator_indices) > 0:
                # regulators.append(regulator_indices)
                self.regulators_dict[g] = regulator_indices
                repressive_dict = (self.adjacency[:, g, None].repeat(self.num_cell_types, axis=1) < 0).tolist()
            else:
                repressive_dict = (np.zeros((self.num_genes, self.num_cell_types), dtype=bool)).tolist()

            self.is_repressive.append(tuple(tuple(vv) for vv in repressive_dict))

        self.is_repressive = tuple(self.is_repressive)

        self.layers = topo_sort_graph_layers(graph)
        return adjacency, graph, self.layers

    def run_one_rollout(self, actions=None):
        """return the gene expression of shape [samples, genes]"""
        basal_production_rates = jnp.array(
            get_basal_production_rate(self.regulators_filename, self.num_genes, self.num_cell_types))
        if actions is not None:
            basal_production_rates = jnp.zeros((self.num_genes, self.num_cell_types))
            for action_gene, master_id in zip(actions, self.layers[0]):
                print("action:", action_gene.primal)
                new_gene_expression = jax.nn.relu(action_gene) # basal_production_rates[master_id] * (action_gene+0.001)
                # print("debug:", basal_production_rates[master_id] * (action_gene+0.001))
                print("looking inside...", new_gene_expression.primal, action_gene.primal)
                basal_production_rates = basal_production_rates.at[master_id].set(new_gene_expression)
        # else:
        #     basal_production_rates = jnp.array(
        #         get_basal_production_rate(self.regulators_filename, self.num_genes, self.num_cell_types))

        self.next_seed()
        x = self.simulate_expression_layer_wise(basal_production_rates, seed=self.seed)
        return x

    def simulate_expression_layer_wise(self, basal_production_rates, seed=0):
        y_axis_plot = []

        key = jax.random.PRNGKey(seed)
        key, subkey = jax.random.split(key)
        subkeys = jax.random.split(subkey, self.num_cell_types)

        layers_copy = [np.array(l) for l in deepcopy(self.layers)]  # TODO: remove deepcopy
        master_layer = np.array(layers_copy[0])

        start_master_layer = time.time()
        init_master_layer_concentration = (basal_production_rates[master_layer] / self.decay_lambda)
        x_0 = dict(zip(master_layer, init_master_layer_concentration))
        curr_genes_expression = x_0

        trajectory_master_layer = jax.vmap(self.simulate_master_layer, in_axes=(1, 0, None, 0))(
            basal_production_rates, curr_genes_expression, master_layer, subkeys)
        x = {master_layer[i]: jnp.vstack((x_0[master_layer[i]].reshape(1, -1),
                                          trajectory_master_layer.T[:, i, :])) for i in range(len(master_layer))}
        mean_expression = {idx: jnp.mean(x[idx], axis=0) for idx in master_layer}

        runtime = time.time() - start_master_layer
        # print("master layer took", time.time() - start_master_layer, f"#genes {len(master_layer)}")
        y_axis_plot.append(runtime)

        for num_layer, layer in enumerate(layers_copy[1:], start=1):
            start_layer = time.time()
            key, subkey = jax.random.split(key)
            subkeys = jax.random.split(subkey, self.num_cell_types)

            half_response = dict(zip(layer, self.calculate_half_response(tuple(layer), mean_expression)))

            x_0.update(dict(zip(layer, self.init_concentration(tuple(layer), half_response, mean_expression))))

            curr_genes_expression = x_0
            params = Params(self.regulators_dict, self.adjacency, self.is_repressive)

            production_rates = jnp.array(
                [calculate_production_rate(params, gene, half_response, x) for gene in
                 layer])
            trajectories = jax.vmap(self.simulate_targets, in_axes=(2, 0, None, 0))(
                production_rates, curr_genes_expression, layer, subkeys
            )
            x.update({layer[i]: jnp.vstack((x_0[layer[i]].reshape(1, -1), trajectories.T[:, i, :])) for i in
                      range(len(layer))})
            mean_expression.update({idx: jnp.mean(x[idx], axis=0) for idx in layer})

            runtime = time.time() - start_layer
            # print(f"num layer {num_layer} took {runtime}", f"#genes {len(layer)}")
        #     y_axis_plot.append(runtime)
        #
        # plt.figure(figsize=(4, 4))
        # plt.title("runtime of layers compiling")>>>>>>> main
        # step = 1.0/len(y_axis_plot)
        # x_pos = np.arange(0.0, 1.0, step)[:len(y_axis_plot)]
        # width = step-(step/3)  # width has to be lower than 'step'
        # plt.bar(x_pos, y_axis_plot, color='pink', width=width, align='center')
        # # plt.xticks(y_axis_plot, np.arange(0, len(y_axis_plot)))
        # plt.xlabel("layer")
        # plt.ylabel("secs")
        # plt.tight_layout()
        # plt.show()
        return x

    def simulate_master_layer(self, basal_production_rate, curr_genes_expression, layer, key):
        subkeys = jax.random.split(key, len(layer))
        dx = jax.vmap(euler_maruyama_master, in_axes=(0, 0, 0, 0, None, None))(
            jnp.array([curr_genes_expression[gene] for gene in layer]),
            basal_production_rate.take(jnp.array(layer)),
            self.noise_parameters_genes.take(jnp.array(layer)),
            subkeys,
            self.simulation_num_steps,
            self.dt
        )
        return dx

    def simulate_targets(self, production_rates, curr_genes_expression, layer, key):
        subkeys = jax.random.split(key, len(layer))
        dx = jax.vmap(euler_maruyama_targets, in_axes=(0, 0, 0, 0, None, None))(
            jnp.array([curr_genes_expression[gene] for gene in layer]),
            self.noise_parameters_genes.take(jnp.array(layer)),
            production_rates,
            subkeys,
            self.simulation_num_steps,
            self.dt
        )
        return dx

    # @functools.partial(jax.jit, static_argnums=(0, 1))
    def calculate_half_response(self, layer, mean_expression):
        half_responses = []
        for gene in layer:
            regulators = self.regulators_dict[gene]
            mean_expression_per_cells_regulators_wise = jnp.vstack(tuple([mean_expression[reg] for reg in regulators]))
            half_response = jnp.mean(mean_expression_per_cells_regulators_wise)
            half_responses.append(half_response)
        return half_responses

    # @functools.partial(jax.jit, static_argnums=(0,))
    def init_concentration(self, layer: list, half_response, mean_expression):
        """ Init concentration genes; Note: calculate_half_response should be run before this method """
        params = Params(self.regulators_dict, self.adjacency, self.is_repressive)
        rates = jnp.array([calculate_production_rate_init(params, gene, half_response, mean_expression) for gene in
                           layer])
        rates = rates / self.decay_lambda
        return rates


@functools.partial(jax.jit, static_argnums=(4, 5))
def euler_maruyama_master(curr_genes_expression, basal_production_rate, q, key, simulation_num_steps, dt):
    production_rates = basal_production_rate
    key, subkey = jax.random.split(key)
    dw_p = jax.random.normal(subkey, shape=(simulation_num_steps - 1,))
    key, subkey = jax.random.split(key)
    dw_d = jax.random.normal(subkey, shape=(simulation_num_steps - 1,))

    def concentration_forward(carry, noises):
        curr_concentration = carry
        decayed_production = jnp.multiply(0.8, curr_concentration)
        dw_production, dw_decay = noises

        amplitude_d = q * jnp.power(decayed_production, 0.5)
        amplitude_p = q * jnp.power(production_rates, 0.5)

        decay = jnp.multiply(0.8, curr_concentration)

        noise = jnp.multiply(amplitude_p, dw_production) + jnp.multiply(amplitude_d, dw_decay)

        next_gene_conc = curr_concentration + (dt * jnp.subtract(production_rates, decay))  + jnp.power(dt, 0.5) * noise

        next_gene_conc = jax.nn.relu(next_gene_conc)
        # next_gene_conc = jax.nn.softplus(next_gene_conc)
        return next_gene_conc, next_gene_conc

    noises = (dw_p, dw_d)
    gene_trajectory_concentration = \
        jax.lax.scan(concentration_forward, curr_genes_expression, noises, length=simulation_num_steps - 1)[1]
    return gene_trajectory_concentration


@functools.partial(jax.jit, static_argnums=(4, 5))
def euler_maruyama_targets(curr_genes_expression, q, production_rates, key, simulation_num_steps, dt):
    key, subkey = jax.random.split(key)
    dw_p = jax.random.normal(subkey, shape=(simulation_num_steps - 1,))
    key, subkey = jax.random.split(key)
    dw_d = jax.random.normal(subkey, shape=(simulation_num_steps - 1,))

    def step(carry, state):
        curr_x = carry
        dw_d_t, dw_p_t, production_rate_t = state
        decay = jnp.multiply(0.8, curr_x)
        amplitude_p = q * jnp.power(production_rate_t, 0.5)
        amplitude_d = q * jnp.power(decay, 0.5)
        noise = jnp.multiply(amplitude_p, dw_p_t) + jnp.multiply(amplitude_d, dw_d_t)
        next_x = curr_x + (dt * jnp.subtract(production_rate_t, decay))  + jnp.power(dt, 0.5) * noise
        next_x = jax.nn.relu(next_x)
        # next_x = jax.nn.softplus(next_x)
        return next_x, next_x

    all_state = dw_d, dw_p, production_rates[1:]
    gene_trajectory = jax.lax.scan(step, curr_genes_expression, all_state)[1]
    return gene_trajectory


class Params:
    def __init__(self, regulators_dict, adjacency, repressive_dict):
        self.regulators_dict = regulators_dict
        self.adjacency = adjacency
        self.repressive_dict = repressive_dict


def calculate_production_rate_init(params: Params, gene, half_response, mean_expression):
    regulators = params.regulators_dict[gene]
    mean_expression = jnp.vstack(tuple([mean_expression[reg] for reg in regulators]))
    half_response = half_response[gene]
    is_repressive = jnp.array([params.repressive_dict[gene][regulator] for regulator in regulators])
    absolute_k = jnp.abs(params.adjacency[regulators][:, gene])

    # rate = hill_function_at_init(
    #     mean_expression,
    #     half_response,
    #     is_repressive,
    #     absolute_k
    # )

    rate = jax.vmap(hill_function, in_axes=(0, None, 0, 0))(mean_expression, half_response, is_repressive,
                                                            absolute_k)
    rate = rate.sum(axis=0)
    return rate


def calculate_production_rate(params: Params, gene, half_response, previous_layer_trajectory):
    regulators = params.regulators_dict[gene]
    regulators_concentration = jnp.stack(tuple(previous_layer_trajectory[reg] for reg in regulators))
    half_response = half_response[gene]
    is_repressive = jnp.array([params.repressive_dict[gene][regulator] for regulator in regulators])
    absolute_k = jnp.abs(params.adjacency[regulators][:, gene])

    # rate = hill_function(
    #     regulators_concentration,
    #     half_response,
    #     is_repressive,
    #     absolute_k
    # )

    rate = jax.vmap(hill_function, in_axes=(0, None, 0, 0))(regulators_concentration, half_response, is_repressive,
                                                            absolute_k)
    rate = rate.sum(axis=0)
    return rate


# @functools.partial(jax.jit, static_argnums=(2,))
@jax.jit
def hill_function_at_init(mean_expression, half_response, is_repressive, absolute_k):
    # is_repressive = jnp.array(is_repressive)
    nom = jnp.power(mean_expression, Sim.hill_coefficient)
    denom = (jnp.power(half_response, Sim.hill_coefficient) + jnp.power(mean_expression, Sim.hill_coefficient))
    rate = nom / denom
    rate2 = jnp.where(is_repressive, 1 - rate, rate)
    # k_rate = jnp.einsum("r,rt->t", absolute_k, rate2)
    return rate2 * absolute_k  # k_rate


# @functools.partial(jax.jit, static_argnums=(2,))
@jax.jit
def hill_function(regulators_concentration, half_response, is_repressive, absolute_k):
    """in einsum r,t,rts->t, r is the number of regulators, t is time t, s is the state or cell type."""
    # is_repressive = jnp.array(is_repressive)
    nom = jnp.power(regulators_concentration, Sim.hill_coefficient)
    denom = (jnp.power(half_response, Sim.hill_coefficient) + jnp.power(regulators_concentration, Sim.hill_coefficient))
    rate = (nom / denom)  # .swapaxes(0, 1)
    rate2 = jnp.where(is_repressive, 1 - rate, rate)
    # k_rate = jnp.einsum("r,trs->ts", absolute_k, rate2)
    return rate2 * absolute_k  # k_rate
