import functools
import time
from copy import deepcopy

import jax
import jax.numpy as jnp
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
        self.noise_parameters_genes = np.repeat(noise_amplitude, num_genes)

        self.adjacency = jnp.zeros(shape=(self.num_genes, self.num_genes))
        self.decay_lambda = 0.8
        self.mean_expression = -1 * jnp.ones((num_genes, num_cells_types))
        print("simulation num step per trajectories: ", self.simulation_num_steps)
        self.half_response = jnp.zeros(num_genes)
        self.dt = 0.01

        self.layers = None
        self.key = jax.random.PRNGKey(1337)

    @property
    def roots(self):
        return self.layers[0]

    def next_key(self):
        self.key, sub_key = jax.random.split(self.key)
        return sub_key

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
        self.is_repressive = np.array(self.is_repressive)

        self.layers = topo_sort_graph_layers(graph)
        return adjacency, graph, self.layers

    def run_one_rollout(self, actions=None):
        """return the gene expression of shape [samples, genes]"""
        if actions is None:
            actions = np.random.rand(len(self.roots))
            basal_production_rates = np.zeros((self.num_genes, self.num_cell_types))
            # for action_gene, master_id in zip(actions, self.layers[0]):
            #     basal_production_rates = basal_production_rates.at[master_id].set(action_gene)
            basal_production_rates[self.roots] = actions
        else:
            basal_production_rates = np.array(
                get_basal_production_rate(self.regulators_filename, self.num_genes, self.num_cell_types))

        x = self.simulate_expression_layer_wise(basal_production_rates)
        return x

    def simulate_expression_layer_wise(self, basal_production_rates):
        y_axis_plot = []

        start_master_layer = time.time()
        init_master_layer_concentration = (basal_production_rates[self.roots] / self.decay_lambda)
        x_0 = np.zeros((self.num_genes, self.num_cell_types))
        x_0[self.roots] = init_master_layer_concentration
        curr_genes_expression = x_0

        trajectory_master_layer = self.simulate_master_layer(basal_production_rates, curr_genes_expression, self.roots, self.next_key())

        x__ = np.zeros((x_0.shape[0], self.simulation_num_steps, x_0.shape[1]))
        x__[self.roots, 0] = x_0[self.roots]
        x__ = jnp.array(x__)
        x__ = x__.at[self.roots, 1:].set(trajectory_master_layer[self.roots])

        runtime = time.time() - start_master_layer
        print("master layer took", time.time() - start_master_layer)
        y_axis_plot.append(runtime)

        half_responses = jnp.zeros((self.num_genes, self.num_cell_types))

        for num_layer, layer in enumerate(self.layers[1:], start=1):
            start_layer = time.time()
            mean_expression = x__.mean(axis=1)

            hr = self.calculate_half_response(tuple(layer), mean_expression)
            half_responses = half_responses.at[layer].set(hr)
            half_response = half_responses[layer]

            x0_l = self.init_concentration(tuple(layer), half_response, mean_expression)
            x__ = x__.at[layer, 0].set(x0_l)

            params = Params(self.regulators_dict, self.adjacency, self.is_repressive)

            pr = []
            for gene in layer: # TODO: can be made faster
                prg = calculate_production_rate(params, gene, half_response, x__)
                pr.append(prg)

            production_rates = jnp.array(pr)

            k = self.next_key()
            p, x = production_rates[:, :, :], x__[layer, 0, :]
            traj = self.simulate_targets(p, x, layer, k)
            x__ = x__.at[layer, 1:, :].set(traj)

            runtime = time.time() - start_layer
            print(f"num layer {num_layer} took {runtime}, it had {len(layer)} genes")
        return x__

    def simulate_master_layer(self, basal_production_rate, curr_genes_expression, layer, key):
        trajectory_master_layer = []
        x = curr_genes_expression[layer]
        p = basal_production_rate[layer]
        n = self.noise_parameters_genes[layer]
        subkeys = jax.random.split(key, len(layer))
        for xi, pi, ni, gene_key in zip(x, p, n, subkeys):  # TODO: re-add vmap
            dx = euler_maruyama_master(xi, pi, ni, gene_key, self.simulation_num_steps, self.dt)
            trajectory_master_layer.append(dx)
        return jnp.stack(trajectory_master_layer)

    def simulate_targets(self, production_rates, curr_genes_expression, layer, key):
        traj = []
        keys = jax.random.split(key, len(layer))
        for in_layer_idx, gene in enumerate(layer):
            xs, ns = curr_genes_expression[in_layer_idx], self.noise_parameters_genes[in_layer_idx]
            p = production_rates[in_layer_idx]
            k = keys[in_layer_idx]
            gene_traj = euler_maruyama_targets(xs, ns, p, k, self.simulation_num_steps, self.dt)
            traj.append(gene_traj)
        return traj

    # @functools.partial(jax.jit, static_argnums=(0, 1))
    def calculate_half_response(self, layer, mean_expression):
        half_responses = []
        for gene in layer:  # TODO: matrix multiplication
            regulators = self.regulators_dict[gene]
            half_responses.append(mean_expression[regulators].mean(axis=0))
        return half_responses

    # @functools.partial(jax.jit, static_argnums=(0,))
    def init_concentration(self, layer: list, half_response, mean_expression):
        """ Init concentration genes; Note: calculate_half_response should be run before this method """
        params = Params(self.regulators_dict, self.adjacency, self.is_repressive)
        rates = jnp.array([calculate_production_rate_init(params, gene, half_response, mean_expression) for gene in
                           layer])
        rates = rates / self.decay_lambda
        return rates


# @functools.partial(jax.jit, static_argnums=(1, 2, 3, 4, 5))
def euler_maruyama_master(curr_genes_expression, basal_production_rate, q, key, simulation_num_steps, dt):
    production_rates = basal_production_rate
    key, subkey = jax.random.split(key)
    dw_p, dw_d = jax.random.normal(subkey, shape=(2, simulation_num_steps - 1,))

    @jax.jit
    def concentration_forward(curr_concentration, noises):
        # print("master compiling step...")
        decayed_production = jnp.multiply(0.8, curr_concentration)
        dw_production, dw_decay = noises

        amplitude_d = q * jnp.power(decayed_production, 0.5)
        amplitude_p = q * jnp.power(production_rates, 0.5)

        decay = jnp.multiply(0.8, curr_concentration)

        noise = jnp.multiply(amplitude_p, dw_production) + jnp.multiply(amplitude_d, dw_decay)

        next_gene_conc = curr_concentration + (dt * jnp.subtract(production_rates, decay)) + jnp.power(
            dt, 0.5) * noise

        next_gene_conc = jax.nn.relu(next_gene_conc)
        # next_gene_conc = jax.nn.softplus(next_gene_conc)
        return next_gene_conc, next_gene_conc

    noises = (dw_p, dw_d)
    start = time.time()
    _, gene_trajectory_concentration = jax.lax.scan(
        concentration_forward, curr_genes_expression, noises, length=simulation_num_steps - 1)
    runtime = time.time() - start
    print(f"master layer runtime: {runtime}")
    return gene_trajectory_concentration


# @functools.partial(jax.jit, static_argnums=(4, 5))
def euler_maruyama_targets(initial_gene_expresison, q, production_rates, key, simulation_num_steps, dt):
    key, subkey = jax.random.split(key)
    dw_p, dw_d = jax.random.normal(subkey, shape=(2, simulation_num_steps - 1,))

    @jax.jit
    def step(curr_x, state):
        dw_d_t, dw_p_t, production_rate_t = state
        decay = jnp.multiply(0.8, curr_x)
        amplitude_p = q * jnp.power(production_rate_t, 0.5)
        amplitude_d = q * jnp.power(decay, 0.5)
        noise = jnp.multiply(amplitude_p, dw_p_t) + jnp.multiply(amplitude_d, dw_d_t)
        next_x = curr_x + (dt * jnp.subtract(production_rate_t, decay)) + jnp.power(dt, 0.5) * noise
        next_x = jax.nn.relu(next_x)
        return next_x, next_x

    all_state = dw_d, dw_p, production_rates[1:]
    _carry, gene_trajectory = jax.lax.scan(step, initial_gene_expresison, all_state)
    return gene_trajectory


class Params:
    def __init__(self, regulators_dict, adjacency, repressive_dict):
        self.regulators_dict = regulators_dict
        self.adjacency = adjacency
        self.repressive_table = repressive_dict


def calculate_production_rate_init(params: Params, gene, half_response, mean_expression):
    regulators = params.regulators_dict[gene]
    mean_expression = jnp.vstack(tuple([mean_expression[reg] for reg in regulators]))
    half_response = half_response[gene]
    is_repressive = jnp.array([params.repressive_table[gene][regulator] for regulator in regulators])
    absolute_k = jnp.abs(params.adjacency[regulators][:, gene])

    # TODO: remove vmap
    rate = jax.vmap(hill_function, in_axes=(0, None, 0, 0))(mean_expression, half_response, is_repressive, absolute_k)
    rate = rate.sum(axis=0)
    return rate


def calculate_production_rate(params: Params, gene, half_response, previous_layer_trajectory):
    regulators = params.regulators_dict[gene]

    regulators_concentration = previous_layer_trajectory[regulators]
    half_response = half_response[gene]
    is_repressive = params.repressive_table[gene, regulators]
    absolute_k = jnp.abs(params.adjacency[regulators, gene])

    rate = 0
    for x, r, k in zip(regulators_concentration, is_repressive, absolute_k):
        rate += hill_function(
            x,
            half_response,
            r,
            k
        )
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
