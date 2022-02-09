import csv
import functools
import time
import typing

# import jax.ops
import jax.numpy as np
import numpy as onp
import jax

SPLICED = 0
UNSPLICED = 1

from functools import partial

from jax import jit
import jax.numpy as jnp


def some_hash_function(x):
    return int(jnp.abs(jnp.sum(x)) * 1000)


class HashableArrayWrapper:
    def __init__(self, val):
        self.val = val

    def __hash__(self):
        return some_hash_function(self.val)

    def __eq__(self, other):
        return (isinstance(other, HashableArrayWrapper) and
                jnp.all(jnp.equal(self.val, other.val)))


def gnool_jit(fun, static_array_argnums=()):
    @partial(jit, static_argnums=static_array_argnums)
    def callee(*args):
        args = list(args)
        for i in static_array_argnums:
            args[i] = args[i].val
        return fun(*args)

    def caller(*args):
        args = list(args)
        for i in static_array_argnums:
            args[i] = HashableArrayWrapper(args[i])
        return callee(*args)

    return caller


class Simulator:
    # p 28 of paper
    noise_amplitude_q = 1.
    decay_parameter_lambda = 0

    unspliced_noise_qu = 0.3
    spliced_noise_qs = 0.07

    unspliced_transcript_decay_rate_mu = 0.8
    spliced_transcript_decay_rate_gamma = 0.2
    dt = 0.01  # integration_time_step
    coop_state = 2

    def __init__(self, adj: np.ndarray, bias: np.ndarray, levels: typing.List[np.array]):
        self.num_genes, _, self.num_bins = adj.shape
        self.master_regulators, self.target_genes = levels

        self.bias = bias
        self.adj = adj

        assert (self.bias[self.master_regulators] != 0.).all(), (
            "Master regulators must have non-zero bias"
            "\"For simplicity, we set bi = 0 for genes other than master regulators.\" p23 of the paper"
        )
        assert (self.bias[self.target_genes] == 0.).all(), (
            "Target genes must have zero bias"
            "\"For simplicity, we set bi = 0 for genes other than master regulators.\" p23 of the paper"
        )

        self.repressives = self.adj < 0
        self.adj = onp.abs(self.adj)

    def simulate(self, x0: onp.ndarray):
        x0 = np.array(x0)

        num_genes, num_bins, repressives, adj, bias = self.num_genes, self.num_bins, self.repressives, self.adj, self.bias

        x1 = _transition_fn(x0, repressives, adj, bias)

        iters = 0
        start = time.time()
        while not np.allclose(x1, x0):
            x0 = x1
            x1 = _transition_fn(x0, repressives, adj, bias)

            iters += 1
            if iters % 10 == 0:
                end = time.time()
                print("norm", np.linalg.norm(x1 - x0))
                print(iters, end - start, "seconds, ", iters / (end - start), "iteration per second")

        end = time.time()
        print("DONE", iters, end - start, "seconds, ", (end - start) / iters, "seconds per iteration")

        return x1


def load_grn(targets_file, regulator_file, bifurcations_file):
    """
    :param targets_file: a csv file, one row per targets. Columns: Target Idx, #regulators, regIdx1,...,regIdx(#regs), K1,...,K(#regs), coop_state1,..., coop_state(#regs)
    :param regulator_file: a csv file, one row per master regulators. Columns: Master regulator Idx, production_rate1,...,productions_rate(#bins)
    :param bifurcations_file: is a numpy array (nBins_ * nBins) of <1 values; bifurcation_matrix[i,j] indicates whether cell type i differentiates to type j or not. Its value
        indicates the rate of transition.

    targets_file should not contain any line for master regulators
    """
    levels = [[], []]
    with open(targets_file, 'r') as f:
        num_genes = len(f.readlines())
        # levels.append(num_genes)
    with open(regulator_file, 'r') as f:
        rows = f.readlines()
        num_genes += len(rows)
        num_states = len(rows[0].split(',')[1:])
        # levels.append(num_genes)

    ignored = []
    adjacency = onp.zeros((num_genes, num_genes))
    bias = onp.zeros((num_genes, 2))

    with open(targets_file, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for indRow, row in enumerate(reader):
            target_node_id = int(float(row.pop(0)))
            num_edges = int(float(row.pop(0)))

            if num_edges == 0:
                raise Exception("Error: a master regulator (#Regs = 0) appeared in input")

            source_ids = [int(float(row.pop(0))) for x in range(num_edges)]
            edge_value = [float(row.pop(0)) for x in range(num_edges)]
            coop_state = [float(row.pop(0)) for x in range(num_edges)]
            ignored.extend(coop_state)

            adjacency[source_ids, target_node_id] = edge_value
            levels[1].append(target_node_id)

    adjacency = onp.stack((adjacency,) * num_states, axis=-1)

    with open(regulator_file, 'r') as f:
        for row in csv.reader(f, delimiter=','):
            node_id = int(float(row[0]))
            regulator_strength = onp.array(row[1:]).astype(float)
            levels[0].append(node_id)
            bias[node_id] = regulator_strength

    assert len(set(ignored)) == 1, "Error: coop_state values are not all the same"
    print(f"Ignored {len(ignored)} values:", set(ignored))
    return adjacency, bias, levels


def main():
    input_file_targets = "data/toy/two_cells_types_denoised_100G_dynamics_interactions_grn.txt"
    input_file_regs = "data/toy/two_cells_types_denoised_dynamics_regulons.txt"
    bifurcation_matrix = "data/toy/two_cells_types_denoised_bifurcation.txt"
    adjacency, bias, levels = load_grn(input_file_targets, input_file_regs, bifurcation_matrix)

    sim = Simulator(adjacency, bias, levels)
    onp.random.seed(714)
    x0 = onp.random.random((sim.num_genes, sim.num_bins, 2))  # 2 is for unspliced and spliced
    # init genes
    x0[sim.master_regulators, :, SPLICED] = sim.bias[
                                                sim.master_regulators] / sim.unspliced_transcript_decay_rate_mu  # g[bIdx].append_Conc(np.true_divide(rate, self.decayVector_[g[0].ID]))
    x0[sim.master_regulators, :, UNSPLICED] = 4 * x0[sim.master_regulators, :, SPLICED]

    xT = sim.simulate(x0)


@functools.partial(gnool_jit, static_array_argnums=[1, 2, 3])
def _transition_fn(x0, repressives, adj, bias):
    num_genes, num_bins, num_states = x0.shape
    unspliced, spliced = x0.T
    unspliced, spliced = unspliced.T, spliced.T

    half_response = onp.full((num_genes, num_genes), 0.5)  # TODO; Define half response
    nums = np.expand_dims(np.power(unspliced, Simulator.coop_state), 1).repeat(num_genes, 1)
    denoms = np.power(half_response, Simulator.coop_state) + nums

    saturation_factor = np.einsum("ijs,ijs->ijs", nums, 1 / denoms)  # TODO: This is 99.51% ones

    # TODO: and this is 100 - 99.51 % zeros, turning off the negative contributions :(
    saturation_factor.at[repressives].set(1 - saturation_factor[repressives])
    effective_contributions = np.einsum("ijs,ijs->ijs", saturation_factor, adj)
    production_rate = effective_contributions.sum(axis=1) + bias
    assert (production_rate == production_rate).any()

    # are_active = x0.astype(bool)

    # The paper, at page 28 says 0.8, but at page 26 is says zero, going with zero because that's close to the modeled formulas
    # lambda being zero makes spliced and unspliced decay rate equal for the old unspliced state contribution
    # lambda_ = 0
    u_decay = Simulator.unspliced_transcript_decay_rate_mu

    alfa, beta, phi, omega = onp.random.normal(size=(4, num_genes))

    # This would be different for each transition if lambda is not zero
    decayed_u = np.einsum(",ib -> ib", u_decay, unspliced)

    amplitude_p = np.einsum("i, ib -> ib", alfa, np.power(production_rate, 0.5))
    amplitude_u = np.einsum("i, ib -> ib", beta, np.power(decayed_u, 0.5))
    u1 = (production_rate - decayed_u) * Simulator.dt + (
            Simulator.unspliced_noise_qu * (amplitude_u + amplitude_p)) * np.sqrt(Simulator.dt)

    assert (u1 == u1).any()

    gamma_s = np.einsum(", ib -> ib", Simulator.spliced_transcript_decay_rate_gamma, unspliced)
    unsliced_noise_on_spliced = np.einsum("i, ib -> ib", phi, np.power(decayed_u, 0.5))
    spliced_noise_on_spliced = np.einsum("i, ib -> ib", omega, np.power(gamma_s, 0.5))

    s1 = ((decayed_u - gamma_s) * Simulator.dt + (
            Simulator.spliced_noise_qs * (unsliced_noise_on_spliced + spliced_noise_on_spliced)) * np.sqrt(
        Simulator.dt))
    assert (s1 == s1).any()
    # TODO: The integration constant is assumed to be one but it should be Simulator.dt

    x1 = np.stack([u1, s1], axis=2)
    x1 = np.clip(x1, 0, np.inf)
    assert (x1 == x1).any()
    return x1


if __name__ == '__main__':
    main()