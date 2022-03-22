from time import time

import jax
import jax.numpy as jnp
import numpy as np

from jax_simulator import Sim
from src.zoo_functions import create_plot_graph
from src.zoo_functions import is_debugger_active, plot_three_genes, open_datasets_json, dataset_namedtuple

import matplotlib.pyplot as plt

def next_pow_two(n):
    i = 1
    while i < n:
        i = i << 1
    return i

def autocorr_func_1d(x, norm=True):
    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")
    n = next_pow_two(len(x))

    # Compute the FFT and then (from that) the auto-correlation function
    f = np.fft.fft(x - np.mean(x), n=2 * n)
    acf = np.fft.ifft(f * np.conjugate(f))[: len(x)].real
    acf /= 4 * n

    # Optionally normalize
    if norm:
        acf /= acf[0]

    return acf

def auto_window(taus, c):
    m = np.arange(len(taus)) < c * taus
    if np.any(m):
        return np.argmin(m)
    return len(taus) - 1

def autocorr_new(y, c=5.0):
    f = np.zeros(y.shape[1])
    for yy in y:
        f += autocorr_func_1d(yy)
    f /= len(y)
    taus = 2.0 * np.cumsum(f) - 1.0
    window = auto_window(taus, c)
    return taus[window]

def sample():
    start = time()
    simulation_num_steps = 1000
    which_dataset_name = "dummy"
    dataset_dict = open_datasets_json(return_specific_dataset=which_dataset_name)
    dataset = dataset_namedtuple(*dataset_dict.values())
    sim = Sim(num_genes=dataset.tot_genes, num_cells_types=dataset.tot_cell_types,
              interactions_filepath=dataset.interactions,
              regulators_filepath=dataset.regulators,
              simulation_num_steps=simulation_num_steps,
              noise_amplitude=0.8,
              )
    adjacency, graph, layers = sim.build()
    create_plot_graph(graph, verbose=False, dataset_name=f"{which_dataset_name}.png")

    num_chains_per_initial_state = 4

    # in debugger mode, jax is much slower than in runtime mode, so set the number of simulation num steps very low
    # like 10.
    if is_debugger_active():
        four_dims_array = np.zeros(shape=(num_chains_per_initial_state, 10, dataset.tot_genes, dataset.tot_cell_types))
    else:
        four_dims_array = np.zeros(
            shape=(num_chains_per_initial_state, simulation_num_steps, dataset.tot_genes, dataset.tot_cell_types))

    for i in range(num_chains_per_initial_state):
        if is_debugger_active():
            with jax.disable_jit():
                simulation_num_steps = 10
                sim.simulation_num_steps = simulation_num_steps
                x = sim.run_one_rollout()
        else:
            actions = None
            x = sim.run_one_rollout(actions)

        expr_clean = jnp.stack(tuple([x[gene] for gene in range(sim.num_genes)])).swapaxes(0, 1)
        four_dims_array[i] = expr_clean

    #
    taus = np.empty((simulation_num_steps, dataset.tot_genes, dataset.tot_cell_types))
    delta = 100
    for t in np.arange(simulation_num_steps):
        for gene in np.arange(dataset.tot_genes):
            for ctype in np.arange(dataset.tot_cell_types):
                y = four_dims_array[:, np.maximum(0, t-delta):np.minimum( np.maximum(t+1, t+delta), simulation_num_steps), gene, ctype]
                taus[t, gene, ctype] = autocorr_new(y)

    print("shape of the four dimensional array:", four_dims_array.shape)
    print(f"time: {time() - start}.3f")
    plot_three_genes(four_dims_array[0].T[0, 44], four_dims_array[0].T[0, 1], four_dims_array.T[0, 99], hlines=None,
                     title="expression")
    # print(taus)
    figure, axes = plt.subplots(3, 3)
    for i, axe in enumerate(figure.axes):
        axe.plot(taus[:, i//3, i%3])
    plt.show()


if __name__ == "__main__":
    sample()
