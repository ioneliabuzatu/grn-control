from time import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from src.all_about_visualization import plot_heatmap_all_expressions
from src.zoo_functions import create_plot_graph
from src.zoo_functions import is_debugger_active, plot_three_genes, open_datasets_json, dataset_namedtuple
from jax_simulator import Sim


def sample():
    start = time()
    simulation_num_steps = 10
    which_dataset_name = "DS1"
    dataset_dict = open_datasets_json(return_specific_dataset=which_dataset_name)
    dataset = dataset_namedtuple(*dataset_dict.values())
    sim = Sim(num_genes=dataset.tot_genes, num_cells_types=dataset.tot_cell_types,
              interactions_filepath=dataset.interactions,
              regulators_filepath=dataset.regulators,
              simulation_num_steps=simulation_num_steps,
              noise_amplitude=0.8,
              )
    adjacency, graph, layers = sim.build()

    num_chains_per_initial_state = 3

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

    print("shape of the four dimensional array:", four_dims_array.shape)
    print(f"time: {time() - start}.3f")
    plot_three_genes(
        [four_dims_array.T[0, layers[0][0]], four_dims_array.T[0, layers[1][0]], four_dims_array.T[0, layers[2][0]]],
        hlines=None,
        title="expression")

    values = four_dims_array[:, -100:]
    means = values.mean(0).mean(1).mean(1)
    last = four_dims_array[:, -1].mean()
    errors = np.abs(means - last)
    errors_mean = errors.mean()
    t = 0.1 * last if last > 1 else 0.2 * last
    converged = errors_mean < t
    print(f"{converged} converged")
    print(f"time: {time() - start}.3f")


if __name__ == "__main__":
    sample()
