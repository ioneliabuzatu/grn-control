import time

import jax
import jax.numpy as jnp
import numpy as np
from matplotlib import pyplot as plt

from jax_simulator import Sim as jax_sim
from numpy_simulator import Sim as np_sim
from src.all_about_visualization import plot_heatmap_all_expressions
# from src.models.expert.classfier_cell_state import CellStateClassifier, torch_to_jax
from src.zoo_functions import create_plot_graph
from src.zoo_functions import is_debugger_active, dataset_namedtuple

np.seterr(invalid="raise")


def plot_2_genes(genes):
    def return_as_jax_primal(gene):
        if isinstance(gene, jax.interpreters.ad.JVPTracer):
            return gene.primal
        return gene

    _, axes = plt.subplots(1, 2, figsize=(15, 5))

    for plot, gene in enumerate(genes):
        axes[plot].plot(return_as_jax_primal(gene))

    axes[0].set_title('A gene from layer 0')
    axes[1].set_title('A gene from layer 1')

    axes[0].set_ylim(ymin=0)
    axes[1].set_ylim(ymin=0)

    plt.suptitle('gene expression')
    plt.show()


def run_jax_sim():
    start = time.time()
    simulation_num_steps = 100
    # dataset_dict = {
    #     "interactions": "tmp_interactions.txt",  # '1,1,0,-200,2'
    #     "regulators": "tmp_regulators.txt",  # '0,1,5'
    #     "params_outliers_genes_noise": [0.01, 0.8, 1],
    #     "params_library_size_noise": [4.8, 0.3],
    #     "params_dropout_noise": [20, 82],
    #     "tot_genes": 2,
    #     "tot_cell_types": 2
# }
    dataset_dict = {
        "interactions": "data/interactions_random_graph_500_genes.txt",  # '1,1,0,-200,2'
        "regulators": "tmp_regulators.txt",  # '0,1,5'
        "params_outliers_genes_noise": [0.01, 0.8, 1],
        "params_library_size_noise": [4.8, 0.3],
        "params_dropout_noise": [20, 82],
        "tot_genes": 500,
        "tot_cell_types": 2
    }
    dataset = dataset_namedtuple(*dataset_dict.values())
    sim = jax_sim(num_genes=dataset.tot_genes, num_cells_types=dataset.tot_cell_types,
                  interactions_filepath=dataset.interactions,
                  regulators_filepath=dataset.regulators,
                  simulation_num_steps=simulation_num_steps,
                  noise_amplitude=0.8,
                  )

    start_time = time.time()
    adjacency, graph, layers = sim.build()
    print(f"compiling took {time.time() - start_time}")

    # create_plot_graph(graph, verbose=False, dataset_name=f"tmp.png")

    # if is_debugger_active():
    #     with jax.disable_jit():
    #         simulation_num_steps = 10
    #         sim.simulation_num_steps = simulation_num_steps
    #         x = sim.run_one_rollout()
    # else:
    #     x = sim.run_one_rollout(actions=1)

    simulation_num_steps = 10
    sim.simulation_num_steps = simulation_num_steps
    x = sim.run_one_rollout()

    print(f"simulation took {time.time() - start_time}")

    expr_clean = jnp.stack(tuple([x[gene] for gene in range(sim.num_genes)])).swapaxes(0, 1)

    plot_heatmap_all_expressions(expr_clean.mean(0), layers[0], show=True, close=False)
    plt.close()
    print(expr_clean.shape)
    print(f"time: {time.time() - start}.3f")

    genes = [expr_clean.T[0, 0], expr_clean.T[0, 1]]
    plot_2_genes(genes)


def run_numpy_sim():
    start = time.time()
    interactions_filename = 'tmp_interactions.txt'
    regulators_filename = 'tmp_regulators.txt'

    sim = np_sim(num_genes=2, num_cells_types=2, num_cells_to_simulate=1000,
                 interactions=interactions_filename, regulators=regulators_filename,
                 noise_amplitude=0.8,
                 )
    sim.run()
    expr_clean = sim.x
    print(expr_clean.shape)
    print(f"took {time.time() - start} seconds")

    genes = [expr_clean.T[0, 0], expr_clean.T[0, 1]]
    plot_2_genes(genes)


if __name__ == '__main__':
    run_jax_sim()
    # run_numpy_sim()
