import time
import os
import jax
import jax.numpy as jnp
import numpy as np
from matplotlib import pyplot as plt

from jax_simulator import Sim as jax_sim
# from speedy_jax_simulator import Sim as jax_sim
from numpy_simulator import Sim as np_sim
from src.all_about_visualization import plot_heatmap_all_expressions
# from src.models.expert.classfier_cell_state import CellStateClassifier, torch_to_jax
from src.zoo_functions import create_plot_graph, open_datasets_json
from src.zoo_functions import is_debugger_active, dataset_namedtuple
import time
from src.techinical_noise import AddTechnicalNoiseJax

from matplotlib import pyplot as plt

from src.zoo_functions import dataset_namedtuple, open_datasets_json
import numpy as np

import jax
import jax.numpy as jnp
import torch
import sys
from jax_simulator import Sim
from src.models.expert.classfier_cell_state import CellStateClassifier, MiniCellStateClassifier
from src.models.expert.classfier_cell_state import torch_to_jax
from src.techinical_noise import AddTechnicalNoiseJax
from src.zoo_functions import is_debugger_active
from src.zoo_functions import plot_three_genes
import experiment_buddy
import wandb
import seaborn as sns
from jax.example_libraries import optimizers

# from scipy.spatial import distance_matrix
# from src.all_about_visualization import plot_heatmap_all_expressions

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
    simulation_num_steps = 10

    dataset_dict = {
        "interactions": 'data/GEO/GSE122662/graph-experiments/toy_graph28nodes.txt',
        "regulators": "data/GEO/GSE122662/graph-experiments/28_nodes_MRs.txt",
        "params_outliers_genes_noise": [0.011039100623008497, 1.4255511751527647, 2.35380330573968],
        "params_library_size_noise": [1.001506520357257, 1.7313202816171356],
        "params_dropout_noise": [4.833139049292777, 62.38254284061924],
        "tot_genes": 28,
        "tot_cell_types": 2,
    }

    # which_dataset_name = "Dummy"
    # dataset_dict = open_datasets_json(return_specific_key=which_dataset_name)
    dataset = dataset_namedtuple(*dataset_dict.values())
    sim = jax_sim(num_genes=dataset.tot_genes, num_cells_types=dataset.tot_cell_types,
                  interactions_filepath=dataset.interactions,
                  regulators_filepath=dataset.regulators,
                  simulation_num_steps=simulation_num_steps,
                  noise_amplitude=0.9,
                  )

    start_time = time.time()
    adjacency, graph, layers = sim.build()
    print(f"compiling took {time.time() - start_time} | #layers={len(layers)}")

    create_plot_graph(graph, verbose=False, dataset_name=f"tmp.png")

    if is_debugger_active():
        with jax.disable_jit():
            simulation_num_steps = 10
            sim.simulation_num_steps = simulation_num_steps
            x = sim.run_one_rollout()
    else:
        x = sim.run_one_rollout()

    print(f"simulation took {time.time() - start_time}")
    arr_expr = jnp.zeros(shape=(sim.num_genes, simulation_num_steps, 2))
    for gene in range(sim.num_genes):
        arr_expr = arr_expr.at[gene].set(x[gene])
    all_expr = jnp.vstack([arr_expr[:, :, 0].T, arr_expr[:, :, 1].T])
    # expr_clean = jnp.stack(tuple([x[gene] for gene in range(sim.num_genes)])).swapaxes(0, 1)
    # plot_heatmap_all_expressions(expr_clean.mean(0), layers[0], show=True, close=False)
    # plt.close()
    # print(f"sim output clean x: {expr_clean.shape}.")

    # genes = [expr_clean.T[0, 0], expr_clean.T[0, 1]]
    # plot_2_genes(genes)

    # expr_per_cell_type = [expr_clean[:, :, i]/expr_clean[:, :, i].mean(axis=1, keepdims=True) for i in range(
    #     expr_clean.shape[2])]
    # expr_per_cell_type = [expr_clean[:, :, i] for i in range(expr_clean.shape[2])]
    # all_expr = jnp.vstack(expr_per_cell_type)
    # print(f"concat_clean_x.shape={all_expr.shape}")
    # np.save("simulated_106G_expr.npy", all_expr)

    # np.save("noisy_simulated_106G_expr.npy", all_expr)
    # os.exist()

    print(f"noisy all_expr.shape={all_expr.shape}")
    classifier = MiniCellStateClassifier(num_genes=28, num_cell_types=2).to("cpu")
    loaded_checkpoint = torch.load(
        "data/GEO/GSE122662/graph-experiments/expert_28_genes_2_layer.pth", map_location=lambda storage, loc: storage
    )
    classifier.load_state_dict(loaded_checkpoint)
    classifier.eval()
    classifier = torch_to_jax(classifier, use_simple_model=True)
    output = classifier(all_expr)
    np.set_printoptions(suppress=True)
    print(output.argmax(1))
    print(jax.nn.softmax(output))


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
