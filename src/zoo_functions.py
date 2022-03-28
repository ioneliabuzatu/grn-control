import json
import sys
from collections import namedtuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn.functional
import torch.nn.functional
import jax.interpreters.ad
from networkx.drawing.nx_agraph import graphviz_layout

from src.models.expert.classfier_cell_state import CellStateClassifier, torch_to_jax

dataset_namedtuple = namedtuple('dataset', ('interactions', 'regulators', 'params_outliers_genes_noise',
                              'params_library_size_noise', 'params_dropout_noise', 'tot_genes', 'tot_cell_types'))


def is_debugger_active() -> bool:
    gettrace = getattr(sys, 'gettrace', lambda : None)
    return gettrace() is not None


def plot_three_genes(genes: list, hlines=None, xmax=1500, title=""):
    """sanity check one gene from each layer"""
    assert len(genes) == 3

    _, axes = plt.subplots(1, 3, figsize=(15, 5))

    def return_as_jax_primal(gene):
        if isinstance(gene, jax.interpreters.ad.JVPTracer):
            return gene.primal
        return gene

    for plot, gene in enumerate(genes):
        axes[plot].plot(return_as_jax_primal(gene))

    axes[0].set_title('A gene from layer 0')
    axes[1].set_title('A gene from layer 1')
    axes[2].set_title('A gene from layer 2')

    if hlines is not None:
        axes[0].hlines(y=hlines[0], xmin=0, xmax=xmax, linewidth=2, color='r')
        axes[1].hlines(y=hlines[1], xmin=0, xmax=xmax, linewidth=2, color='r')
        axes[2].hlines(y=hlines[2], xmin=0, xmax=xmax, linewidth=2, color='r')

    plt.suptitle(title)
    plt.show()

    
def convert_mtx_matrix_to_csv_format(mex_dir, counts_filename, features_filename, barcodes_filename):
    """works as bash cmd: `cellranger mat2csv mtx-format/ data_converted.csv`
        where mtx-format containes 3 files [barcodes, features, counts]"""

    read_mex_format_matrix_as_table = scipy.io.mmread(os.path.join(mex_dir, counts_filename))
    features_path = os.path.join(mex_dir, features_filename)
    barcodes_path = os.path.join(mex_dir, barcodes_filename)

    feature_ids = [row[0] for row in csv.reader(gzip.open(features_path, mode="rt"), delimiter="\t")]
    gene_names = [row[1] for row in csv.reader(gzip.open(features_path, mode="rt"), delimiter="\t")]
    feature_types = [row[2] for row in csv.reader(gzip.open(features_path, mode="rt"), delimiter="\t")]
    barcodes = [row[0] for row in csv.reader(gzip.open(barcodes_path, mode="rt"), delimiter="\t")]

    # transform table to pandas dataframe and label rows and columns
    matrix = pd.DataFrame.sparse.from_spmatrix(read_mex_format_matrix_as_table)
    matrix.columns = barcodes
    matrix.insert(loc=0, column="feature_id", value=feature_ids)
    matrix.insert(loc=0, column="gene", value=gene_names)
    matrix.insert(loc=0, column="feature_type", value=feature_types)

    matrix.to_csv("mex_matrix.csv", index=False)


def load_simulator(use_jax: bool, interactions_filepath, regulators_filepath, tot_genes, tot_cell_types,
                   sim_tot_steps, noise_amplitude=0.8):
    if use_jax:
        from jax_simulator import Sim
        sim = Sim(num_genes=tot_genes,
                  num_cells_types=tot_cell_types,
                  interactions_filepath=interactions_filepath,
                  regulators_filepath=regulators_filepath,
                  simulation_num_steps=sim_tot_steps,
                  noise_amplitude=noise_amplitude,
                  )
        sim.build()
        if is_debugger_active():
            with jax.disable_jit():
                x = sim.run_one_rollout()
        else:
            x = sim.run_one_rollout()

        expr_clean = jnp.stack(tuple([x[gene] for gene in range(sim.num_genes)])).swapaxes(0, 1)
    else:
        from numpy_simulator import Sim
        sim = Sim(num_genes=tot_genes,
                  num_cells_types=tot_cell_types,
                  num_cells_to_simulate=sim_tot_steps // 10,
                  interactions=interactions_filepath,
                  regulators=regulators_filepath,
                  noise_amplitude=noise_amplitude,
                  )
        sim.run()
        expr_clean = sim.x

    return expr_clean


def open_datasets_json(filepath: str = "data/data_sets_sergio.json", return_specific_dataset=None) -> dict:
    with open(filepath) as json_file:
        data_sets = json.load(json_file)
        if return_specific_dataset is not None:
            return data_sets[return_specific_dataset]
        else:
            return data_sets


def classify(expr):
    classifier = CellStateClassifier(num_genes=400, num_cell_types=9).to("cpu")
    loaded_checkpoint = torch.load(
        "src/models/expert/checkpoints/expert_ds2.pth", map_location=lambda storage, loc: storage)
    classifier.load_state_dict(loaded_checkpoint)
    classifier.eval()
    classifier = torch_to_jax(classifier)

    predictions = []  # [41, 64, 87, 49, 32, 38, 96, 57, 77] # [28, 89, 88, 35, 93, 64, 78, 91, 91]
    for sample in range(expr.shape[0]):
        output = classifier(expr[sample])
        probs = torch.nn.functional.softmax(torch.Tensor(np.array(output)))
        prediction = probs.argmax()
        predictions.append(prediction)
    print([np.sum(np.array(predictions[start:end]) == truth) for start, end, truth in
           zip([0, 100, 200, 300, 400, 500, 600, 700, 800],
               [100, 200, 300, 400, 500, 600, 700, 800, 900],
               [0, 1, 2, 3, 4, 5, 6, 7, 8])])


def create_plot_graph(graph, verbose=False, dataset_name="rename.png"):
    """
    param graph: the graph from sim.build()
    param verbose: if true the graphs is not clattered
    param dataset_name: name to save the plot
    """
    if not dataset_name.endswith(".png"):
        dataset_name = dataset_name + ".png"

    if verbose:
        p = nx.drawing.nx_pydot.to_pydot(graph)
        p.write_png(dataset_name)
        return
    nx.draw(graph, cmap=plt.get_cmap('jet'), pos=graphviz_layout(graph, prog='dot'), with_labels=True)
    plt.savefig(dataset_name)
