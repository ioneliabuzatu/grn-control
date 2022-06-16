import csv
import typing
from itertools import repeat

import networkx as nx
import numpy as np


# TODO: We actually don't need adjacency, just the number of genes
def load_grn(interactions_filename, adjacency):
    topo_sort_graph = nx.DiGraph()

    with open(interactions_filename, 'r') as f:
        for row in csv.reader(f, delimiter=','):
            target_node_id, num_regulators = int(float(row.pop(0))), int(float(row.pop(0)))
            regulators_nodes_ids = [int(float(row.pop(0))) for x in range(num_regulators)]
            contributions = [float(row.pop(0)) for x in range(num_regulators)]
            coop_state = [float(row.pop(0)) for x in range(num_regulators)]  # TODO add it

            adjacency[regulators_nodes_ids, target_node_id] = contributions

            topo_sort_graph.add_weighted_edges_from(
                zip(regulators_nodes_ids, repeat(target_node_id), contributions)
            )
    return adjacency, topo_sort_graph


# TODO: We actually don't need adjacency, just the number of genes
def load_grn_jax(interactions_filename, adjacency):
    adjacency = np.zeros(shape=adjacency.shape)
    topo_sort_graph = nx.DiGraph()

    with open(interactions_filename, 'r') as f:
        for row in csv.reader(f, delimiter=','):
            target_node_id, num_regulators = int(float(row.pop(0))), int(float(row.pop(0)))
            regulators_nodes_ids = [int(float(row.pop(0))) for x in range(num_regulators)]
            contributions = [float(row.pop(0)) for x in range(num_regulators)]
            # coop_state = [float(row.pop(0)) for x in range(num_regulators)]  # TODO add it

            # adjacency = adjacency.at[regulators_nodes_ids, target_node_id].set(contributions)
            adjacency[regulators_nodes_ids, target_node_id] = contributions

            topo_sort_graph.add_weighted_edges_from(
                zip(regulators_nodes_ids, repeat(target_node_id), contributions)
            )
    return adjacency, topo_sort_graph


def topo_sort_graph_layers(graph: nx.DiGraph) -> typing.Tuple[typing.Tuple, ...]:
    layers = tuple(tuple(l) for l in nx.topological_generations(graph))
    return layers


def get_basal_production_rate(regulators_filename, num_genes, num_cell_types):
    """this is a user defined parameter. set to 0 but the master regulators
    Example: regulator_id = g0 --> g1 --| g2, in three cell types: 0, 0.5, 1.5, 3
    """
    basal_production_rates = np.zeros((num_genes, num_cell_types))
    with open(regulators_filename, 'r') as f:
        for row in csv.reader(f, delimiter=','):
            master_regulator_node_id = int(float(row.pop(0)))
            b_for_cell_type = np.array(list(map(float, row)))
            basal_production_rates[master_regulator_node_id] = b_for_cell_type

    return basal_production_rates
