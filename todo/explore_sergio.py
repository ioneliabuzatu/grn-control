import time

import numpy as np
import pandas as pd
import experiment_buddy

# import config
# from SERGIO.SERGIO.sergio import sergio
# from SERGIO.Demo.sergio import sergio
from steady_state import sergio
# from sergio_plot_trajectories import sergio

params = {'num_genes': 100, 'num_cells_types': 9, 'num_cells_to_simulate': 5}
experiment_buddy.register_defaults(params)
writer = experiment_buddy.deploy()


def steady_state(number_genes=None,
                 number_bins=None,
                 number_sc=None,
                 noise_params=None,
                 decays=None,
                 sampling_state=None,
                 noise_type=None,
                 input_file_targets=None,
                 input_file_regs=None):
    sim = sergio(number_genes=number_genes,
                 number_bins=number_bins,
                 number_sc=number_sc,
                 noise_params=noise_params,
                 decays=decays,
                 sampling_state=sampling_state,
                 noise_type=noise_type)
    sim.build_graph(input_file_taregts=input_file_targets, input_file_regs=input_file_regs, shared_coop_state=2)
    sim.simulate()
    plot_trajectory_from_sim(sim)
    expression = sim.getExpressions()  # shape(#types, #genes, #trajectories)
    print(expression.shape)
    # expr_add_outlier_genes = sim.outlier_effect(expression, outlier_prob=0.01, mean=0.8, scale=1)
    # libFactor, expr_O_L = sim.lib_size_effect(expr_add_outlier_genes, mean=4.6, scale=0.4)
    # binary_ind = sim.dropout_indicator(expr_O_L, shape=6.5, percentile=82)
    # expr_O_L_D = np.multiply(binary_ind, expr_O_L)
    # count_matrix_umi_count_format = sim.convert_to_UMIcounts(expr_O_L_D)
    # count_expression_matrix = np.concatenate(count_matrix_umi_count_format, axis=1)
    # transposed_count_matrix = count_expression_matrix.T
    # return transposed_count_matrix


def differentiated_states(bmat_filepath,
                          targets_filepath,
                          regs_filepath,
                          number_of_cell_types,
                          genes_number):
    df = pd.read_csv(bmat_filepath, sep='\t', header=None, index_col=None)
    bMat = df.values
    sim = sergio(number_genes=genes_number, number_bins=number_of_cell_types, number_sc=1, noise_params=0.2,
                 decays=0.8, sampling_state=1, noise_params_splice=0.07, noise_type='dpd',
                 dynamics=True, bifurcation_matrix=bMat)
    sim.build_graph(input_file_taregts=targets_filepath, input_file_regs=regs_filepath, shared_coop_state=2)
    sim.simulate_dynamics()
    exprU, exprS = sim.getExpressions_dynamics()
    exprU_O, exprS_O = sim.outlier_effect_dynamics(exprU, exprS, outlier_prob=0.01, mean=0.8, scale=1)
    libFactor, exprU_O_L, exprS_O_L = sim.lib_size_effect_dynamics(exprU_O, exprS_O, mean=4.6, scale=0.4)
    binary_indU, binary_indS = sim.dropout_indicator_dynamics(exprU_O_L, exprS_O_L, shape=6.5, percentile=82)
    exprU_O_L_D = np.multiply(binary_indU, exprU_O_L)
    exprS_O_L_D = np.multiply(binary_indS, exprS_O_L)
    count_matrix_U, count_matrix_S = sim.convert_to_UMIcounts_dynamics(exprU_O_L_D, exprS_O_L_D)
    count_matrix_U = np.concatenate(count_matrix_U, axis=1)
    count_matrix_S = np.concatenate(count_matrix_S, axis=1)
    return count_matrix_U, count_matrix_S


def plot_trajectory_from_sim(sim):
    layers = sim.level2verts_
    for t in range(len(layers[0][0][0].Conc)):
        for _, layer in layers.items():
            for gene in layer:
                for cell_type in gene:
                    if cell_type.ID in [44,1,99]:
                        writer.add_scalar(f"gene{cell_type.ID}/type{cell_type.binID}", cell_type.Conc[t], t)
                    # if cell_type.Type == "MR":
                    #     writer.add_scalar(f"gene{cell_type.ID}/type{cell_type.binID}", cell_type.Conc[t], t)

if __name__ == "__main__":
    start = time.time()
    steady_state(number_genes=100,
                 number_bins=9,
                 number_sc=300,
                 noise_params=1,
                 decays=0.8,
                 sampling_state=1,
                 noise_type='dpd',
                 input_file_targets="../SERGIO/data_sets/De-noised_100G_9T_300cPerT_4_DS1/Interaction_cID_4.txt",
                 input_file_regs="../SERGIO/data_sets/De-noised_100G_9T_300cPerT_4_DS1/Regs_cID_4.txt")
    #
    # differentiated_states(config.filepath_small_dynamics_bifurcation_matrix,
    #                       config.filepath_small_dynamics_targets,
    #                       config.filepath_small_dynamics_regulons,
    #                       number_of_cell_types=2,
    #                       genes_number=12)

    print(f"Took {time.time() - start:.4f} sec.")