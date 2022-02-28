#!/usr/bin/env python

import numpy as np
import seaborn as sns
import pandas as pd
from SERGIO.SERGIO.sergio import sergio
from time import time
import matplotlib.pyplot as plt
import json
import sys



def run_sergio_full_dynamics(how_many_runs = 1):
    # df = pd.read_csv(
    #     'SERGIO/data_sets/De-noised_100G_6T_300cPerT_dynamics_7_DS6/bMat_cID7.tab', sep='\t', header=None, index_col=None)

    timing_runs = np.zeros(shape=how_many_runs)
    mean_runs = np.zeros(shape=how_many_runs)
    std_runs = np.zeros(shape=how_many_runs)

    mean_genes_run = np.zeros(shape=(how_many_runs, 100))
    std_genes_run = np.zeros(shape=(how_many_runs, 100))

    for run in range(how_many_runs):
        start_time = time()
        sim = sergio(number_genes=100, number_bins=9, number_sc=10, noise_params=0.2, decays=0.8, sampling_state=1,
                     noise_params_splice=0.07, noise_type='dpd', dynamics=False)
        sim.build_graph(input_file_taregts='../SERGIO/data_sets/De-noised_100G_9T_300cPerT_4_DS1/Interaction_cID_4.txt',
                        input_file_regs='../SERGIO/data_sets/De-noised_100G_9T_300cPerT_4_DS1/Regs_cID_4.txt',
                        shared_coop_state=2)
        sim.simulate()
        expr = sim.getExpressions()
        print(expr.shape)
        # exprU_O, exprS_O = sim.outlier_effect_dynamics(exprU, exprS, outlier_prob=0.01, mean=0.8, scale=1)
        # libFactor, exprU_O_L, exprS_O_L = sim.lib_size_effect_dynamics(exprU_O, exprS_O, mean=4.6, scale=0.4)
        # binary_indU, binary_indS = sim.dropout_indicator_dynamics(exprU_O_L, exprS_O_L, shape=6.5, percentile=82)
        # exprU_O_L_D = np.multiply(binary_indU, exprU_O_L)
        # exprS_O_L_D = np.multiply(binary_indS, exprS_O_L)
        # count_matrix_U, count_matrix_S = sim.convert_to_UMIcounts_dynamics(exprU_O_L_D, exprS_O_L_D)
        # count_matrix_U = np.concatenate(count_matrix_U, axis=1).T
        # count_matrix_S = np.concatenate(count_matrix_S, axis=1).T

        tot_time = (time() - start_time)/60
        print(f"Tot time: {tot_time}.")

        timing_runs[run] = tot_time
        # mean_runs[run] = count_matrix_S.mean()
        # std_runs[run] = count_matrix_S.std()
        # mean_genes_run[run] = count_matrix_S.mean(axis=0)
        # std_genes_run[run] = count_matrix_S.std(axis=0)
        print(f"run #{run} took {tot_time}")

        # np.savez('resources/experiments/timing_runs_sergio.npz',
        #          timing=timing_runs,
        #          means_runs = mean_runs,
        #          mean_genes_run = mean_genes_run,
        #          std_runs = std_runs,
        #          std_genes_run = std_genes_run
        #          )
#

def heatmap2d(arr: np.ndarray):
    plt.imshow(arr, cmap='viridis')
    plt.colorbar()
    plt.show()


def plot_stats():

    stats = np.load("../resources/experiments/timing_runs_sergio.npz")
    timing = stats["timing"]
    means_runs = stats["means_runs"]
    std_runs = stats["std_runs"]
    mean_genes_run = stats["mean_genes_run"]
    std_genes_run = stats["std_genes_run"]

    x = np.array(list(range(len(timing))))

    plt.figure(1)
    plt.subplot(3, 1, 1)
    plt.title("timing")
    plt.scatter(x,  timing)
    # plt.subplot(3, 1, 2)
    # plt.title("mean")
    # plt.scatter(x, means_runs)
    # plt.subplot(3, 1, 3)
    # plt.title("std")
    # plt.scatter(x, std_runs)
    plt.show()

    # plt.figure(2, figsize=(20, 5))
    # plt.imshow(mean_genes_run, cmap='hot', interpolation='nearest')
    # plt.show()
    #
    # plt.figure(3, figsize=(20, 5))
    # ax = sns.heatmap(mean_genes_run, linewidth=0.01)
    # plt.show()
    #
    # plt.figure(4, figsize=(16, 5))
    # ax = sns.heatmap(std_genes_run, linewidth=0.5)
    # plt.show()
    #
    # # plt.figure(5)
    # plt.figure(5, figsize=(24, 4))
    # heatmap2d(mean_genes_run)
    #
    # plt.figure(6, figsize=(20, 4))
    # heatmap2d(std_genes_run)

######################################################################################################################
# Tot time: 3.2498849034309387 for 10 cells 6T file.


if __name__ == "__main__":
    run_sergio_full_dynamics(how_many_runs=1)
    # plot_stats()
