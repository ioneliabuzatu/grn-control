from collections import namedtuple
from time import time

import numpy as np

from src.techinical_noise import AddTechnicalNoise
from src.zoo_functions import dataset_namedtuple
from src.zoo_functions import load_simulator, open_datasets_json


def generate(dataset: namedtuple, samples_per_cell_type: int = 100, use_jax_sim: bool = True, save_data_as_npy=False,
             filepath="data/rename.npy", noise_amplitude=1.0):
    start = time()
    expr_clean = load_simulator(
        use_jax_sim,
        dataset.interactions,
        dataset.regulators,
        dataset.tot_genes,
        dataset.tot_cell_types,
        samples_per_cell_type,
        noise_amplitude=noise_amplitude,
    )
    print(expr_clean.shape)
    print(f"time: {time() - start}")

    expr = AddTechnicalNoise(
        dataset.tot_genes, dataset.tot_cell_types, samples_per_cell_type,
        dataset.params_outliers_genes_noise, dataset.params_library_size_noise, dataset.params_dropout_noise
    ).get_noisy_technical_concentration(expr_clean.T)
    expr = expr.T

    if save_data_as_npy:
        print(f"**saving data of shape {expr.shape} to <{filepath}>**")
        np.save(filepath, expr)

    print(f"shape generated data: {expr.shape}")
    return expr


if __name__ == "__main__":
    dataset_dict = open_datasets_json(return_specific_dataset='DS4')
    counts = generate(
        dataset_namedtuple(*dataset_dict.values()),
        samples_per_cell_type=10000,
        use_jax_sim=True,
        save_data_as_npy=True,
        filepath="data/ds4_10k_each_type.npy"
    )
    # classify(counts)
