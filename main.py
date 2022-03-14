import time

from matplotlib import pyplot as plt

from src.zoo_functions import dataset_namedtuple, open_datasets_json
import numpy as np

import jax
import jax.numpy as jnp
import torch
import sys
from jax_simulator import Sim
from src.models.expert.classfier_cell_state import CellStateClassifier
from src.models.expert.classfier_cell_state import torch_to_jax
from src.techinical_noise import AddTechnicalNoiseJax
from src.zoo_functions import is_debugger_active
from src.zoo_functions import plot_three_genes
import experiment_buddy
import wandb
import seaborn as sns
from jax.example_libraries import optimizers
from scipy.spatial import distance_matrix


def cross_entropy(logprobs, targets):
    """
    targets = jnp.array([0] * output_classifier.shape[0])
    v0: -jnp.mean(jnp.sum(logprobs * targets, axis=1))
    v1: jnp.mean(jnp.sum(output_classifier * jnp.eye(3)[targets], axis=1))

    @logprobs: expert output or could be the softmax output
    @targets: the target class, can be int or the one-hot encoding is choose to use softmax
    """
    cs = jnp.mean(logprobs[:, targets])
    return cs


def control(env, num_episodes, num_cell_types, num_master_genes, expert, visualise_samples_genes=False,
            writer=None, add_technical_noise_function=None):
    start = time.time()

    @jax.jit
    def loss_fn(actions):
        expr = env.run_one_rollout(actions)
        expr = jnp.stack(tuple([expr[gene] for gene in range(env.num_genes)])).swapaxes(0, 1)

        if visualise_samples_genes:
            plot_three_genes(expr.T[0, 44], expr.T[0, 1], expr.T[0, 99], hlines=None, title="expression")

        if add_technical_noise_function is not None:
            expr = add_technical_noise_function.get_noisy_technical_concentration(expr.T).T
        else:
            expr = jnp.concatenate(expr, axis=1).T

        print(expr.shape)
        output_classifier = expert(expr)
        loss = cross_entropy(output_classifier, 2)

        return -loss, expr

    opt_init, opt_update, get_params = optimizers.adam(step_size=0.001)
    opt_state = opt_init(jnp.ones(shape=(num_master_genes, num_cell_types)))

    def update(episode, opt_state_):
        actions = get_params(opt_state_)
        (loss, expr), grad = jax.value_and_grad(loss_fn, has_aux=True)(actions)

        grad = jnp.clip(grad, -1, 1)
        print("loss", loss)
        print(f"grad shape: {grad.shape} \n grad: {grad}")
        print(f"episode#{episode} took {time.time() - start:.3f} secs.")

        writer.run.log({"loss": loss}, step=episode)
        writer.run.log({"grads": wandb.Image(sns.heatmap(grad, linewidth=0.5))}, step=episode)
        plt.close()
        writer.run.log({"actions": wandb.Image(sns.heatmap(actions, linewidth=0.5))}, step=episode)
        plt.close()
        fig = plt.figure(figsize=(10, 7))
        plt.plot(jnp.mean(jnp.array(expr), axis=0))
        writer.run.log({"control/mean": wandb.Image(fig)}, step=episode)
        plt.close()

        return opt_update(episode, grad, opt_state_)

    for episode in range(num_episodes):
        print("Episode#", episode)
        opt_state = update(episode, opt_state)

    print(f"Took {time.time() - start:.3f} secs.")


if __name__ == "__main__":
    ds4_ground_truth_initial_dist = np.load("data/ds4_10k_each_type.npy")
    plt_mean = np.mean(ds4_ground_truth_initial_dist, axis=0)
    plt_std = np.std(ds4_ground_truth_initial_dist, axis=0)

    params = {'num_genes': 100, 'NUM_SIM_CELLS': 100}
    experiment_buddy.register_defaults(params)
    buddy = experiment_buddy.deploy(host="", disabled=False)
    fig = plt.figure(figsize=(10, 7))
    plt.plot(plt_mean)
    buddy.run.log({"ground_truth/mean": fig}, step=0)
    plt.close()
    fig = plt.figure()
    plt.plot(plt_std)
    buddy.run.log({"ground_truth/std": fig}, step=0)
    plt.close()

    # manual calculation of some distance matrix
    mean_samples_wise_t0 = ds4_ground_truth_initial_dist[:10000].mean(axis=0)
    mean_samples_wise_t1 = ds4_ground_truth_initial_dist[10000:20000].mean(axis=0)
    mean_samples_wise_t2 = ds4_ground_truth_initial_dist[20000:].mean(axis=0)
    d01 = mean_samples_wise_t0 - mean_samples_wise_t1
    d02 = mean_samples_wise_t0 - mean_samples_wise_t2
    d12 = mean_samples_wise_t1 - mean_samples_wise_t2
    print(f"distances: \n 0 <-> 1 {abs(d01.sum()):3f} \n 0 <-> 2 {abs(d02.sum()):3f} \n 1 <-> 2 {abs(d12.sum()):3f}")

    dataset_dict = open_datasets_json(return_specific_dataset='DS4')
    dataset = dataset_namedtuple(*dataset_dict.values())

    expert_checkpoint_filepath = "src/models/expert/checkpoints/classifier_ds4.pth"
    classifier = CellStateClassifier(num_genes=dataset.tot_genes, num_cell_types=dataset.tot_cell_types).to("cpu")
    loaded_checkpoint = torch.load(expert_checkpoint_filepath, map_location=lambda storage, loc: storage)
    classifier.load_state_dict(loaded_checkpoint)
    classifier.eval()
    classifier = torch_to_jax(classifier)

    sim = Sim(
        num_genes=dataset.tot_genes, num_cells_types=dataset.tot_cell_types,
        simulation_num_steps=params['NUM_SIM_CELLS'],
        interactions_filepath=dataset.interactions, regulators_filepath=dataset.regulators, noise_amplitude=0.5
    )
    sim.build()

    add_technical_noise = AddTechnicalNoiseJax(
        dataset.tot_genes, dataset.tot_cell_types, params['NUM_SIM_CELLS'],
        dataset.params_outliers_genes_noise, dataset.params_library_size_noise, dataset.params_dropout_noise
    )

    if is_debugger_active():
        with jax.disable_jit():
            control(sim, 100, dataset.tot_cell_types, len(sim.layers[0]), classifier,
                    writer=buddy,
                    add_technical_noise_function=add_technical_noise)
    else:
        control(sim, 100, dataset.tot_cell_types, len(sim.layers[0]), classifier,
                writer=buddy,
                # add_technical_noise_function=add_technical_noise
                )
