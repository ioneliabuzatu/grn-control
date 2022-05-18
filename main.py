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
# from scipy.spatial import distance_matrix
# from src.all_about_visualization import plot_heatmap_all_expressions

# jax.config.update('jax_platform_name', 'cpu')


def cross_entropy(logprobs, targets):
    """
    targets = jnp.array([0] * output_classifier.shape[0])
    v0: -jnp.mean(jnp.sum(logprobs * targets, axis=1))
    v1: jnp.mean(jnp.sum(output_classifier * jnp.eye(3)[targets], axis=1))

    @logprobs: expert output or could be the softmax output
    @targets: the target class, can be int or the one-hot encoding is chosen to use softmax
    """
    cs = jnp.mean(logprobs[:, targets])
    return cs


def control(env, num_episodes, num_cell_types, num_master_genes, expert, visualise_samples_genes=False,
            writer=None, add_technical_noise_function=None):
    start = time.time()

    @jax.jit
    def loss_fn(actions):
        expression = env.run_one_rollout(actions)
        expression = jnp.stack(tuple([expression[gene] for gene in range(env.num_genes)])).swapaxes(0, 1)

        if visualise_samples_genes:
            plot_three_genes(expression.T[0, 44], expression.T[0, 1], expression.T[0, 99], hlines=None, title="expression")

        if add_technical_noise_function is not None:
            concat_expression = add_technical_noise_function.get_noisy_technical_concentration(expression.T).T
        else:
            concat_expression = jnp.concatenate(expression, axis=1).T

        print(concat_expression.shape)
        output_classifier = expert(concat_expression)
        loss = cross_entropy(output_classifier, 2)

        return loss, expression

    opt_init, opt_update, get_params = optimizers.adam(step_size=0.005)
    opt_state = opt_init(jnp.ones(shape=(num_master_genes, num_cell_types)))

    check_expressions_for_convergence = []

    def update(episode, opt_state_, check_expressions_for_convergence):
        actions = get_params(opt_state_)
        (loss, expression), grad = jax.value_and_grad(loss_fn, has_aux=True)(actions)

        grad = jnp.clip(grad, -1, 1)
        # print("loss", loss)
        # print(f"grad shape: {grad.shape} \n grad: {grad}")
        # print(f"episode#{episode} took {time.time() - start:.3f} secs.")

        writer.run.log({"loss": loss}, step=episode)
        writer.run.log({"grads": wandb.Image(sns.heatmap(grad, linewidth=0.5))}, step=episode)
        plt.close()
        writer.run.log({"actions": wandb.Image(sns.heatmap(actions, linewidth=0.5))}, step=episode)
        plt.close()
        fig = plt.figure(figsize=(10, 7))
        plt.plot(jnp.mean(jnp.array(expression), axis=0))
        writer.run.log({"control/mean": wandb.Image(fig)}, step=episode)
        plt.close()

        check_expressions_for_convergence.append(expression)
        if len(check_expressions_for_convergence) == 3:
            expression = jnp.array(check_expressions_for_convergence)
            last_x_points = expression[:, -20]
            mean_x_points = jnp.mean(last_x_points, axis=0)
            var_mean_x_points = jnp.var(last_x_points - mean_x_points, axis=0)
            print((var_mean_x_points < 0.1).sum())
            check_expressions_for_convergence = []

        return opt_update(episode, grad, opt_state_), check_expressions_for_convergence

    for episode in range(num_episodes):
        print("Episode#", episode)
        opt_state, check_expressions_for_convergence = update(episode, opt_state, check_expressions_for_convergence)  # opt_state are the actions

    print(f"Took {time.time() - start:.3f} secs.")


if __name__ == "__main__":
    # ds4_ground_truth_initial_dist = np.load("data/ds4_10k_each_type.npy")

    params = {'num_genes': '', 'NUM_SIM_CELLS': 200}
    experiment_buddy.register_defaults(params)
    buddy = experiment_buddy.deploy(host="", disabled=True)

    # dataset_dict = open_datasets_json(return_specific_key='DS4')
    # dataset = dataset_namedtuple(*dataset_dict.values())

    tot_genes = 500
    tot_cell_types = 2
    interactions_filepath = f"data/interactions_random_graph_{tot_genes}_genes.txt"

    # expert_checkpoint_filepath = "src/models/expert/checkpoints/classifier_ds4.pth"
    classifier = CellStateClassifier(num_genes=tot_genes, num_cell_types=tot_cell_types).to("cpu")
    # loaded_checkpoint = torch.load(expert_checkpoint_filepath, map_location=lambda storage, loc: storage)
    # classifier.load_state_dict(loaded_checkpoint)
    classifier.eval()
    classifier = torch_to_jax(classifier)

    # sim = Sim(
    #     num_genes=dataset.tot_genes, num_cells_types=dataset.tot_cell_types,
    #     simulation_num_steps=params['NUM_SIM_CELLS'],
    #     interactions_filepath=dataset.interactions, regulators_filepath=dataset.regulators, noise_amplitude=0.9
    # )


    sim = Sim(
        num_genes=tot_genes, num_cells_types=tot_cell_types,
        simulation_num_steps=params['NUM_SIM_CELLS'],
        interactions_filepath=interactions_filepath, regulators_filepath="data/Regs_cID_4.txt", noise_amplitude=0.9
    )

    start = time.time()
    adjacency, graph, layers = sim.build()
    print(f"Took {time.time() - start:.3f} secs.")

    # fig = plot_heatmap_all_expressions(
    #     ds4_ground_truth_initial_dist.reshape(3, 100, 10000).mean(2).T,
    #     layers[0],
    #     show=False)
    # buddy.run.log({"heatmap/expression/gd": wandb.Image(fig)}, step=0)

    # add_technical_noise = AddTechnicalNoiseJax(
    #     dataset.tot_genes, dataset.tot_cell_types, params['NUM_SIM_CELLS'],
    #     dataset.params_outliers_genes_noise, dataset.params_library_size_noise, dataset.params_dropout_noise
    # )
    if is_debugger_active():
        with jax.disable_jit():
            control(sim, 5, tot_cell_types, len(sim.layers[0]), classifier,
                    writer=buddy,
                    # add_technical_noise_function=add_technical_noise
                    )
    else:
        num_episodes = 1
        control(sim, num_episodes, tot_cell_types, len(sim.layers[0]), classifier,
                writer=buddy,
                # add_technical_noise_function=add_technical_noise
                )
