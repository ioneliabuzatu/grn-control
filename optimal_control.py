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

np.set_printoptions(suppress=True)


def cross_entropy(logprobs, target_to_steer):
    """
    targets = jnp.array([0] * output_classifier.shape[0])
    v0: -jnp.mean(jnp.sum(logprobs * targets, axis=1))
    v1: jnp.mean(jnp.sum(output_classifier * jnp.eye(3)[targets], axis=1))

    @logprobs: expert output or could be the softmax output
    @desired_target: the target class, can be int or the one-hot encoding is chosen to use softmax
    """
    # print("logprobs",logprobs.primal)
    print("argmax",logprobs.argmax(1))
    # logprobs2 = logprobs - jnp.max(logprobs, axis=1, keepdims=True)
    cs = 2 * jnp.mean(logprobs[:, target_to_steer]) - jnp.mean(logprobs)
    print("=========> logprobs", logprobs.primal[:, target_to_steer])
    probs = jax.nn.softmax(logprobs, axis=0)
    print("=========> probs", probs.primal[:, target_to_steer])
    return cs


def control(env, num_episodes, num_cell_types, num_master_genes, expert, visualise_samples_genes=False,
            writer=None, add_technical_noise_function=None):
    # @jax.jit
    def loss_fn(actions):
        expression = env.run_one_rollout(actions)
        expression = jnp.stack(tuple([expression[gene] for gene in range(env.num_genes)])).swapaxes(0, 1)
        print("gene 70 espression", float(expression[:, :, 0][:, 70].mean().primal), float(expression[:, :, 1][:,
                                                                                           70].mean().primal))

        if visualise_samples_genes:
            plot_three_genes(expression.T[0, 44], expression.T[0, 1], expression.T[0, 99], hlines=None,
                             title="expression")

        if add_technical_noise_function is not None:
            all_expr_stack_by_type = add_technical_noise_function.get_noisy_technical_concentration(expression.T).T
        else:
            all_expr_stack_by_type = jnp.vstack([expression[:, :, i] for i in range(expression.shape[2])])

        print(all_expr_stack_by_type.shape)
        print(jnp.isnan(all_expr_stack_by_type).any())
        output_classifier = expert(all_expr_stack_by_type)
        loss = cross_entropy(output_classifier, 1)
        return loss, expression

    def update(episode, opt_state_, check_expressions_for_convergence):
        actions = get_params(opt_state_)
        # print("actions going in value_and_grad", actions)
        # actions = jnp.ones(shape=(num_master_genes, num_cell_types))
        (loss, expression), grad = jax.value_and_grad(loss_fn, has_aux=True)(actions)
        # loss, grad = jax.value_and_grad(loss_fn)(actions)

        # print("grad before clipping", grad)
        grad = jnp.clip(grad, -1, 1)
        print("loss", loss)
        # print(f"grad shape: {grad.shape} \n grad: {grad}")

        # writer.run.log({"loss": loss}, step=episode)
        # writer.run.log({"grads": wandb.Image(sns.heatmap(grad, linewidth=0.5))}, step=episode)
        # plt.close()
        # writer.run.log({"actions": wandb.Image(sns.heatmap(actions, linewidth=0.5))}, step=episode)
        # plt.close()
        # fig = plt.figure(figsize=(10, 7))
        # plt.plot(jnp.mean(jnp.array(expression), axis=0))
        # writer.run.log({"control/mean": wandb.Image(fig)}, step=episode)
        # plt.close()

        # check_expressions_for_convergence.append(expression)
        # if len(check_expressions_for_convergence) == 3:
        #     expression = jnp.array(check_expressions_for_convergence)
        #     last_x_points = expression[:, -20]
        #     mean_x_points = jnp.mean(last_x_points, axis=0)
        #     var_mean_x_points = jnp.var(last_x_points - mean_x_points, axis=0)
        #     print((var_mean_x_points < 0.1).sum())
        #     check_expressions_for_convergence = []

        return opt_update(episode, grad, opt_state_), check_expressions_for_convergence

    start = time.time()

    # opt_init, opt_update, get_params = optimizers.adam(step_size=0.005)
    opt_init, opt_update, get_params = optimizers.momentum(step_size=0.005, mass=0.1)
    # opt_init, opt_update, get_params = optimizers.adam(step_size=0.1)
    # opt_state = opt_init(jnp.ones(shape=(num_master_genes, num_cell_types)))
    opt_state = opt_init(jnp.array([
        [0.6181008, 0.5028295214],
    ]))

    check_expressions_for_convergence = []

    for episode in range(num_episodes):
        opt_state, check_expressions_for_convergence = update(episode, opt_state,
                                                              check_expressions_for_convergence)  # opt_state are the actions
        print(f"Episode# {episode} took {time.time() - start:.1f} secs.")
        print(sim.layers[0])
        print("-------------------------------------------------------------------")

    print(f"policy gradient simulation control took {round(time.time() - start)} secs.")


if __name__ == "__main__":
    # ds4_ground_truth_initial_dist = np.load("data/ds4_10k_each_type.npy")

    params = {'num_genes': '', 'NUM_SIM_CELLS': 11}
    experiment_buddy.register_defaults(params)
    buddy = experiment_buddy.deploy(host="", disabled=True)

    # dataset_dict = open_datasets_json(return_specific_key='DS4')
    # dataset = dataset_namedtuple(*dataset_dict.values())

    dataset_dict = {
        "interactions": 'data/GEO/GSE122662/final-graph/107G-graph/106G_interactions.txt',
        "regulators": "data/GEO/GSE122662/final-graph/107G-graph/master_regulators.txt",
        "params_outliers_genes_noise": [0.011175966309981848, 2.328873447557661, 0.5011137928428419],
        "params_library_size_noise": [9.961818165607404, 1.2905366314510822],
        "params_dropout_noise": [6.3136458044016655, 62.50611701257209],
        "tot_genes": 106,
        "tot_cell_types": 2,
    }

    dataset = dataset_namedtuple(*dataset_dict.values())

    expert_checkpoint_filepath = "src/models/expert/checkpoints/expert_106G_onelayer_or.pth"
    classifier = CellStateClassifier(num_genes=dataset.tot_genes,
                                     num_cell_types=dataset.tot_cell_types).to("cpu")
    loaded_checkpoint = torch.load(expert_checkpoint_filepath, map_location=lambda storage, loc: storage)
    classifier.load_state_dict(loaded_checkpoint)
    classifier.eval()
    classifier = torch_to_jax(classifier)

    sim = Sim(
        num_genes=dataset.tot_genes, num_cells_types=dataset.tot_cell_types,
        simulation_num_steps=params['NUM_SIM_CELLS'],
        interactions_filepath=dataset.interactions, regulators_filepath=dataset.regulators, noise_amplitude=0.9
    )

    start = time.time()
    adjacency, graph, layers = sim.build()
    print(f"Compiling the graph took {round(time.time() - start)} secs.")

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
        sim.simulation_num_steps = 1
        with jax.disable_jit():
            control(sim, 5, dataset.tot_cell_types, len(sim.layers[0]), classifier,
                    writer=buddy,
                    # add_technical_noise_function=add_technical_noise
                    )
    else:
        num_episodes = 999
        sim.simulation_num_steps = 1
        control(sim, num_episodes, dataset.tot_cell_types, len(sim.layers[0]), classifier,
                writer=buddy,
                # add_technical_noise_function=add_technical_noise
                )
