import time

from matplotlib import pyplot as plt

from src.zoo_functions import dataset_namedtuple, open_datasets_json
import numpy as np

import jax
import jax.numpy as jnp
import torch
import sys
from jax_simulator import Sim
from src.models.expert.classfier_cell_state import MiniCellStateClassifier as CellStateClassifier
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

target_class = 0


def cross_entropy(logprobs, target_to_steer):
    """
    targets = jnp.array([0] * output_classifier.shape[0])
    v0: -jnp.mean(jnp.sum(logprobs * targets, axis=1))
    v1: jnp.mean(jnp.sum(output_classifier * jnp.eye(3)[targets], axis=1))

    @logprobs: expert output or could be the softmax output
    @desired_target: the target class, can be int or the one-hot encoding is chosen to use softmax
    """
    # print("logprobs",logprobs.primal)
    print("argmax", logprobs.argmax(1))
    # logprobs2 = logprobs - jnp.max(logprobs, axis=1, keepdims=True)
    cs = 2 * jnp.mean(logprobs[:, target_to_steer]) - jnp.mean(logprobs)
    print("=========> logprobs", logprobs.primal[:, target_to_steer])
    probs = jax.nn.softmax(logprobs, axis=0)
    print("=========> probs", probs.primal[:, target_to_steer])
    return cs


def control(env, num_episodes, num_cell_types, num_master_genes, expert, visualise_samples_genes=False,
            writer=None, add_technical_noise_function=None):
    mean_K = env.adjacency[env.adjacency > 0].mean()

    @jax.jit
    def loss_exp(actions):
        trajectory = env.run_one_rollout(actions * mean_K)
        trajectory = jnp.stack(tuple([trajectory[gene] for gene in range(env.num_genes)])).swapaxes(0, 1)

        # if add_technical_noise_function is not None:
        #   all_expr_stack_by_type = add_technical_noise_function.get_noisy_technical_concentration(expression.T).T
        # else:
        #   all_expr_stack_by_type = jnp.vstack([expression[:, :, i] for i in range(expression.shape[2])])

        last_state = trajectory[-1].T  # cell type is batch
        output_classifier = expert(last_state / mean_K)

        # TODO: make the sum, max over the non target logits
        # TODO: afterwards, replace max with logsumexp to make it smooth
        gain = 2 * jnp.mean(output_classifier[:, target_class]) - jnp.mean(jnp.sum(output_classifier, axis=1), axis=0)
        return gain, last_state

    def update(episode, opt_state_, check_expressions_for_convergence):
        actions = get_params(opt_state_)
        # print("actions going in value_and_grad", actions)
        (gain, last_state), grad = jax.value_and_grad(loss_exp, has_aux=True)(actions)

        # print("grad before clipping", grad)
        grad = jnp.clip(grad, -1, 1)
        print("gain", gain)
        print(f"grad shape: {grad.shape} \n grad: {grad}")
        return opt_update(episode, -grad, opt_state_), check_expressions_for_convergence, gain, grad, last_state

    start = time.time()
    opt_init, opt_update, get_params = optimizers.adam(step_size=0.05)
    # opt_init, opt_update, get_params = optimizers.momentum(step_size=0.005, mass=0.1)
    # opt_init, opt_update, get_params = optimizers.adam(step_size=0.1)
    # opt_state = opt_init(jnp.ones(shape=(num_master_genes, num_cell_types)))
    a0 = jnp.array(np.random.random(size=(num_master_genes, num_cell_types)))
    opt_state = opt_init(a0)

    check_expressions_for_convergence = []

    for episode in range(num_episodes):
        print("Episode#", episode)
        opt_state, check_expressions_for_convergence, gain, grad, last_state = update(
            episode, opt_state, check_expressions_for_convergence)  # opt_state are the actions
        logits = expert(last_state)
        probs = jax.nn.softmax(logits, axis=1)

        target_class_mean_prob = jnp.mean(probs[:, target_class])
        offtarget_class_mean_prob = jnp.mean(probs[:, 1 - target_class])

        if episode % 1 == 0:
            # logits = last_state
            print("gain", gain, target_class_mean_prob, grad.mean())
            # print("ctypes", logits.argmax(axis=1))
            # print("ctypes-target:", logits[:, 0])
            # print("ctypes-other:", logits[:, 1])
            # print(f"grad: {grad}")
            writer.run.log({"metrics/gain": gain}, step=episode)
            writer.run.log({"metrics/grads": grad.mean()}, step=episode)

            writer.run.log({"logits/argmax(axis=1).mean()": logits.argmax(axis=1).mean()}, step=episode)

            writer.run.log({"logits/target_class.mean()": logits[:, target_class].mean()}, step=episode)
            writer.run.log({"logits/not_target_class.mean()": logits[:, 1 - target_class].mean()}, step=episode)

            writer.run.log({f"p/target": wandb.Histogram(probs[:, target_class])}, step=episode)
            writer.run.log({f"p/not_target": wandb.Histogram(probs[:, 1 - target_class])}, step=episode)
            writer.run.log({"p/target_class_prob": target_class_mean_prob}, step=episode)
            writer.run.log({"p/offtarget_class_prob": offtarget_class_mean_prob}, step=episode)

            actions = get_params(opt_state) * mean_K
            writer.run.log({f"actions/0": actions[:, 0]}, step=episode)
            writer.run.log({f"actions/1": actions[:, 1]}, step=episode)

    print(f"policy gradient simulation control took {round(time.time() - start)} secs.")


if __name__ == "__main__":
    import getpass
    from pathlib import Path
    whoami = getpass.getuser()
    home = str(Path.home())
    print("home:", home)
    if whoami == 'ionelia':
        repo_path = f"{home}/pycharm-projects/grn-control"
        expert_checkpoint = f"{repo_path}/data/GEO/GSE122662/graph-experiments/expert_28_genes_2_layer.pth"
        graph_interactions_filepath = f"{repo_path}/data/GEO/GSE122662/graph-experiments/toy_graph28nodes.txt"
        master_regulators_init = f"{repo_path}/data/GEO/GSE122662/graph-experiments/28_nodes_MRs.txt"
    elif whoami == 'manuel':
        expert_checkpoint = "final_experiments(1)"

    NUM_SIM_CELLS = 10
    experiment_buddy.register_defaults(locals())
    buddy = experiment_buddy.deploy(host="")

    # dataset_dict = open_datasets_json(return_specific_key='DS4')
    # dataset = dataset_namedtuple(*dataset_dict.values())


    dataset_dict = {
        "interactions": graph_interactions_filepath,
        "regulators": master_regulators_init,
        "params_outliers_genes_noise": [0.011175966309981848, 2.328873447557661, 0.5011137928428419],
        "params_library_size_noise": [9.961818165607404, 1.2905366314510822],
        "params_dropout_noise": [6.3136458044016655, 62.50611701257209],
        "tot_genes": 28,
        "tot_cell_types": 2
    }
    dataset = dataset_namedtuple(*dataset_dict.values())

    expert_checkpoint_filepath = f"{expert_checkpoint}"
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)

    classifier = CellStateClassifier(num_genes=dataset.tot_genes,
                                     num_cell_types=dataset.tot_cell_types).to("cpu")
    loaded_checkpoint = torch.load(expert_checkpoint_filepath, map_location=lambda storage, loc: storage)
    classifier.load_state_dict(loaded_checkpoint)
    classifier.eval()
    classifier = torch_to_jax(classifier, use_simple_model=True)

    sim = Sim(
        num_genes=dataset.tot_genes, num_cells_types=dataset.tot_cell_types,
        simulation_num_steps=NUM_SIM_CELLS,
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
