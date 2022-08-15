import time

import experiment_buddy
import jax
import jax.numpy as jnp
import numpy as np
import seaborn as sns
import torch
import wandb
from jax.example_libraries import optimizers
from matplotlib import pyplot as plt

from jax_simulator import Sim
from src.models.expert.classfier_cell_state import MiniCellStateClassifier as CellStateClassifier
from src.models.expert.classfier_cell_state import torch_to_jax
from src.zoo_functions import dataset_namedtuple
from src.zoo_functions import is_debugger_active

# gene_names_28 = { 0: 'Sox2', 1: 'Obox6', 2: 'Klf4', 3: 'Esrrb',
# 4: 'Hmx1', 5: 'Myc', 6: 'Pou5f1', 7: 'Elf2', 8: 'Fmnl2', 9: 'Nfkb1', 10: 'Sirt7', 11: 'Rfx3', 12: 'Nr1i3',
# 13: 'Rfx5', 14: 'Hmg20a', 15: 'Ppp1r13b', 16: 'Polr3e', 17: 'Gmeb2', 18: 'Gmeb1', 19: 'Zfp282', 20: 'Ep300',
# 21: 'B930041F14Rik', 22: 'Zfp2', 23: 'Hdac10', 24: 'Asb6', 25: 'Zfp37', 26: 'Pou2f3', 27: 'Gdf9'}

gene_names = {
    0: 'Cdkn2a', 1: 'Xist', 2: 'Sox2', 3: 'Nanog', 4: 'Tdgf1', 5: 'Zfp42', 6: 'Fmr1nb', 7: 'Ooep', 8: 'Tcl1',
    9: 'Obox6', 10: 'Klf4', 11: 'Esrrb', 12: 'Dppa4', 13: 'Myc', 14: 'Lncenc1', 15: 'Sohlh2', 16: 'Pou5f1', 17: 'Gdf9'
}

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


def control(env, num_episodes, num_master_genes, expert, visualise_samples_genes=False,
            writer=None, add_technical_noise_function=None):
    ### mean_K = env.adjacency[env.adjacency > 0].mean()
    ### print(f"mean Ks: {mean_K:.3f}")

    actions_names = [gene_names[gene] for gene in sim.layers[0]]
    heatmap_kwargs = {'linewidth': 5,
                      'cbar_kws': {"shrink": .7},
                      'square': True, 'cmap': 'viridis',
                      }

    @jax.jit
    def loss_exp(actions):
        trajectory = env.run_one_rollout(actions, target_idx=target_class, context_bandits=True)
        trajectory = jnp.stack(tuple([trajectory[gene] for gene in range(env.num_genes)])).swapaxes(0, 1)

        # if add_technical_noise_function is not None:
        #   all_expr_stack_by_type = add_technical_noise_function.get_noisy_technical_concentration(expression.T).T
        # else:
        #   all_expr_stack_by_type = jnp.vstack([expression[:, :, i] for i in range(expression.shape[2])])

        last_state = trajectory[-1].T  # cell type is batch
        output_classifier = expert(last_state)

        gain_target_class = jnp.mean(output_classifier[:, target_class])
        penalty_off_target_class = jnp.mean(output_classifier[:, 1 - target_class], axis=0)
        gain = 2 * gain_target_class - penalty_off_target_class
        return gain, last_state

    def update(episode, opt_state_, check_expressions_for_convergence):
        actions = get_params(opt_state_)
        # print("actions going in value_and_grad", actions)
        (gain, last_state), grad = jax.value_and_grad(loss_exp, has_aux=True)(actions)

        # print("grad before clipping", grad)
        grad = jnp.clip(grad, -1, 1)
        print("gain", gain)
        print(f"grad shape: {grad.shape} \n grad: {grad}")
        print("actions:", actions)
        return opt_update(episode, -grad, opt_state_), check_expressions_for_convergence, gain, grad, last_state

    start = time.time()
    opt_init, opt_update, get_params = optimizers.adam(step_size=0.05)
    # opt_init, opt_update, get_params = optimizers.momentum(step_size=0.005, mass=0.1)
    # opt_init, opt_update, get_params = optimizers.adam(step_size=0.1)
    a0 = jnp.array(np.random.random(size=num_master_genes))  # MEMO remember initially was also had num_cell_types)))
    opt_state = opt_init(a0)

    check_expressions_for_convergence = []

    for episode in range(num_episodes):
        print("Episode#", episode)
        opt_state, check_expressions_for_convergence, gain, grad, last_state = update(
            episode, opt_state, check_expressions_for_convergence
        )  # opt_state are the actions

        logits = expert(last_state)
        probs = jax.nn.softmax(logits, axis=1)

        target_class_mean_prob = jnp.mean(probs[:, target_class])
        offtarget_class_mean_prob = jnp.mean(probs[:, 1 - target_class])

        if episode % 1 == 0:
            actions = get_params(opt_state)
            print(f"episode:{episode}|gain:{gain}|target_class_mean_prob|grad.mean()")

            plt.figure(figsize=(3.5, 10))
            heatmap_grads = (grad - grad.mean(0)) / grad.std(0)
            heatmap_grads = sns.heatmap(heatmap_grads.reshape(1, *heatmap_grads.shape).T, **heatmap_kwargs,
                                        xticklabels=['D0' if target_class == 0 else 'iPSC'],
                                        yticklabels=actions_names
                                        )
            writer.run.log({"heatmaps/sensitivity_analysis": wandb.Image(heatmap_grads)}, step=episode)
            plt.close()

            writer.run.log({"metrics/gain": gain}, step=episode)
            writer.run.log({"metrics/grads.mean()": grad.mean()}, step=episode)

            writer.run.log({"logits/argmax(axis=1).mean()": logits.argmax(axis=1).mean()}, step=episode)
            writer.run.log({"logits/target_class.mean()": logits[:, target_class].mean()}, step=episode)
            writer.run.log({"logits/not_target_class.mean()": logits[:, 1 - target_class].mean()}, step=episode)

            writer.run.log({f"p/target": wandb.Histogram(probs[:, target_class])}, step=episode)
            writer.run.log({f"p/not_target": wandb.Histogram(probs[:, 1 - target_class])}, step=episode)
            writer.run.log({"p/target_class_prob": target_class_mean_prob}, step=episode)
            writer.run.log({"p/off_target_class_prob": offtarget_class_mean_prob}, step=episode)

            plt.figure(figsize=(3.5, 10))
            heatmap_actions = (actions - actions.mean(0)) / actions.std(0)
            heatmap_actions_reshape = sns.heatmap(heatmap_actions.reshape(1, *heatmap_actions.shape).T,
                                                  **heatmap_kwargs,
                                                  xticklabels=['D0' if target_class == 0 else 'iPSC'],
                                                  )
            writer.run.log({"heatmaps/actions": wandb.Image(heatmap_actions_reshape)}, step=episode)
            plt.close()

            heatmap_gene_expr = sns.heatmap(
                last_state.T, xticklabels=['D0', 'iPSC'], yticklabels=list(gene_names.values()), **heatmap_kwargs)
            writer.run.log({"heatmaps/gene_expression": wandb.Image(heatmap_gene_expr)}, step=episode)
            plt.close()

            for idx, _action in enumerate(actions):
                writer.run.log({f"actions/{actions_names[idx]}": actions[idx]}, step=episode)

    print(f"policy gradient simulation control took {round(time.time() - start)} secs.")


if __name__ == "__main__":
    import getpass
    from pathlib import Path

    whoami = getpass.getuser()
    home = str(Path.home())
    print("home:", home)
    if whoami == "ionelia":
        repo_path = f"{home}/projects/grn-control"
        files_path = "/data/GEO/GSE122662/graph-experiments"
        # expert_checkpoint = f"{repo_path}/data/GEO/GSE122662/graph-experiments/expert_28_genes_2_layer.pth"
        # graph_interactions_filepath = f"{repo_path}/data/GEO/GSE122662/graph-experiments/toy_graph28nodes.txt"
        # master_regulators_init = f"{repo_path}/data/GEO/GSE122662/graph-experiments/28_nodes_MRs.txt"
        expert_checkpoint = f"{repo_path}/src/models/expert/checkpoints/expert_thesis_trained_on_real.pth"
        graph_interactions_filepath = f"{repo_path}/{files_path}/interactions_18_genes_thesis.txt"
        master_regulators_init = f"{repo_path}/{files_path}/mrs_18_genes.txt"
    elif whoami == "ionelia.buzatu":
        glob_source = "/network/projects/_groups/grn_control/graphD0D18genes#18"
        expert_checkpoint=f"{glob_source}/expert_thesis_trained_on_real.pth"
        graph_interactions_filepath = f"{glob_source}/interactions_18_genes_thesis.txt"
        master_regulators_init = f"{glob_source}/mrs_18_genes.txt"

    NUM_SIM_CELLS = 1
    experiment_buddy.register_defaults(locals())
    writer = experiment_buddy.deploy(
        host="mila", wandb_run_name=f"###steer{target_class}|6u|18G|expert_real|OC|",
        disabled=False, wandb_kwargs={'entity': 'control-grn', 'project': 'OC'}
    )

    # dataset_dict = open_datasets_json(return_specific_key='DS4')
    # dataset = dataset_namedtuple(*dataset_dict.values())

    dataset_dict = {
        "interactions": graph_interactions_filepath,
        "regulators": master_regulators_init,
        "params_outliers_genes_noise": [0.011175966309981848, 2.328873447557661, 0.5011137928428419],
        "params_library_size_noise": [9.961818165607404, 1.2905366314510822],
        "params_dropout_noise": [6.3136458044016655, 62.50611701257209],
        "tot_genes": 18,
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
        interactions_filepath=dataset.interactions, regulators_filepath=dataset.regulators, noise_amplitude=0.7
    )

    start = time.time()
    adjacency, graph, layers = sim.build()
    print(f"Compiling the graph took {round(time.time() - start)} secs.")
    print(f"actions names: {[gene_names[mr] for mr in layers[0]]}")

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
        sim.simulation_num_steps = 5
        with jax.disable_jit():
            control(
                sim, 5, len(sim.layers[0]), classifier, writer=writer,
                # add_technical_noise_function=add_technical_noise
            )
    else:
        num_episodes = 700
        control(
            sim, num_episodes, len(sim.layers[0]), classifier,
            writer=writer,
            # add_technical_noise_function=add_technical_noise
        )