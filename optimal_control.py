import time

import experiment_buddy
import jax
import jax.numpy as jnp
import numpy as np
import torch
import wandb
from jax.example_libraries import optimizers

from jax_simulator import Sim
from src.models.expert.classfier_cell_state import CellStateClassifier
from src.models.expert.classfier_cell_state import torch_to_jax

target_class = 0


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
    mean_K = env.adjacency[env.adjacency > 0].mean()

    @jax.jit
    def loss_exp(actions):
        trajectory = env.run_one_rollout(actions * mean_K)
        trajectory = jnp.stack(tuple([trajectory[gene] for gene in range(env.num_genes)])).swapaxes(0, 1)

        last_state = trajectory[-1].T  # cell type is batch
        # mask = jnp.zeros_like(last_state)
        # mask = mask.at[:, env.layers[0]].set(1)  # Predict only based on the first layer
        # mask = mask.at[:, env.layers[1]].set(1)  # Predict only based on the first layer
        # last_state = last_state * mask

        output_classifier = expert(last_state / mean_K)
        gain = 2 * jnp.mean(output_classifier[:, target_class]) - jnp.mean(jnp.sum(output_classifier, axis=1), axis=0)
        # gain = jnp.mean(output_classifier[:, target_class])
        return gain, last_state

    def update(episode, opt_state_, check_expressions_for_convergence):
        actions = get_params(opt_state_)
        (gain, last_state), grad = jax.value_and_grad(loss_exp, has_aux=True)(actions)
        return opt_update(episode, -grad, opt_state_), check_expressions_for_convergence, gain, grad, last_state

    start = time.time()
    opt_init, opt_update, get_params = optimizers.adam(step_size=0.05)
    # jnp.ones(shape=(num_master_genes, num_cell_types))
    a0 = jnp.array(np.random.random(size=(num_master_genes, num_cell_types)))
    opt_state = opt_init(a0)

    check_expressions_for_convergence = []

    for episode in range(num_episodes):
        print("Episode#", episode)
        opt_state, check_expressions_for_convergence, gain, grad, last_state = update(episode, opt_state, check_expressions_for_convergence)  # opt_state are the actions
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
    # ds4_ground_truth_initial_dist = np.load("data/ds4_10k_each_type.npy")

    params = {'num_genes': '', 'NUM_SIM_CELLS': 10}
    experiment_buddy.register_defaults(params)
    buddy = experiment_buddy.deploy(host="")

    # dataset_dict = open_datasets_json(return_specific_key='DS4')
    # dataset = dataset_namedtuple(*dataset_dict.values())
    folder = "final_experiments(1)"
    dataset_dict = {
        "interactions": f"{folder}/interactions.txt",
        "regulators": f"{folder}/master_regulators.txt",
        "params_outliers_genes_noise": [0.011039100623008497, 1.4255511751527647, 2.35380330573968],
        "params_library_size_noise": [1.001506520357257, 1.7313202816171356],
        "params_dropout_noise": [4.833139049292777, 62.38254284061924],
        "tot_genes": 28,
        "tot_cell_types": 2,
    }

    tot_genes = dataset_dict["tot_genes"]
    tot_cell_types = dataset_dict["tot_cell_types"]
    interactions_filepath = dataset_dict["interactions"]  # f"data/interactions_random_graph_{tot_genes}_genes.txt"

    expert_checkpoint_filepath = f"{folder}/expert.pth"
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    classifier = CellStateClassifier(num_genes=tot_genes, num_cell_types=tot_cell_types).to("cpu")

    loaded_checkpoint = torch.load(expert_checkpoint_filepath, map_location=lambda storage, loc: storage)
    classifier.load_state_dict(loaded_checkpoint)

    classifier.eval()
    classifier = torch_to_jax(classifier)

    sim = Sim(
        num_genes=tot_genes, num_cells_types=tot_cell_types,
        simulation_num_steps=params['NUM_SIM_CELLS'],
        interactions_filepath=interactions_filepath, regulators_filepath=dataset_dict["regulators"], noise_amplitude=0.9
    )

    start = time.time()
    adjacency, graph, layers = sim.build()
    print(f"Compiling the graph took {round(time.time() - start)} secs.")

    num_episodes = 10_000
    # with jax.disable_jit():
    #     control(
    #         sim, num_episodes, tot_cell_types, len(sim.layers[0]), classifier,
    #         writer=buddy,
    #     )
    control(
        sim, num_episodes, tot_cell_types, len(sim.layers[0]), classifier,
        writer=buddy,
    )
