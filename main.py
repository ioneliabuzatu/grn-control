import time

import experiment_buddy
import jax
import jax.numpy as jnp
import numpy as np
import seaborn as sns
import wandb
from matplotlib import pyplot as plt

from jax_simulator import Sim
from src.models.expert.classfier_cell_state import CellStateClassifier
from src.models.expert.classfier_cell_state import torch_to_jax
from src.techinical_noise import AddTechnicalNoiseJax
from src.zoo_functions import dataset_namedtuple, open_datasets_json
from src.zoo_functions import is_debugger_active
from src.zoo_functions import plot_three_genes


def cross_entropy(logprobs, targets):
    return -jnp.mean(jnp.sum(logprobs * targets, axis=1))


def control(env, num_episodes, num_cell_types, num_master_genes, expert, visualise_samples_genes=False,
            writer=None, add_technical_noise_function=None):
    start = time.time()
    x0 = None
    xt = None

    def loss_fn(actions):
        nonlocal x0, xt

        expr = env.run_one_rollout(actions)
        expr = jnp.stack(tuple([expr[gene] for gene in range(env.num_genes)])).swapaxes(0, 1)
        if x0 is None:
            x0 = np.array(expr.primal)
        xt = np.array(expr.primal)

        if visualise_samples_genes:
            plot_three_genes(expr.T[0, 44], expr.T[0, 1], expr.T[0, 99], hlines=None, title="expression")

        if add_technical_noise_function is not None:
            expr = add_technical_noise_function.get_noisy_technical_concentration(expr.T).T
            return expr.mean()
        else:
            expr = jnp.concatenate(expr, axis=1).T

        print(expr.shape)
        output_classifier = expert(expr)
        targets = jnp.array([8] * output_classifier.shape[0])
        loss = -jnp.mean(jnp.sum(jax.nn.log_softmax(output_classifier).T * targets, axis=1))
        return loss

    actions = jnp.ones(shape=(num_master_genes, num_cell_types))

    for episode in range(num_episodes):
        print("Episode#", episode)
        loss, grad = jax.value_and_grad(loss_fn)(actions)
        grad = jnp.clip(grad, -1, 1)
        print("loss", loss)
        print(f"grad shape: {grad.shape} \n grad: {grad}")
        actions += 0.01 * -grad

        writer.add_scalar(f"loss", loss, episode)
        writer.run.log({"grads": wandb.Image(sns.heatmap(grad, linewidth=0.5))}, step=episode)
        plt.close()
        writer.run.log({"actions": wandb.Image(sns.heatmap(actions, linewidth=0.5))}, step=episode)
        plt.close()
        print(f"episode#{episode} took {time.time() - start:.3f} secs.")

        desired_concentration = x0[:, :, :1]
        print("Distance from initial state:", jnp.linalg.norm(x0 - xt))
        print("Distance from cell_type 0:", jnp.linalg.norm(np.tile(desired_concentration, (1, 1, num_cell_types)) - xt))

    print(f"Took {time.time() - start:.3f} secs.")


if __name__ == "__main__":
    params = {'num_genes': 100}
    experiment_buddy.register_defaults(params)
    buddy = experiment_buddy.deploy(host="", disabled=True)

    dataset_dict = open_datasets_json(return_specific_dataset='Dummy')
    dataset = dataset_namedtuple(*dataset_dict.values())

    expert_checkpoint_filepath = "src/models/expert/checkpoints/classifier_ds4.pth"
    classifier = CellStateClassifier(num_genes=dataset.tot_genes, num_cell_types=dataset.tot_cell_types).to("cpu")
    # loaded_checkpoint = torch.load(expert_checkpoint_filepath, map_location=lambda storage, loc: storage)
    # classifier.load_state_dict(loaded_checkpoint)
    classifier.eval()
    classifier = torch_to_jax(classifier)

    sim = Sim(
        num_genes=dataset.tot_genes, num_cells_types=dataset.tot_cell_types, simulation_num_steps=2,
        interactions_filepath=dataset.interactions, regulators_filepath=dataset.regulators,
    )
    sim.build()

    add_technical_noise = AddTechnicalNoiseJax(
        dataset.tot_genes, dataset.tot_cell_types, 2,
        dataset.params_outliers_genes_noise, dataset.params_library_size_noise, dataset.params_dropout_noise
    )

    if is_debugger_active():
        with jax.disable_jit():
            control(sim, 100, 9, num_master_genes=len(sim.layers[0]), expert=classifier, writer=buddy, )
    else:
        control(sim, 51, dataset.tot_cell_types, len(sim.layers[0]), expert=classifier, writer=buddy,
                # add_technical_noise_function=add_technical_noise
                )
