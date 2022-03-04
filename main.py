import time
import numpy as np

import jax
import jax.numpy as jnp
import torch

from jax_simulator import Sim
from src.models.expert.classfier_cell_state import CellStateClassifier
from src.models.expert.classfier_cell_state import torch_to_jax
from src.techinical_noise import AddTechnicalNoiseJax
from src.zoo_functions import is_debugger_active
from src.zoo_functions import plot_three_genes
import experiment_buddy
import wandb
import seaborn as sns


def cross_entropy(logprobs, targets):
    return -jnp.mean(jnp.sum(logprobs * targets, axis=1))


def control(env, num_episodes, num_cell_types, num_master_genes, expert, visualise_samples_genes=False,
            use_technical_noise=False, writer=None):
    start = time.time()

    def loss_fn(actions):
        expr = env.run_one_rollout(actions)
        expr = jnp.stack(tuple([expr[gene] for gene in range(env.num_genes)])).swapaxes(0, 1)

        if visualise_samples_genes:
            plot_three_genes(expr.T[0, 44], expr.T[0, 1], expr.T[0, 99], hlines=None, title="expression")

        if use_technical_noise:
            outlier_genes_noises = (0.01, 0.8, 1)
            library_size_noises = (6, 0.4)
            dropout_noises = (12, 80)
            expr = AddTechnicalNoiseJax(400, 9, 2, outlier_genes_noises, library_size_noises,
                                     dropout_noises).get_noisy_technical_concentration(expr.T)
        else:
            expr = jnp.concatenate(expr, axis=1).T

        print(expr.shape)
        output_classifier = expert(expr.T)
        targets = jnp.array([8] * output_classifier.shape[0])
        loss = -jnp.mean(jnp.sum(jax.nn.log_softmax(output_classifier).T * targets, axis=1))
        return loss

    actions = jnp.ones(shape=(num_master_genes, num_cell_types))

    for episode in range(num_episodes):
        print("Episode#", episode)
        loss, grad = jax.value_and_grad(loss_fn)(actions)
        grad = jnp.clip(grad, -1, 1)
        print("loss", loss)
        print(f"grad shape: {grad.shape}")
        actions += 0.001 * -grad

        writer.add_scalar(f"loss", loss, episode)
        # writer.run.log({"gradients cell condition 0": wandb.Histogram(np_histogram=np.histogram(grad[0]))})
        # writer.run.log({"gradients cell condition 1": wandb.Histogram(np_histogram=np.histogram(grad[1]))})
        writer.run.log({"grads": wandb.Image(
            sns.heatmap(grad.T,
                        linewidth=0.5,
        #                 xticklabels=config.disease_gene_names
                        ))},
                        step=episode
        )

        writer.run.log({"actions": wandb.Image(sns.heatmap(actions))}, step=episode) # , linewidth=0.5,
                        # xticklabels=config.disease_gene_names


    print(f"Took {time.time() - start:.3f} secs.")


if __name__ == "__main__":
    params = {'num_genes': 400}
    experiment_buddy.register_defaults(params)
    buddy = experiment_buddy.deploy(host="")

    expert_checkpoint_filepath = "src/models/expert/checkpoints/expert_ds2.pth"

    classifier = CellStateClassifier(num_genes=400, num_cell_types=9).to("cpu")
    loaded_checkpoint = torch.load(expert_checkpoint_filepath, map_location=lambda storage, loc: storage)
    classifier.load_state_dict(loaded_checkpoint)
    classifier.eval()
    classifier = torch_to_jax(classifier)

    sim = Sim(num_genes=400, num_cells_types=9,
              interactions_filepath="SERGIO/data_sets/De-noised_400G_9T_300cPerT_5_DS2/Interaction_cID_5.txt",
              regulators_filepath="SERGIO/data_sets/De-noised_400G_9T_300cPerT_5_DS2/Regs_cID_5.txt",
              simulation_num_steps=2,
              )
    sim.build()

    if is_debugger_active():
        with jax.disable_jit():
            control(sim, 100, 9, num_master_genes=len(sim.layers[0]))
    else:
        control(sim, 100, 9, len(sim.layers[0]), classifier, use_technical_noise=True, writer=buddy)
