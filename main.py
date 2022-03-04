import time
import numpy as np

import jax
import jax.numpy as jnp
import torch

from jax.example_libraries import optimizers

from jax_simulator import Sim
from src.models.expert.classfier_cell_state import CellStateClassifier
from src.models.expert.classfier_cell_state import torch_to_jax
from src.techinical_noise import AddTechnicalNoiseJax
from src.zoo_functions import is_debugger_active
from src.zoo_functions import plot_three_genes
import experiment_buddy
import wandb
import seaborn as sns
print(f"Jax device/s: {jax.devices()}")


def cross_entropy(logprobs, targets):
    return -jnp.mean(jnp.sum(logprobs * targets, axis=1))


def control(env, num_episodes, num_cell_types, num_master_genes, tot_genes, expert, visualise_samples_genes=False,
            use_technical_noise=False, use_buddy=False):
    start = time.time()
    if use_buddy:
        writer = experiment_buddy.deploy(host="")

    @jax.jit
    def loss_fn(actions):
        expr = env.run_one_rollout(actions)
        expr = jnp.stack(tuple([expr[gene] for gene in range(env.num_genes)])).swapaxes(0, 1)

        if visualise_samples_genes:
            plot_three_genes(expr.T[0, 44], expr.T[0, 1], expr.T[0, 99], hlines=None, title="expression")

        if use_technical_noise:
            outlier_genes_noises = (0.01, 0.8, 1)
            library_size_noises = (6, 0.4)
            dropout_noises = (12, 80)

            expr = AddTechnicalNoiseJax(tot_genes, num_cell_types, 2, outlier_genes_noises, library_size_noises,
                                        dropout_noises).get_noisy_technical_concentration(expr.T)
        else:
            expr = jnp.concatenate(expr, axis=1).T

        print(expr.shape)
        assert expr.shape[1] == tot_genes
        output_classifier = expert(expr)
        targets = jnp.array([8] * output_classifier.shape[0])
        loss = -jnp.mean(jnp.sum(jax.nn.log_softmax(output_classifier).T * targets, axis=1))
        return loss

    actions = jnp.ones(shape=(num_master_genes, num_cell_types))
    optimizer = optimizers.adam(step_size=0.01)

    for episode in range(num_episodes):
        print("Episode#", episode)
        loss, grad = jax.value_and_grad(loss_fn)(actions)
        grad = jnp.clip(grad, -1, 1)
        print("loss", loss)
        print(f"grad shape: {grad.shape}")
        actions += 0.001 * -grad

        if use_buddy:
            writer.add_scalar(f"loss", loss, episode)
            writer.run.log({"grads": wandb.Image(sns.heatmap(grad.T, linewidth=0.5, ))}, step=episode)
            writer.run.log({"actions": wandb.Image(sns.heatmap(actions))}, step=episode)

    print(f"Took {time.time() - start:.3f} secs.")


if __name__ == "__main__":
    params = {'tot_genes': 400, 'use_buddy': False}
    if params['use_buddy']:
        experiment_buddy.register_defaults(params)

    expert_checkpoint_filepath = "src/models/expert/checkpoints/expert_ds2.pth"
    classifier = CellStateClassifier(num_genes=params["tot_genes"], num_cell_types=9).to("cpu")
    loaded_checkpoint = torch.load(expert_checkpoint_filepath, map_location=lambda storage, loc: storage)
    # classifier.load_state_dict(loaded_checkpoint)
    # classifier.eval()
    classifier = torch_to_jax(classifier)

    sim = Sim(num_genes=params["tot_genes"], num_cells_types=9,
              interactions_filepath="SERGIO/data_sets/De-noised_400G_9T_300cPerT_5_DS2/Interaction_cID_5.txt",
              regulators_filepath="SERGIO/data_sets/De-noised_400G_9T_300cPerT_5_DS2/Regs_cID_5.txt",
              simulation_num_steps=2,
              )
    sim.build()

    if is_debugger_active():
        with jax.disable_jit():
            control(sim, 100, 9, num_master_genes=len(sim.layers[0]))
    else:
        control(sim, 2, 9, len(sim.layers[0]), params["tot_genes"], classifier, use_technical_noise=False,
                use_buddy=False)
