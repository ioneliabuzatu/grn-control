import time

import jax
import jax.numpy as jnp
import torch

from jax.example_libraries import optimizers


from jax_simulator import Sim
from src.models.expert.classfier_cell_state import CellStateClassifier
from src.models.expert.classfier_cell_state import torch_to_jax
from src.techinical_noise import AddTechnicalNoise
from src.zoo_functions import is_debugger_active
from src.zoo_functions import plot_three_genes


class config:
    expert_checkpoint_filepath = "src/models/expert/checkpoints/expert_ds2.pth"


def cross_entropy(logprobs, targets):
    return -jnp.mean(jnp.sum(logprobs * targets, axis=1))


def control(env, num_episodes, num_cell_types, num_master_genes, visualise_samples_genes=False,
            use_technical_noise=False):
    start = time.time()
    classifier = CellStateClassifier(num_genes=100, num_cell_types=9).to("cpu")
    # loaded_checkpoint = torch.load(config().expert_checkpoint_filepath, map_location=lambda storage, loc: storage)
    # classifier.load_state_dict(loaded_checkpoint)
    classifier.eval()
    classifier = torch_to_jax(classifier)

    expert = classifier

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
            expr = AddTechnicalNoise(100, 9, 2, outlier_genes_noises, library_size_noises,
                                     dropout_noises).get_noisy_technical_concentration(expr.T)
        else:
            expr = jnp.concatenate(expr, axis=1).T

        print(expr.shape)
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

    print(f"Took {time.time() - start:.3f} secs.")


if __name__ == "__main__":
    sim = Sim(
        num_genes=100, num_cells_types=9,
        interactions_filepath="data/Interaction_cID_4.txt",
        regulators_filepath="data/Regs_cID_4.txt",
        simulation_num_steps=2,
    )
    sim.build()

    # if is_debugger_active():
    #     with jax.disable_jit():
    #         control(sim, 100, 9, num_master_genes=len(sim.layers[0]))
    # else:
    control(sim, 100, 9, len(sim.layers[0]), use_technical_noise=False)
