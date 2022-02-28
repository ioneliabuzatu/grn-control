import time
from jax_simulator import Sim
import jax.numpy as jnp
import jax
from src.zoo_functions import is_debugger_active
# from src.check_for_convergence import check_for_convergence
from src.zoo_functions import plot_three_genes


def control(env, num_episodes, num_cell_types,  num_master_genes):
    start = time.time()
    classifier = None  # TODO used in the cost function, is data-driven

    def loss_fn(actions, expert=classifier):
        expression_shape_trajectories_genes_cells = env.run_one_rollout(actions)
        expression_shape_trajectories_genes_cells = jnp.stack(tuple([expression_shape_trajectories_genes_cells[gene]
                                                                    for gene in range(env.num_genes)])).swapaxes(0, 1)

        expression_shape_cells_genes = jnp.concatenate(expression_shape_trajectories_genes_cells, axis=1).T

        plot_three_genes(expression_shape_trajectories_genes_cells[0, 44],
                         expression_shape_trajectories_genes_cells[0, 1],
                         expression_shape_trajectories_genes_cells[0, 99],
                         hlines=None)
        return -expression_shape_cells_genes.mean()

    actions = jnp.ones(shape=(num_master_genes, num_cell_types))

    for episode in range(num_episodes):
        print("##################################################################################", episode)
        loss, grad = jax.value_and_grad(loss_fn)(actions)
        grad = jnp.clip(grad, -1, 1)
        print("loss", loss)
        print(f"grad shape: {grad.shape} \n grad: {grad}")
        actions += 0.001 * -grad

    print(f"Took {time.time() - start:.3f} secs.")


if __name__ == "__main__":
    sim = Sim(num_genes=100, num_cells_types=9,
              interactions_filepath="data/Interaction_cID_4.txt",
              regulators_filepath="data/Regs_cID_4.txt",
              simulation_num_steps=3,
              )
    sim.build()

    if is_debugger_active():
        with jax.disable_jit():
            control(sim, 100, 9, num_master_genes=len(sim.layers[0]))
    else:
        control(sim, 2, 9, len(sim.layers[0]))
