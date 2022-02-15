import time
from jax_simulator import Sim
import jax.numpy as jnp
import jax
from src.zoo_functions import is_debugger_active


def control(env, num_episodes, num_states, num_genes):
    start = time.time()
    classifier = None  # TODO used in the cost function, is data-driven
    actions = jnp.zeros(shape=(num_states, num_genes)) + 0.1

    def loss_fn(actions):
        expression = env.run(actions) 
        return -1.0  # TODO: is loss just a scalar here?

    for _ in range(10):
        loss, grad = jax.value_and_grad(loss_fn)(actions)
        print("loss", loss)
        print(f"grad shape: {grad.shape} \n grad: {grad}")
        actions += 0.001 * -grad

    print(f"Took {time.time() - start:.3f} secs.")


if __name__ == "__main__":
    sim = Sim(num_genes=100, num_cells_types=9,
              interactions_filepath="data/Interaction_cID_4.txt",
              regulators_filepath="data/Regs_cID_4.txt",
              simulation_num_steps=100,
              )

    if is_debugger_active():
        with jax.disable_jit():
            control(sim, 100, 9, 100)
    else:
        control(sim, 100, 9, 100)
