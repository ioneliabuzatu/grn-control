import experiment_buddy
import numpy as np

import jax_simulator
import numpy_simulator

params = {'num_genes': 100, 'num_cells_types': 9, 'num_cells_to_simulate': 50,
          'interactions': 'data/Interaction_cID_4.txt',
          'regulators': 'data/Regs_cID_4.txt',
          'noise_amplitude': 0.1,
          }

if __name__ == '__main__':
    experiment_buddy.register_defaults(params)
    writer = experiment_buddy.deploy()

    simulator = jax_simulator.Sim(**params)
    # simulator = numpy_simulator.Sim(**params, deterministic=True)

    def plot(trajectory):
        for t, x_t in enumerate(trajectory):
            for gene_idx, g_t in enumerate(x_t):
                if gene_idx in [44,1,99]:
                    for cell_idx, c_t in enumerate(g_t):
                        writer.add_scalar(f"gene{gene_idx}/type{cell_idx}", c_t, t)

    simulator.run()
    trajectory = simulator.x
    plot(trajectory)
