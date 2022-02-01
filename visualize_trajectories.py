import experiment_buddy
import numpy as np

import jax_simulator
import numpy_simulator

params = {'num_genes': 100, 'num_cells_types': 9, 'num_cells_to_simulate': 5}

if __name__ == '__main__':
    main = False
    experiment_buddy.register_defaults(params)
    writer = experiment_buddy.deploy()

    simulator = jax_simulator.Sim(**params)
    # simulator = numpy_simulator.Sim(**params)
    simulator.interactions_filename = 'data/Interaction_cID_4.txt'
    simulator.regulators_filename = 'data/Regs_cID_4.txt'

    def plot(trajectory):
        for t, x_t in enumerate(trajectory):
            for gene_idx, g_t in enumerate(x_t):
                for cell_idx, c_t in enumerate(g_t):
                    writer.add_scalar(f"gene{gene_idx}/type{cell_idx}", c_t, t)


    if main:
        simulator.run()
        trajectory = simulator.x
        plot(trajectory)
    else:
        trajectory = simulator.run()
        keys, values = zip(*trajectory.items())
        num_cells_types, time_steps, num_genes = *values[0].shape, len(keys)
        concentrations = np.zeros(shape=(time_steps, num_genes, num_cells_types))

        for gene_idx, x_t in enumerate(values):
            x_t = x_t.T
            gene = keys[gene_idx]
            concentrations[:, gene, :] = x_t

        plot(concentrations)