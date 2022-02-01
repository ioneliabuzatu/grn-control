import experiment_buddy

import numpy_simulator

params = {'num_genes': 100, 'num_cells_types': 9, 'num_cells_to_simulate': 5}

if __name__ == '__main__':
    experiment_buddy.register_defaults(params)
    writer = experiment_buddy.deploy()

    simulator = numpy_simulator.Sim(**params)
    simulator.interactions_filename = 'data/Interaction_cID_4.txt'
    simulator.regulators_filename = 'data/Regs_cID_4.txt'
    simulator.run()
    trajectory = simulator.x

    for t, x_t in enumerate(trajectory):
        for gene_idx, g_t in enumerate(x_t):
            for cell_idx, c_t in enumerate(g_t):
                writer.add_scalar(f"gene{gene_idx}/type{cell_idx}", c_t, t)