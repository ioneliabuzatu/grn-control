import experiment_buddy

import sim

params = {'num_genes': 100, 'num_cells_types': 9, 'num_cells_to_simulate': 5}

if __name__ == '__main__':
    main = False
    experiment_buddy.register_defaults(params)
    writer = experiment_buddy.deploy()

    simulator = sim.Sim(**params)
    simulator.interactions_filename = 'data/Interaction_cID_4.txt'
    simulator.regulators_filename = 'data/Regs_cID_4.txt'

    if main:
        simulator.run()
        trajectory = simulator.x

        for t, x_t in enumerate(trajectory):
            for gene_idx, g_t in enumerate(x_t):
                for cell_idx, c_t in enumerate(g_t):
                    writer.add_scalar(f"gene{gene_idx}/type{cell_idx}", c_t, t)


    else:
        trajectory = simulator.run()
        for t, x_t in trajectory.items():
            x_T = x_t.T
            for gene_idx, g_t in enumerate(x_t):
                for cell_idx, c_t in enumerate(g_t):
                    writer.add_scalar(f"gene{gene_idx}/type{cell_idx}", c_t, t)

