import time
from copy import deepcopy

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import ttest_ind

from src.load_utils import load_grn, topo_sort_graph_layers, get_basal_production_rate

np.random.seed(123)


class Sim:

    def __init__(self, num_genes, num_cells_types, num_cells_to_simulate, interactions, regulators,
                 deterministic=False, **kwargs):
        self.interactions_filename = interactions
        self.regulators_filename = regulators
        self.num_genes = num_genes
        self.num_cell_types = num_cells_types
        self.num_cells_to_simulate = num_cells_to_simulate
        self.deterministic = deterministic
        self.adjacency = np.zeros(shape=(self.num_genes, self.num_genes))
        self.decay_lambda = 0.8
        self.mean_expression = -1 * np.ones((num_genes, num_cells_types))
        self.sampling_state = 15
        self.simulation_time_steps = self.sampling_state * self.num_cells_to_simulate
        print("sampling time steps: ", self.simulation_time_steps)
        self.x = np.zeros(shape=(self.simulation_time_steps, num_genes, num_cells_types))
        self.half_response = np.zeros(num_genes)
        self.hill_coefficient = 2

        self.p_value_for_convergence = 1e-3
        self.window_len = 100
        self.noise_parameters_genes = np.ones(num_genes)
        self._x = np.zeros(shape=(num_cells_to_simulate, num_genes, num_cells_types))

    def run(self):
        self.adjacency, graph = load_grn(self.interactions_filename, self.adjacency)
        layers = topo_sort_graph_layers(graph)
        basal_production_rate = get_basal_production_rate(self.regulators_filename, self.num_genes, self.num_cell_types)
        self.simulate_expression_layer_wise(layers, basal_production_rate)

    def simulate_expression_layer_wise(self, layers, basal_production_rate):
        random_sampling_state = np.random.randint(low=-self.simulation_time_steps, high=0,
                                                  size=self.num_cells_to_simulate)
        layers_copy = deepcopy(layers)
        for num_layer, layer in enumerate(layers_copy):
            if num_layer != 0:  # not the master layer
                self.calculate_half_response(layer)
            self.init_concentration(layer, basal_production_rate)
            print("layer: ", num_layer)
            for step in range(1, self.simulation_time_steps):
                curr_genes_expression = self.x[step - 1, layer]
                dx = self.euler_maruyama(basal_production_rate, curr_genes_expression, layer)

                updated_concentration_gene = curr_genes_expression + dx
                self.x[step, layer] = updated_concentration_gene.clip(0)  # clipping is important!

            # self.mean_expression[layer] = np.mean(self.x[random_sampling_state][:, layer], axis=0)
            # self._x[:, layer] = self.x[random_sampling_state][:, layer]
            self.mean_expression[layer] = np.mean(self.x[:, layer], axis=0)

    def calculate_half_response(self, layer):

        for gene in layer:
            regulators = np.where(self.adjacency[:, gene] != 0)
            mean_expression_per_cells_regulators_wise = self.mean_expression[regulators]
            half_response = np.mean(mean_expression_per_cells_regulators_wise)
            self.half_response[gene] = half_response

    def init_concentration(self, layer: list, basal_production_rate):
        """ Init concentration genes; Note: calculate_half_response should be run before this method """
        rates = np.array([self.calculate_production_rate(gene, basal_production_rate) for gene in layer])
        self.x[0, layer] = 1 # rates / self.decay_lambda

    def calculate_production_rate(self, gene, basal_production_rate):
        gene_basal_production = basal_production_rate[gene]
        if (gene_basal_production != 0).all():
            return gene_basal_production

        regulators = np.where(self.adjacency[:, gene] != 0)
        mean_expression = self.mean_expression[regulators]
        absolute_k = np.abs(self.adjacency[regulators][:, gene])
        is_repressive = np.expand_dims(self.adjacency[regulators][:, gene] < 0, -1).repeat(self.num_cell_types,
                                                                                           axis=-1)
        half_response = self.half_response[gene]
        hill_function = self.hill_function(mean_expression, half_response, is_repressive)
        rate = np.einsum("r,rt->t", absolute_k, hill_function)

        return rate

    def hill_function(self, regulators_concentration, half_response, is_repressive):
        rate = np.power(regulators_concentration, self.hill_coefficient) / (
                np.power(half_response, self.hill_coefficient) + np.power(regulators_concentration,
                                                                          self.hill_coefficient))

        rate[is_repressive] = 1 - rate[is_repressive]
        return rate

    def euler_maruyama(self, basal_production_rate, curr_genes_expression, layer):
        production_rates = [self.calculate_production_rate(gene, basal_production_rate) for gene in layer]
        decays = np.multiply(self.decay_lambda, curr_genes_expression)
        dw_p = np.random.normal(size=curr_genes_expression.shape)
        dw_d = np.random.normal(size=curr_genes_expression.shape)
        amplitude_p = np.einsum("g,gt->gt", self.noise_parameters_genes[layer], np.power(production_rates, 0.5))
        amplitude_d = np.einsum("g,gt->gt", self.noise_parameters_genes[layer], np.power(decays, 0.5))
        noise = np.multiply(amplitude_p, dw_p) + np.multiply(amplitude_d, dw_d)
        if self.deterministic:
            d_genes = 0.01 * np.subtract(production_rates, decays)
            return d_genes
        d_genes = 0.01 * np.subtract(production_rates, decays) + np.power(0.01, 0.5) * noise  # shape=(#genes,#types)
        return d_genes

    def check_for_convergence(self, gene_concentration, concentration_criteria='np_all_close'):
        converged = False

        if concentration_criteria == 't_test':
            sample1 = gene_concentration[-2 * self.window_len:-1 * self.window_len]
            sample2 = gene_concentration[-1 * self.window_len:]
            _, p = ttest_ind(sample1, sample2)
            if p >= self.p_value_for_convergence:
                converged = True

        elif concentration_criteria == 'mean':
            abs_mean_gene = np.abs(np.mean(gene_concentration[-self.window_len:]))
            if abs_mean_gene <= self.p_value_for_convergence:
                converged = True

        elif concentration_criteria == 'np_all_close':
            converged = np.allclose(gene_concentration[-2 * self.window_len:-1 * self.window_len],
                                    gene_concentration[-self.window_len:],
                                    atol=self.p_value_for_convergence)

        return converged


class TechnicalNoise:
    
    def __init__(self, num_genes: int, outlier_genes_noises: tuple, library_size_noises: tuple, dropout_noises: tuple):
        """ 
        @:param num_genes: number of genes
        @:param outlier_genes_noises: the noise of the outlier genes (pie, miu, sigma)
        @:param library_size_noises: the noise of the library size (miu, sigma)
        @:param dropout_noises: the noise of the dropout (k, q)
        
        this is the technical noise from the scRNA sequencing machine/ for seqFISH data
        those noises are data-driven parameters, have to be calculated from the real count matrix data
        """
        self.num_genes = num_genes
        self.pie_outlier_genes, self.miu_outlier_genes, self.sigma_outlier_genes = outlier_genes_noises
        self.miu_library_size, self.sigma_library_size = library_size_noises
        self.k_dropout, q_dropout = dropout_noises
    
    def get_noisy_technical_concentration(self, clean_concentration):
        assert clean_concentration.shape[1] == self.num_genes
        expr_add_outlier_genes = self.outlier_genes_effect(clean_concentration, outlier_prob=0.01, mean=0.8, scale=1)
        libFactor, expr_O_L = self.library_size_effect(expr_add_outlier_genes, mean=4.8, scale=0.3)
        binary_ind = self.dropout_indicator(expr_O_L, shape=20, percentile=82)
        expr_O_L_D = np.multiply(binary_ind, expr_O_L)
        count_matrix_umi_count_format = self.convert_to_UMIcounts(expr_O_L_D)
        noisy_concentration = np.concatenate(count_matrix_umi_count_format, axis=1)
        print(noisy_concentration.shape)
        return noisy_concentration

    def outlier_genes_effect(self, scData, outlier_genes_prob, mean, scale):
        out_indicator = np.random.binomial(n=1, p=outlier_genes_prob, size=self.nGenes_)
        outlierGenesIndx = np.where(out_indicator == 1)[0]
        numOutliers = len(outlierGenesIndx)

        #### generate outlier factors ####
        outFactors = np.random.lognormal(mean=mean, sigma=scale, size=numOutliers)
        ##################################

        scData = np.concatenate(scData, axis=1)
        for i, gIndx in enumerate(outlierGenesIndx):
            scData[gIndx, :] = scData[gIndx, :] * outFactors[i]

        return np.split(scData, self.nBins_, axis=1)

    def library_size_effect(self, scData, mean, scale):
        """
        This functions adjusts the mRNA levels in each cell seperately to mimic
        the library size effect. To adjust mRNA levels, cell-specific factors are sampled
        from a log-normal distribution with given mean and scale.

        scData: the simulated data representing mRNA levels (concentrations);
        np.array (#bins * #genes * #cells)

        mean: mean for log-normal distribution

        var: var for log-normal distribution

        returns libFactors ( np.array(nBin, nCell) )
        returns modified single cell data ( np.array(nBin, nGene, nCell) )
        """

        # TODO make sure that having bins does not intefere with this implementation
        ret_data = []

        libFactors = np.random.lognormal(mean=mean, sigma=scale, size=(self.nBins_, self.nSC_))
        for binExprMatrix, binFactors in zip(scData, libFactors):
            normalizFactors = np.sum(binExprMatrix, axis=0)
            binFactors = np.true_divide(binFactors, normalizFactors)
            binFactors = binFactors.reshape(1, self.nSC_)
            binFactors = np.repeat(binFactors, self.nGenes_, axis=0)

            ret_data.append(np.multiply(binExprMatrix, binFactors))

        return libFactors, np.array(ret_data)

    def dropout_indicator(self, scData, shape=1, percentile=65):
        """
        This is similar to Splat package

        Input:
        scData can be the output of simulator or any refined version of it
        (e.g. with technical noise)

        shape: the shape of the logistic function

        percentile: the mid-point of logistic functions is set to the given percentile
        of the input scData

        @returns: np.array containing binary indicators showing dropouts
        """
        scData = np.array(scData)
        scData_log = np.log(np.add(scData, 1))
        log_mid_point = np.percentile(scData_log, percentile)
        prob_ber = np.true_divide(1, 1 + np.exp(-1 * shape * (scData_log - log_mid_point)))

        binary_ind = np.random.binomial(n=1, p=prob_ber)

        return binary_ind

    def to_umi_counts(self, scData):
        return np.random.poisson(scData)


if __name__ == '__main__':
    start = time.time()
    interactions_filename = 'SERGIO/data_sets/De-noised_100G_9T_300cPerT_4_DS1/Interaction_cID_4.txt'
    regulators_filename = 'SERGIO/data_sets/De-noised_100G_9T_300cPerT_4_DS1/Regs_cID_4.txt'
    sim = Sim(num_genes=100, num_cells_types=9, num_cells_to_simulate=300,
              interactions=interactions_filename, regulators=regulators_filename, deterministic=True)
    sim.run()
    expr_clean = sim.x
    print(expr_clean.shape)
    print(f"took {time.time() - start} seconds")
    plt.plot(expr_clean.T[0, 1, :])
    plt.show()
    plt.plot(expr_clean.T[0, 99, :])
    plt.show()