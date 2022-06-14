import numpy as np
import jax.numpy as jnp
import jax


class AddTechnicalNoiseJax:
    def __init__(self, num_genes: int, num_cell_types: int, num_simulated_cells: int, outlier_genes_noises: tuple,
                 library_size_noises: tuple,
                 dropout_noises: tuple, seed: int = 0):
        """
        @:param num_genes: number of genes
        @:param outlier_genes_noises: the noise of the outlier genes (pie, miu, sigma)
        @:param library_size_noises: the noise of the library size (miu, sigma)
        @:param dropout_noises: the noise of the dropout (k, q)
        @:param seed: np seed for numpy rng; included for experiments reproducibility purpose

        this is the technical noise from the scRNA sequencing machine/ for seqFISH data
        those noises are data-driven parameters, have to be calculated from the real count matrix data
        """
        self.num_genes = num_genes
        self.num_cell_types = num_cell_types
        self.num_simulated_cells = num_simulated_cells
        self.pie_outlier_genes, self.miu_outlier_genes, self.sigma_outlier_genes = outlier_genes_noises
        self.miu_library_size, self.sigma_library_size = library_size_noises
        self.k_dropout, self.q_dropout = dropout_noises
        self.seed = seed

        np.random.seed(seed=self.seed)  # Set the initial seed

    def get_noisy_technical_concentration(self, clean_concentration):
        assert clean_concentration.shape[1] == self.num_genes

        expr_add_outlier_genes = self._outlier_genes_effect(
            clean_concentration,
            outlier_genes_prob=self.pie_outlier_genes,
            mean=self.miu_outlier_genes,
            scale=self.sigma_outlier_genes
        )

        libFactor, expr_O_L = self._library_size_effect(
            expr_add_outlier_genes,
            self.miu_library_size,
            scale=self.sigma_library_size
        )
        binary_ind, key = self._dropout_indicator(expr_O_L, shape=self.k_dropout, percentile=self.q_dropout)
        expr_O_L_D = jnp.multiply(binary_ind, expr_O_L)
        # return jnp.vstack([expr_O_L_D.T[:,:,0], expr_O_L_D.T[:,:,1]])

        count_matrix_umi_count_format = jnp.array(self._to_umi_counts(expr_O_L_D, key))
        return jnp.vstack([count_matrix_umi_count_format.T[:, :, 0], count_matrix_umi_count_format.T[:, :, 1]])


    def _outlier_genes_effect(self, scData, outlier_genes_prob, mean, scale):
        out_indicator = np.random.binomial(n=1, p=outlier_genes_prob, size=self.num_genes)
        outlierGenesIndx = np.where(out_indicator == 1)[0]
        numOutliers = len(outlierGenesIndx)

        #### generate outlier factors ####
        outFactors = np.random.lognormal(mean=mean, sigma=scale, size=numOutliers)
        ##################################

        scData = jnp.concatenate(scData, axis=1)
        for i, gIndx in enumerate(outlierGenesIndx):
            # scData[gIndx, :] = scData[gIndx, :] * outFactors[i]
            scData = scData.at[gIndx, :].set(scData[gIndx, :] * outFactors[i])

        return jnp.array(jnp.split(scData, self.num_cell_types, axis=1))  # TODO: check if this is correct

    def _library_size_effect(self, scData, mean, scale):
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

        libFactors = np.random.lognormal(mean=mean, sigma=scale, size=(self.num_cell_types, self.num_simulated_cells))
        for binExprMatrix, binFactors in zip(scData, libFactors):
            normalizFactors = jnp.sum(binExprMatrix, axis=0)
            binFactors = jnp.true_divide(binFactors, normalizFactors)
            binFactors = binFactors.reshape(1, self.num_simulated_cells)
            binFactors = jnp.repeat(binFactors, self.num_genes, axis=0)

            ret_data.append(jnp.multiply(binExprMatrix, binFactors))

        return libFactors, jnp.array(ret_data)

    def _dropout_indicator(self, scData, shape=1, percentile=65):
        """
        Used because scRNA-seq destroys cells in the course of recording their profiles.

        This is similar to Splat package

        Input:
        scData can be the output of simulator or any refined version of it
        (e.g. with technical noise)

        shape: the shape of the logistic function

        percentile: the mid-point of logistic functions is set to the given percentile
        of the input scData

        @returns: np.array containing binary indicators showing dropouts
        """
        scData = jnp.array(scData)
        scData_log = jnp.log(jnp.add(scData, 1))
        log_mid_point = jnp.percentile(scData_log, percentile)
        prob_ber = jnp.true_divide(1, 1 + jnp.exp(-1 * shape * (scData_log - log_mid_point)))
        key = jax.random.PRNGKey(0)
        key, subkey = jax.random.split(key)
        binary_ind = jax.random.bernoulli(subkey, p=prob_ber, shape=None)
        return binary_ind, key

    def _to_umi_counts(self, scData, key):
        keys, subkey= jax.random.split(key)
        return jax.random.poisson(keys, lam=jnp.array(scData), shape=None) # TODO output
        # should be
        # dynamics
        # tracer
        # but it'd not, why?


class AddTechnicalNoise:

    def __init__(self, num_genes: int, num_cell_types: int, num_simulated_cells: int, outlier_genes_noises: tuple,
                 library_size_noises: tuple,
                 dropout_noises: tuple, seed: int = 0):
        """
        @:param num_genes: number of genes
        @:param outlier_genes_noises: the noise of the outlier genes (pie, miu, sigma)
        @:param library_size_noises: the noise of the library size (miu, sigma)
        @:param dropout_noises: the noise of the dropout (k, q)
        @:param seed: np seed for numpy rng; included for experiments reproducibility purpose

        this is the technical noise from the scRNA sequencing machine/ for seqFISH data
        those noises are data-driven parameters, have to be calculated from the real count matrix data
        """
        self.num_genes = num_genes
        self.num_cell_types = num_cell_types
        self.num_simulated_cells = num_simulated_cells
        self.pie_outlier_genes, self.miu_outlier_genes, self.sigma_outlier_genes = outlier_genes_noises
        self.miu_library_size, self.sigma_library_size = library_size_noises
        self.k_dropout, self.q_dropout = dropout_noises
        self.seed = seed

        np.random.seed(seed=self.seed)  # Set the initial seed

    def get_noisy_technical_concentration(self, clean_concentration):
        assert clean_concentration.shape[1] == self.num_genes

        expr_add_outlier_genes = self.outlier_genes_effect(
            clean_concentration,
            outlier_genes_prob=self.pie_outlier_genes,
            mean=self.miu_outlier_genes,
            scale=self.sigma_outlier_genes
        )

        libFactor, expr_O_L = self.library_size_effect(
            expr_add_outlier_genes,
            self.miu_library_size,
            scale=self.sigma_library_size
        )

        binary_ind = self.dropout_indicator(expr_O_L, shape=self.k_dropout, percentile=self.q_dropout)

        expr_O_L_D = np.multiply(binary_ind, expr_O_L)
        count_matrix_umi_count_format = self.to_umi_counts(expr_O_L_D)
        noisy_concentration = np.concatenate(count_matrix_umi_count_format, axis=1)

        assert noisy_concentration.shape[0] == self.num_genes

        return noisy_concentration

    def outlier_genes_effect(self, scData, outlier_genes_prob, mean, scale):
        out_indicator = np.random.binomial(n=1, p=outlier_genes_prob, size=self.num_genes)
        outlierGenesIndx = np.where(out_indicator == 1)[0]
        numOutliers = len(outlierGenesIndx)

        #### generate outlier factors ####
        outFactors = np.random.lognormal(mean=mean, sigma=scale, size=numOutliers)
        ##################################

        scData = np.concatenate(scData, axis=1)
        for i, gIndx in enumerate(outlierGenesIndx):
            scData[gIndx, :] = scData[gIndx, :] * outFactors[i]

        return np.split(scData, self.num_cell_types, axis=1)

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

        libFactors = np.random.lognormal(mean=mean, sigma=scale, size=(self.num_cell_types, self.num_simulated_cells))
        for binExprMatrix, binFactors in zip(scData, libFactors):
            normalizFactors = np.sum(binExprMatrix, axis=0)
            binFactors = np.true_divide(binFactors, normalizFactors)
            binFactors = binFactors.reshape(1, self.num_simulated_cells)
            binFactors = np.repeat(binFactors, self.num_genes, axis=0)

            ret_data.append(np.multiply(binExprMatrix, binFactors))

        return libFactors, np.array(ret_data)

    def dropout_indicator(self, scData, shape=1, percentile=65):
        """
        Used because scRNA-seq destroys cells in the course of recording their profiles.

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
