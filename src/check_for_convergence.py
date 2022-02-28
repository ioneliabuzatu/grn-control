def check_for_convergence(gene_concentration, window_len=100, p_value=1e-3, concentration_criteria='np_all_close'):
    assert len(gene_concentration.shape) == 3
    converged = False

    if concentration_criteria == 't_test':
        sample1 = gene_concentration[-2 * window_len:-1 * self.window_len]
        sample2 = gene_concentration[-1 * window_len:]
        _, p = ttest_ind(sample1, sample2)
        if p >= p_value:
            converged = True

    elif concentration_criteria == 'mean':
        abs_mean_gene = np.abs(np.mean(gene_concentration[-window_len:]))
        if abs_mean_gene <= p_value:
            converged = True

    elif concentration_criteria == 'np_all_close':
        converged = np.allclose(gene_concentration[-2 * window_len:-1 * window_len],
                                gene_concentration[-window_len:],
                                atol=p_value)

    return converged
