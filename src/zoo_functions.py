import sys
import matplotlib.pyplot as plt


def is_debugger_active() -> bool:
    gettrace = getattr(sys, 'gettrace', lambda : None)
    return gettrace() is not None


def plot_three_genes(gene1, gene2, gene3, hlines=None):
    """sanity check one gene from each layer"""
    _, axes = plt.subplots(1, 3, figsize=(10, 5))
    axes[0].plot(gene1)
    axes[0].hlines(y=hlines[0], xmin=0, xmax=1500, linewidth=2, color='r')
    axes[0].set_title('Gene 44 in layer 0')
    axes[1].plot(gene2)
    axes[1].hlines(y=hlines[1], xmin=0, xmax=1500, linewidth=2, color='r')
    axes[1].set_title('Gene 1 in layer 1')
    axes[2].plot(gene3)
    axes[2].hlines(y=hlines[2], xmin=0, xmax=1500, linewidth=2, color='r')
    axes[2].set_title('Gene 99 in layer 2')
    plt.show()

    
def convert_mtx_matrix_to_csv_format(mex_dir, counts_filename, features_filename, barcodes_filename):
    """works as bash cmd: `cellranger mat2csv mtx-format/ data_converted.csv`
        where mtx-format containes 3 files [barcodes, features, counts]"""

    read_mex_format_matrix_as_table = scipy.io.mmread(os.path.join(mex_dir, counts_filename))
    features_path = os.path.join(mex_dir, features_filename)
    barcodes_path = os.path.join(mex_dir, barcodes_filename)

    feature_ids = [row[0] for row in csv.reader(gzip.open(features_path, mode="rt"), delimiter="\t")]
    gene_names = [row[1] for row in csv.reader(gzip.open(features_path, mode="rt"), delimiter="\t")]
    feature_types = [row[2] for row in csv.reader(gzip.open(features_path, mode="rt"), delimiter="\t")]
    barcodes = [row[0] for row in csv.reader(gzip.open(barcodes_path, mode="rt"), delimiter="\t")]

    # transform table to pandas dataframe and label rows and columns
    matrix = pd.DataFrame.sparse.from_spmatrix(read_mex_format_matrix_as_table)
    matrix.columns = barcodes
    matrix.insert(loc=0, column="feature_id", value=feature_ids)
    matrix.insert(loc=0, column="gene", value=gene_names)
    matrix.insert(loc=0, column="feature_type", value=feature_types)

    matrix.to_csv("mex_matrix.csv", index=False)