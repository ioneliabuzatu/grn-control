import sys
import matplotlib.pyplot as plt


def is_debugger_active() -> bool:
    gettrace = getattr(sys, 'gettrace', lambda : None)
    return gettrace() is not None


def plot_two_genes(gene1, gene2, gene3):
    _, axes = plt.subplots(1, 3, figsize=(10, 5))
    axes[0].plot(gene1)
    axes[0].set_title('Gene 44 layer 0')
    axes[1].plot(gene2)
    axes[1].set_title('Gene 1 in layer 1')
    axes[2].plot(gene3)
    axes[2].set_title('Gene 99 in layer 2')
    plt.show()
