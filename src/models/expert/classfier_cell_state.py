import numpy as np
import jax.experimental.stax as stax
import jax.random
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import Subset


@torch.no_grad()
def evaluate(network: nn.Module, data: DataLoader, metric, use_bce_loss=False) -> list:
    network.eval()
    device = next(network.parameters()).device

    errors = []
    for x, y in data:
        logits = network(x.to(device))
        errors.append(metric(logits, y.long().to(device)).item())

    return errors


@torch.enable_grad()
def update(network: nn.Module, data: DataLoader, loss: nn.Module,
           opt: optim.Optimizer, regulariser: nn.Module = None) -> list:
    network.train()
    device = next(network.parameters()).device

    # sensitivity analysis
    # x.requires_grad = True
    # logits = network(x)
    # logits.abs().sum().backward()
    # print("Sensitivity:", x.grad.abs().mean(0))
    # x.requires_grad = False

    errs = []
    for x, y in data:
        opt.zero_grad()
        x = x.to(device)
        logits = network(x)
        err = loss(logits, y.long().to(device))
        # print(nn.Sigmoid()(logits.detach().cpu()), y)
        errs.append(err.item())
        err.backward()
        opt.step()

    return errs


@torch.no_grad()
def accuracy(logits, targets):
    """
    Compute the accuracy for given logits and targets.

    Parameters
    ----------
    logits : (N, K) torch.Tensor
        A mini-batch of logit vectors from the network.
    targets : (N, ) torch.Tensor
        A mini_batch of target scalars representing the labels.

    Returns
    -------
    acc : () torch.Tensor
        The accuracy over the mini-batch of samples.
    """
    tot_correct = tot_samples = 0

    # _, predictions = logits.max(1)
    predictions = logits
    tot_correct += (predictions == targets).sum()
    tot_samples += predictions.size(0)

    accuracy_samples = (tot_correct.item() / tot_samples) * 100

    return accuracy_samples


@torch.no_grad()
def binary_acc(logits, targets):
    if logits.dim != targets.dim:
        targets = targets.unsqueeze(1)

    y_pred_tag = torch.round(torch.sigmoid(logits))

    correct_results_sum = (y_pred_tag == targets).sum().float()
    acc = correct_results_sum / targets.shape[0]
    acc = torch.round(acc * 100)
    return acc.item()


@torch.no_grad()
def multiclass_acc(logits, targets):
    probs = torch.softmax(logits, dim=1)
    winners = probs.argmax(dim=1)
    corrects = (winners == targets)
    accuracy = corrects.sum().float() / float(targets.size(0))
    return accuracy.item()


@torch.no_grad()
def get_accuracy(network, dataloader):
    network.eval()
    device = next(network.parameters()).device
    accuracies = []
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        logits = network(inputs)
        # accuracy_mini_batch = binary_acc(logits, targets)
        accuracy_mini_batch = multiclass_acc(logits, targets)
        accuracies.append(accuracy_mini_batch)

    return accuracies


class CellStateClassifier(nn.Module):
    """ Classifier network for classifying cell state {healthy, unhealthy} """

    def __init__(self, num_genes: int, num_cell_types):
        """
        Parameters
        ----------
        num_classes : int
            The number of output classes in the data.
        """
        super(CellStateClassifier, self).__init__()

        self.fc1 = nn.Linear(num_genes, num_genes // 2)
        self.fc2 = nn.Linear(num_genes // 2, num_cell_types)
        self.classifier = nn.Sequential(
            self.fc1,
            nn.SELU(),
            self.fc2,
            # nn.SELU(),
            # self.fcx,
            # nn.SELU(),
            # self.fc3
        )

    def forward(self, x):
        x = self.classifier(x)
        return x


class TranscriptomicsDataset(Dataset):
    def __init__(self, filepath_data, device, normalize_by_max=True, num_genes_for_batch_sgd=5000):
        """
        :param filepath_data: stacked npy file with first rows genes names and last column the labels.
        :param device:
        :param normalize_by_max:
        :param num_genes_for_batch_sgd:

        self.labels_encoding is ["2weeks_after_crush", "contro"] so `label 0` is disease and `label 1` is control.
        """
        self.normalize_data = normalize_by_max
        self.device = device
        self.data = np.load(filepath_data, allow_pickle=True)
        self.num_genes_to_zero_for_batch_sgd = (self.data.shape[1] - num_genes_for_batch_sgd) - 1
        self.num_genes_to_take_for_batch_sgd = num_genes_for_batch_sgd
        print(f"data input has size: {self.data.shape}")
        if isinstance(self.data[0, 0], str):
            self.genes_names = self.data[0, :]
            self.data = self.data[1:, :]
        self.preprocess_data()
        # self.labels_encoding, self.labels_categorical = np.unique(self.data[:, -1], return_inverse=True)
        self.labels_encoding, self.labels_categorical = np.unique(
            np.concatenate([np.array([str(x)] * 10000, dtype=object) for x in range(9)], axis=0), return_inverse=True)

    def __getitem__(self, idx):
        # data = self.data[idx, :-1]
        data = self.data[idx]  # indices_to_set_to_zero = np.random.permutation(self.num_genes_to_zero_for_batch_sgd)
        # data = data[indices_to_set_to_zero]
        # data[indices_to_set_to_zero] = 0.0
        x = torch.tensor(data, dtype=torch.float32)
        y = torch.from_numpy(np.array(self.labels_categorical[idx], dtype=np.float32))
        del data
        return x, y

    def __len__(self):
        return len(self.data)

    def preprocess_data(self):
        """
        TODO: try TPM normalization too
        what is TPM normalization:  https://www.rna-seqblog.com/rpkm-fpkm-and-tpm-clearly-explained/
        """
        # x_normed = data / data.max(axis=1)
        if self.normalize_data:
            # self.data[1:, :-1] = normalize(self.data[1:, :-1], axis=1, norm="max")
            self.data = normalize(self.data, axis=1, norm="max")


class RandomDataset(Dataset):
    def __init__(self, device, num_genes, num_samples, num_classes):
        """
        :param filepath_data: stacked npy file with first rows genes names and last column the labels.
        :param device:
        :param normalize_by_max:
        :param num_genes_for_batch_sgd:

        self.labels_encoding is ["2weeks_after_crush", "contro"] so `label 0` is disease and `label 1` is control.
        """
        self.device = device
        self.data = np.random.random((num_samples * num_classes, num_genes))
        # self.num_genes_to_zero_for_batch_sgd = (self.data.shape[1] - num_genes_for_batch_sgd) - 1
        # self.num_genes_to_take_for_batch_sgd = num_genes_for_batch_sgd
        print(f"data input has size: {self.data.shape}")
        if isinstance(self.data[0, 0], str):
            self.genes_names = self.data[0, :]
            self.data = self.data[1:, :]
        # self.labels_encoding, self.labels_categorical = np.unique(self.data[:, -1], return_inverse=True)
        self.labels_encoding, self.labels_categorical = np.unique(
            np.concatenate(
                [np.array([str(x)] * num_samples, dtype=object) for x in range(num_classes)], axis=0
            ),
            return_inverse=True)

    def __getitem__(self, idx):
        # data = self.data[idx, :-1]
        data = self.data[idx]  # indices_to_set_to_zero = np.random.permutation(self.num_genes_to_zero_for_batch_sgd)
        # data = data[indices_to_set_to_zero]
        # data[indices_to_set_to_zero] = 0.0
        x = torch.tensor(data, dtype=torch.float32)
        y = torch.from_numpy(np.array(self.labels_categorical[idx], dtype=np.float32))
        del data
        return x, y

    def __len__(self):
        return len(self.data)


def train_val_dataset(dataset, val_split=0.25):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {'train': Subset(dataset, train_idx), 'val': Subset(dataset, val_idx)}
    return datasets


def torch_to_jax(model=None):
    import jax.experimental.stax as stax
    import jax.random
    # if model is None:
    #     model = CellStateClassifier(100)
    init_fn, predict_fn = stax.serial(
        stax.Dense(model.fc1.out_features, lambda *_: model.fc1.weight.detach().numpy().T,
                   lambda *_: model.fc1.bias.detach().numpy()),
        stax.Selu,
        stax.Dense(model.fc2.out_features, lambda *_: model.fc2.weight.detach().numpy().T,
                   lambda *_: model.fc2.bias.detach().numpy()),

        # stax.Selu,
        # stax.Dense(model.fcx.out_features, lambda *_: model.fcx.weight.detach().numpy().T,
        #            lambda *_: model.fcx.bias.detach().numpy()),

        # stax.Selu,
        # stax.Dense(model.fc3.out_features, lambda *_: model.fc3.weight.detach().numpy().T,
        #            lambda *_: model.fc3.bias.detach().numpy()),
    )
    rng_key = jax.random.PRNGKey(0)
    _, params = init_fn(rng_key, (model.fc1.in_features,))

    def jax_model(x):
        return predict_fn(params, x)

    return jax_model


def torch_to_jax_resnet(model=None):
    import jax.experimental.stax as stax
    import jax.random
    block1 = stax.serial(
        stax.Dense(model.fc1.out_features, lambda *_: model.fc1.weight.detach().numpy().T,
                   lambda *_: model.fc1.bias.detach().numpy()),
        stax.Selu, ),
    block2 = stax.serial(stax.Dense(model.fc2.out_features, lambda *_: model.fc2.weight.detach().numpy().T,
                                    lambda *_: model.fc2.bias.detach().numpy()), stax.Selu, ),
    block3 = stax.serial(
        stax.Dense(model.fcx.out_features, lambda *_: model.fcx.weight.detach().numpy().T,
                   lambda *_: model.fcx.bias.detach().numpy()),
        stax.Selu,
    )
    block4 = stax.serial(
        stax.Dense(model.fc3.out_features, lambda *_: model.fc3.weight.detach().numpy().T,
                   lambda *_: model.fc3.bias.detach().numpy()),
    )

    init_fn, predict_fn = stax.serial(
        stax.parallel(block1, stax.Identity),
        stax.parallel(block2, stax.Identity),
        stax.parallel(block3, stax.Identity),
        block4,
    )

    rng_key = jax.random.PRNGKey(0)
    _, params = init_fn(rng_key, (model.fc1.in_features,))

    def jax_model(x):
        return predict_fn(params, x)

    return jax_model


if __name__ == '__main__':
    predict_fn = torch_to_jax()
    out = predict_fn(np.zeros((1, 100)))
    print(out)
