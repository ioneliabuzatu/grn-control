import logging
import os

import experiment_buddy
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# import config
from src.models.expert.classfier_cell_state import MiniCellStateClassifier
from src.models.expert.classfier_cell_state import TranscriptomicsDataset
from src.models.expert.classfier_cell_state import evaluate
from src.models.expert.classfier_cell_state import get_accuracy
from src.models.expert.classfier_cell_state import train_val_dataset
from src.models.expert.classfier_cell_state import update, RandomDataset


class config():
    import getpass
    whoami = getpass.getuser()
    if whoami == 'ionelia':
        checkpoint_folder = "src/models/expert/checkpoints/"
        data = "/home/ionelia/pycharm-projects/grn-control/data/GEO/GSE122662/graph-experiments" \
               "/28_genes_last_col_are_labels.npy"
    elif whoami == 'ionelia.buzatu':
        from pathlib import Path
        home = str(Path.home())
        checkpoint_folder = home
        data = "data/ds4_10k_each_type.npy"

    tensorboard = True
    num_genes = 28
    num_cell_types = 2
    lr = 1e-4
    epochs = 300
    batch_size = 600
    normalize_by_max = False


config = config()
experiment_buddy.register_defaults({'dataset': '?', **config.__dict__})
writer = experiment_buddy.deploy(host="", wandb_run_name="2 layers 28 genes", disabled=False, wandb_kwargs={
    'project': "train-expert"})


def train(filepath_training_data, epochs=200, show_confusion_matrix=False):
    # os.system("nvidia-smi")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Run on {device}")

    torch.manual_seed(186)
    if device == "cuda":
        torch.cuda.manual_seed(186)

    # writer = config.writer

    network = MiniCellStateClassifier(num_genes=config.num_genes, num_cell_types=config.num_cell_types).to(device)
    sgd = optim.SGD(network.parameters(), lr=config.lr, momentum=0.95)

    criterium = nn.CrossEntropyLoss()

    dataset = TranscriptomicsDataset(
        filepath_data=filepath_training_data, device=device, normalize_by_max=config.normalize_by_max
    )
    # dataset = RandomDataset(device, config.num_genes, 20, 2)
    train_and_val_dataset = train_val_dataset(dataset, val_split=0.25)
    train_dataloader = DataLoader(train_and_val_dataset["train"], batch_size=config.batch_size, shuffle=True)
    val_dataloader = DataLoader(train_and_val_dataset["val"], batch_size=config.batch_size, shuffle=True)

    init_mean_acc = np.asarray(get_accuracy(network, val_dataloader)).mean()
    print("accuracy of untrained model:", init_mean_acc)

    val_accuracy = float("-inf")
    for epoch in tqdm(range(epochs)):

        train_errors = update(network, train_dataloader, criterium, sgd)
        writer.add_scalar("TrainingLoss", np.asarray(train_errors).mean(), epoch)
        print("train error", np.asarray(train_errors).mean())

        val_errors = evaluate(network, val_dataloader, criterium)
        writer.add_scalar("ValidationLoss", np.asarray(val_errors).mean(), epoch)
        print("val error", np.asarray(val_errors).mean())

        accuracies = get_accuracy(network, val_dataloader)
        mean_accuracy = np.asarray(accuracies).mean()
        writer.add_scalar("AccuracyVal", mean_accuracy, epoch)
        print("acc val", mean_accuracy)

        if mean_accuracy > val_accuracy:
            val_accuracy = mean_accuracy
            logging.info(f"Saving new model...new accuracy (val): {val_accuracy}")
            torch.save(network.state_dict(), os.path.join(config.checkpoint_folder,
                                                          "expert_106G_trained_on_de_noised_debug.pth"))

        if show_confusion_matrix:
            nb_classes = 2
            confusion_matrix = torch.zeros(nb_classes, nb_classes)
            with torch.no_grad():
                for i, (inputs, classes) in enumerate(val_dataloader):
                    inputs = inputs.to(device)
                    classes = classes.to(device)
                    outputs = network(inputs)
                    _, preds = torch.max(outputs, 1)
                    for t, p in zip(classes.view(-1), preds.view(-1)):
                        confusion_matrix[t.long(), p.long()] += 1
            print(confusion_matrix)


if __name__ == "__main__":
    train(filepath_training_data=config.data, epochs=config.epochs)
