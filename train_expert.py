import logging
import os

import numpy as np
import torch
from torch import nn
import experiment_buddy
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# import config
from src.models.expert.classfier_cell_state import CellStateClassifier
from src.models.expert.classfier_cell_state import TranscriptomicsDataset, RandomDataset
from src.models.expert.classfier_cell_state import evaluate
from src.models.expert.classfier_cell_state import get_accuracy
from src.models.expert.classfier_cell_state import train_val_dataset
from src.models.expert.classfier_cell_state import update


class config():
    checkpoint_folder = "src/models/expert/checkpoints/"
    data = "data/ds4_10k_each_type.npy"
    tensorboard = True
    num_genes = 7000
    num_cell_types = 2
    lr = 1e-3
    epochs = 500
    batch_size = 128
    samples_each_type = 10000


config = config()
experiment_buddy.register_defaults({'dataset': 'ds4', **config.__dict__})
writer = experiment_buddy.deploy(host='mila', disabled=True)


def train(filepath_training_data, epochs=200):
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    print(f"Run on {device}")

    torch.manual_seed(1806)
    if device == "cuda":
        torch.cuda.manual_seed(1806)

    # writer = config.writer

    network = CellStateClassifier(num_genes=config.num_genes, num_cell_types=config.num_cell_types).to(device)
    sgd = optim.SGD(network.parameters(), lr=config.lr, momentum=0.9, weight_decay=1e-3)
    # criterium = nn.BCEWithLogitsLoss()
    criterium = nn.CrossEntropyLoss()

    # dataset = TranscriptomicsDataset(filepath_data=filepath_training_data, device=device)
    dataset = RandomDataset(device, config.num_genes, 10_000, 2)
    train_and_val_dataset = train_val_dataset(dataset, val_split=0.25)
    train_dataloader = DataLoader(train_and_val_dataset["train"], batch_size=config.batch_size, shuffle=True)
    val_dataloader = DataLoader(train_and_val_dataset["val"], batch_size=config.batch_size, shuffle=True)

    val_accuracy = float("-inf")
    for epoch in tqdm(range(epochs)):
        train_errors = update(network, train_dataloader, criterium, sgd)
        writer.add_scalar("TrainingLoss", np.asarray(train_errors).mean(), epoch)
        print("train", np.asarray(train_errors).mean())
        val_errors = evaluate(network, val_dataloader, criterium)
        writer.add_scalar("ValidationLoss", np.asarray(val_errors).mean(), epoch)
        print("val", np.asarray(val_errors).mean())
        accuracies = get_accuracy(network, val_dataloader)
        mean_accuracy = np.asarray(accuracies).mean()
        writer.add_scalar("AccuracyVal", mean_accuracy, epoch)
        print("acc", mean_accuracy)

        if mean_accuracy > val_accuracy:
            val_accuracy = mean_accuracy
            logging.info(f"Saving new model...new accuracy (val): {val_accuracy}")
            torch.save(network.state_dict(), os.path.join(
                config.checkpoint_folder, "delete_me.pth")
                       )


if __name__ == "__main__":
    train(filepath_training_data=config.data, epochs=config.epochs)
