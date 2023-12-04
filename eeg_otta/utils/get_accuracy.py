from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader


def get_accuracy(model: nn.Module, data_loader: DataLoader, device: torch.device):
    outputs, labels = [], []
    with torch.no_grad():
        for batch in tqdm(data_loader):
            x, y = batch
            output = torch.softmax(model(x.to(device)), -1)
            outputs.append(output)
            labels.append(y)

    outputs = torch.concatenate(outputs)
    labels = torch.concatenate(labels)

    y_pred = outputs.argmax(-1).cpu()
    accuracy = (y_pred == labels).float().numpy().mean()

    return accuracy
