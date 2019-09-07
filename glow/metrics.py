import torch


def categorical_accuracy(y_true, y_pred):
    with torch.no_grad():
        total = y_true.size(0)
        y_pred = torch.argmax(y_pred, dim=1).long().view(-1)
        correct = (y_pred == y_true).sum().item()
    return correct / total


def get(identifier):
    if identifier == "accuracy":
        return categorical_accuracy
    else:
        raise ValueError("Could not interpret " "metric identifier:", identifier)
