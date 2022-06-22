import torch
from sklearn.metrics import accuracy_score, f1_score


def graph_accuracy_score(preds: torch.tensor, gts: torch.tensor) -> float:
    preds = torch.round(torch.cat(preds))
    gts = torch.cat(gts)
    return accuracy_score(preds, gts)


def graph_f1_score(preds: torch.tensor, gts: torch.tensor) -> float:
    preds = torch.round(torch.cat(preds))
    gts = torch.cat(gts)
    return f1_score(preds, gts)
