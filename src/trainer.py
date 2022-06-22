import torch
from metric import graph_accuracy_score, graph_f1_score
from typing import Tuple


class Trainer:
    def __init__(self, model, loss_func, optimizer, device=None):
        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.device = device or "cpu"

    def train(self, train_loader) -> float:
        self.model.train()
        total_loss = 0
        for data in train_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            out = self.model(data.x, data.edge_index, data.batch)
            loss = self.loss_func(torch.reshape(out, (-1,)), data.y.float())
            loss.backward()
            self.optimizer.step()
            total_loss += float(loss) * data.num_graphs
        return total_loss / len(train_loader.dataset)

    @torch.no_grad()
    def test(self, test_loader) -> Tuple[float, float, float]:
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        for data in test_loader:
            data = data.to(self.device)
            out = self.model(data.x, data.edge_index, data.batch)
            loss = self.loss_func(torch.reshape(out, (-1,)), data.y.float())
            total_loss += float(loss) * data.num_graphs
            all_preds.append(torch.reshape(out, (-1,)).cpu())
            all_labels.append(data.y.float().cpu())

        # Calculate Metrics
        accuracy = graph_accuracy_score(all_preds, all_labels)
        f1 = graph_f1_score(all_preds, all_labels)

        return total_loss / len(test_loader.dataset), accuracy, f1
