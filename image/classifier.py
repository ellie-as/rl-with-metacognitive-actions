import logging
import numpy as np
import torch
import torch.nn as nn
from skorch import NeuralNetClassifier
import torch.nn as nn
import torch.nn.functional as F
from config import Config


class _SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.net(x)


class Classifier(NeuralNetClassifier):
    def __init__(
        self,
        module=_SimpleNet,
        criterion=nn.CrossEntropyLoss,
        max_epochs=1,
        lr=1e-4,
        optimizer=torch.optim.Adam,
        iterator_train__shuffle=True,
        device=Config.DEVICE,
        **kwargs,
    ):
        # kwargs is expected to contain {"train_split": False}
        super().__init__(
            module=module,
            criterion=criterion,
            max_epochs=max_epochs,
            lr=lr,
            optimizer=optimizer,
            iterator_train__shuffle=iterator_train__shuffle,
            device=device,
            **kwargs,    # train_split=False comes here, not twice
        )

    def __sklearn_clone__(self):
        params = self.get_params(deep=False)
        params["train_split"] = False        # ensure every clone has no splitter
        clone = Classifier(**params)
        if hasattr(self, "module_"):
            clone.initialize()
            clone.module_.load_state_dict(self.module_.state_dict())
            clone.initialized_ = True
        return clone

    def evaluate_with_breakdown(self, dataloader):
        if not hasattr(self, "module_"):
            logging.warning("evaluate() called before any fit.")
            return 0.0, np.zeros(Config.NUM_CLASSES, dtype=np.float32)
        self.module_.eval()
        total = correct = 0
        per_class_correct = np.zeros(Config.NUM_CLASSES, dtype=np.float32)
        per_class_total = np.zeros(Config.NUM_CLASSES, dtype=np.float32)
        with torch.no_grad():
            for X, y in dataloader:
                preds = self.predict(X.numpy())
                y_np = y.numpy()
                total += len(y_np)
                correct += (preds == y_np).sum()
                for cls in range(Config.NUM_CLASSES):
                    mask = y_np == cls
                    if mask.any():
                        per_class_total[cls] += mask.sum()
                        per_class_correct[cls] += (preds[mask] == cls).sum()
        acc = correct / total if total else 0.0
        per_class_acc = np.divide(
            per_class_correct,
            np.maximum(per_class_total, 1),
            out=np.zeros_like(per_class_correct),
            where=per_class_total > 0,
        )
        logging.info(f"Classifier => {acc*100:.2f}% accuracy")
        return acc, per_class_acc.astype(np.float32)

    def evaluate(self, dataloader):
        acc, _ = self.evaluate_with_breakdown(dataloader)
        return acc
