"""One-epoch classification loops; accuracy is a percentage.

Pass device explicitly. Loss must be a scalar per-sample batch mean
(as with the notebooks' unweighted CrossEntropyLoss(reduction="mean")).
Model/optimizer construction and the outer epoch loop stay in the notebook.
"""
from timeit import default_timer as timer

import numpy as np
import torch

def train_step(model, data_loader, loss_fn, optimizer, device):
    model.to(device)
    model.train()
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
    start = timer()
    loss_sum, correct, count = 0.0, 0, 0
    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        # CrossEntropyLoss uses the batch mean; weight it by actual batch size.
        batch_size = y.size(0)
        loss_sum += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == y).sum().item()
        count += batch_size
    if count == 0:
        raise ValueError("Empty training loader")
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
    metrics = {"loss": loss_sum / count, "accuracy": 100.0 * correct / count,
               "training_seconds": timer() - start}
    print(f"Train loss: {metrics['loss']:.5f} | Train accuracy: {metrics['accuracy']:.2f}%")
    return metrics


def test_step(model, data_loader, loss_fn, device, num_classes=10, *, return_predictions=False):
    """Return loss, accuracy (%), confusion counts and evaluation_seconds.

    With return_predictions=True, also retain NumPy probabilities, targets and
    predictions in loader order for inference exports. Training callers leave
    this off to avoid retaining per-sample outputs.
    """
    model.to(device)
    model.eval()
    loss_sum, count = 0.0, 0
    confusion_counts = np.zeros((num_classes, num_classes), dtype=np.int64)
    probability_batches, target_batches = [], []
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
    start = timer()
    with torch.inference_mode():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            if return_predictions and not torch.isfinite(logits).all():
                raise ValueError("Non-finite logits")
            if return_predictions:
                probability_batches.append(logits.softmax(dim=1).cpu().numpy())
                target_batches.append(y.cpu().numpy())
            loss_sum += loss_fn(logits, y).item() * y.size(0)
            predictions = logits.argmax(dim=1)
            pairs = (y * num_classes + predictions).cpu().numpy()
            confusion_counts += np.bincount(
                pairs, minlength=num_classes**2).reshape(num_classes, num_classes)
            count += y.size(0)
    if count == 0:
        raise ValueError("Empty test loader")
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
    metrics = {"evaluation_seconds": timer() - start, "loss": loss_sum / count,
               "accuracy": 100.0 * float(np.trace(confusion_counts)) / count,
               "confusion_counts": confusion_counts.tolist()}
    if return_predictions:
        metrics["probabilities"] = np.concatenate(probability_batches)
        metrics["targets"] = np.concatenate(target_batches)
        metrics["predictions"] = metrics["probabilities"].argmax(axis=1)
    print(f"Test loss: {metrics['loss']:.5f} | Test accuracy: {metrics['accuracy']:.2f}%")
    return metrics


def summarize_training_time(entries):
    measured = [float(entry["training_seconds"]) for entry in entries if "training_seconds" in entry]
    return {
        "unit": "seconds",
        "total_seconds": sum(measured),
        "timed_epochs": len(measured),
        "scope": "Completed epochs: train_step wall time including data loading, preprocessing, transfers and optimizer updates; excludes setup, checks, test, plots, saving, pauses and interrupted epochs.",
    }

