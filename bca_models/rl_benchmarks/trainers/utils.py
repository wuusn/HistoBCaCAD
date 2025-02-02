# Copyright (c) Owkin, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Training and inference step for slide-level experiments."""

from typing import Optional, Tuple

import numpy as np
import torch


def slide_level_train_step(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str = "cpu",
    gc_step: Optional[int] = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Training step for slide-level experiments. This will serve as the
    ``train_step`` in ``TorchTrainer``class.

    Parameters
    ----------
    model: nn.Module
        The PyTorch model to be trained.
    train_dataloader: torch.utils.data.DataLoader
        Training data loader.
    criterion: nn.Module
        The loss criterion used for training.
    optimizer: Callable = Adam
        The optimizer class to use.
    device : str = "cpu"
        The device to use for training and evaluation.
    gc_step: Optional[int] = 1
        The number of gradient accumulation steps.
    """
    model.train()

    _epoch_loss, _epoch_logits, _epoch_preds, _epoch_labels = [], [[],[]], [[],[]], [[],[]]

    for i, batch in enumerate(train_dataloader):
        # Get data.
        # features, mask, labels = batch
        features, labels = batch

        # Put on device.
        features = features.to(device)
        # mask = mask.to(device)
        labels = [l.to(device) for l in labels]

        # Compute logits and loss.
        # logits = model(features, mask)
        logits = model(features)
        outputs = logits
        losses = []
        for j in range(len(outputs)):
            if j ==1:
                output = outputs[j]
                # output = torch.sigmoid(outputs[i])
                pred = torch.max(output, dim=1)[1]
                label = labels[j]
                # print(label)
                # print(output)
                mask = labels[0] != 0
                loss = criterion(output, label) * mask.float()
                loss = torch.mean(loss)
                # pred = pred[mask]
                # label = label[mask]
                # output = output[mask]
            else: 
                output = outputs[j]
                # output = torch.sigmoid(outputs[i])
                pred = torch.max(output, dim=1)[1]
                label = labels[j]
                loss = criterion(output, label)
                loss = torch.mean(loss)
            losses.append(loss)
            _epoch_logits[j].extend(torch.sigmoid(logits[j]).tolist())
            _epoch_labels[j].extend(labels[j].tolist())
            _epoch_preds[j].extend(pred.tolist())
        
        loss = (losses[0]+losses[1])/2

        # Optional: Gradient accumulation.
        loss = loss / gc_step
        loss.backward(retain_graph=True)   

        if ((i + 1) % gc_step == 0) or ((i + 1) == len(train_dataloader)):
            optimizer.step()
            optimizer.zero_grad()

        # Stack logits & labels to compute epoch metrics.
        _epoch_loss.append(loss.detach().cpu().numpy())
        

    _epoch_loss = np.mean(_epoch_loss)
    # _epoch_logits = torch.cat(_epoch_logits, dim=0).cpu().numpy()
    # _epoch_labels = torch.cat(_epoch_labels, dim=0).cpu().numpy()

    return _epoch_loss, _epoch_logits, _epoch_preds, _epoch_labels


def slide_level_val_step(
    model: torch.nn.Module,
    val_dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inference step for slide-level experiments. This will serve as the
    ``val_step`` in ``TorchTrainer``class.

    Parameters
    ----------
    model: nn.Module
        The PyTorch model to be trained.
    val_dataloader: torch.utils.data.DataLoader
        Inference data loader.
    criterion: nn.Module
        The loss criterion used for training.
    device : str = "cpu"
        The device to use for training and evaluation.
    """
    model.eval()

    with torch.no_grad():
        _epoch_loss, _epoch_logits, _epoch_preds, _epoch_labels = [], [[],[]], [[],[]], [[],[]]

        for batch in val_dataloader:
            # Get data.
            # features, mask, labels = batch
            features, labels = batch

            # Put on device.
            features = features.to(device)
            # mask = mask.to(device)
            labels = [l.to(device) for l in labels]

            # Compute logits and loss.
            # logits = model(features, mask)
            logits = model(features)
            outputs = logits
            losses = []
            for i in range(len(outputs)):
                if i ==1:
                    output = outputs[i]
                    # output = torch.sigmoid(outputs[i])
                    pred = torch.max(output, dim=1)[1]
                    label = labels[i]
                    # print(label)
                    # print(output)
                    mask = labels[0] != 0
                    loss = criterion(output, label) * mask.float()
                    loss = torch.mean(loss)
                    # pred = pred[mask]
                    # label = label[mask]
                    # output = output[mask]
                else: 
                    output = outputs[i]
                    # output = torch.sigmoid(outputs[i])
                    pred = torch.max(output, dim=1)[1]
                    label = labels[i]
                    loss = criterion(output, label)
                    loss = torch.mean(loss)
                losses.append(loss)
                _epoch_logits[i].extend(torch.sigmoid(logits[i]).tolist())
                _epoch_labels[i].extend(labels[i].tolist())
                _epoch_preds[i].extend(pred.tolist())
            
            loss = (losses[0]+losses[1])/2

            # Stack logits & labels to compute epoch metrics.
            _epoch_loss.append(loss.detach().cpu().numpy())


    _epoch_loss = np.mean(_epoch_loss)
    # _epoch_logits = torch.cat(_epoch_logits, dim=0).cpu().numpy()
    # _epoch_labels = torch.cat(_epoch_labels, dim=0).cpu().numpy()

    return _epoch_loss, _epoch_logits, _epoch_preds, _epoch_labels
