# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/10_training_pytorch.ipynb.

# %% auto 0
__all__ = ['FocalLoss', 'iou_metric', 'false_positive_negative', 'train_segmentation_model']

# %% ../nbs/10_training_pytorch.ipynb 5
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from fastcore.all import *
from tqdm.auto import tqdm

# %% ../nbs/10_training_pytorch.ipynb 6
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        return F_loss


# %% ../nbs/10_training_pytorch.ipynb 7
def iou_metric(preds, labels, threshold=0.5):
    # Convert predictions to binary format
    preds = (preds > threshold).float().squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    
    # Ensure labels are binary
    labels = labels.float().squeeze(1)  # Similarly squeeze if necessary

    # Compute intersection and union
    intersection = (preds * labels).sum((1, 2))  # Logical AND
    union = ((preds + labels) > 0).float().sum((1, 2))  # Logical OR

    # Calculate IoU
    iou = (intersection + 1e-6) / (union + 1e-6)  # Smoothing to avoid 0/0

    return iou.mean()


# %% ../nbs/10_training_pytorch.ipynb 8
def false_positive_negative(outputs, labels):
    FP = ((outputs == 1) & (labels == 0)).float().sum()
    FN = ((outputs == 0) & (labels == 1)).float().sum()
    return FP, FN

# %% ../nbs/10_training_pytorch.ipynb 19
def train_segmentation_model(
                            train_loader,
                            val_loader,
                            model,
                            optimizer,
                            scheduler, 
                            loss_fn, 
                            model_save_path, 
                            warmup_epochs, 
                            total_epochs, 
                            initial_lr, 
                            dtype=torch.float,
                            device='cuda'):
    # Set the device for training
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device=device, dtype=dtype)

    for epoch in range(total_epochs):
        model.train()

        train_loss, train_iou_sum, train_fp_sum, train_fn_sum = 0.0, 0.0, 0.0, 0.0
        train_total = 0

        with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{total_epochs}', unit="batch") as pbar:
            for data, target in train_loader:
                # Move data and target to the specified device and data type
                data = data.to(device=device, dtype=dtype)
                target = target.to(device=device, dtype=dtype)

                optimizer.zero_grad()
                outputs = model(data)
                loss = loss_fn(outputs, target)
                loss.backward()
                optimizer.step()

                # Update metrics
                train_loss += loss.item()
                iou_score = iou_metric(outputs, target)
                train_iou_sum += iou_score.item()
                fp, fn = false_positive_negative(outputs, target)
                train_fp_sum += fp.item()
                train_fn_sum += fn.item()
                train_total += 1

                # Update progress bar
                pbar.set_postfix({'loss': loss.item(), 'iou': iou_score.item(), 'fp': fp.item(), 'fn': fn.item()})
                pbar.update()

        # ... Rest of the training loop as before ...
        train_loss /= train_total
        train_iou = train_iou_sum / train_total
        train_fp = train_fp_sum / train_total
        train_fn = train_fn_sum / train_total

        # Validation
        model.eval()
        val_loss, val_iou_sum, val_fp_sum, val_fn_sum = 0.0, 0.0, 0.0, 0.0
        val_total = 0
        with torch.no_grad():
            for data, target in val_loader:

                data = data.to(device=device, dtype=dtype)
                target = target.to(device=device, dtype=dtype)

                outputs = model(data)
                loss = loss_fn(outputs, target)

                # Update metrics
                val_loss += loss.item()
                iou_score = iou_metric(outputs, target)
                val_iou_sum += iou_score.item()
                fp, fn = false_positive_negative(outputs, target)
                val_fp_sum += fp.item()
                val_fn_sum += fn.item()
                val_total += 1

        val_loss /= val_total
        val_iou = val_iou_sum / val_total
        val_fp = val_fp_sum / val_total
        val_fn = val_fn_sum / val_total

        # Adjust learning rate after warm-up
        if epoch < warmup_epochs:
            lr = initial_lr * (epoch / warmup_epochs)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        else:
            scheduler.step()

          # Save model
        torch.save(model.state_dict(), f'{model_save_path}/model_epoch_{epoch+1}_valiou_{val_iou:.2f}_fp_{val_fp:.0f}_fn_{val_fn:.0f}.pth')


        # Print epoch summary
        print(f'Epoch {epoch+1}/{total_epochs} - Train Loss: {train_loss:.4f}, IoU: {train_iou:.4f}, FP: {train_fp}, FN: {train_fn}')
        print(f'Validation Loss: {val_loss:.4f}, IoU: {val_iou:.4f}, FP: {val_fp}, FN: {val_fn}')

