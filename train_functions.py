import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import tqdm
import torch.nn as nn
import torch.optim as optim
import logging
import wandb
from omegaconf import DictConfig
from metrics import calculate_metrics, calculate_iou_metrics
from typing import Optional
import gc

def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    loss_function: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    cfg: DictConfig,
    log: logging.Logger,
    epoch: int,
) -> float:
    model.train()
    running_loss = 0.0
    for batch_idx, (image, mask) in enumerate(
        tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}")
    ):
        optimizer.zero_grad(set_to_none=True)
        image = image.to(device)
        mask = mask.to(device)

        mask = 1 - (mask / 255)
        mask = F.one_hot(mask.long(), num_classes=2).squeeze(1).permute(0, 3, 1, 2)

        optimizer.zero_grad()

        #############################
        #angepasst an das Hedunet mit Deep Supervision
        ##############################
        prediction = model(image)

        # Loss funktioniert mit Tensor oder Liste
        loss = loss_function(prediction, mask)

        # Für Debugging/Visualisierung nur Hauptoutput nehmen
        if isinstance(prediction, (list, tuple)):
            main_output = prediction[0]
        else:
            main_output = prediction
        ##############################

        loss.backward()
        optimizer.step()
        if cfg.get("use_wandb", False):
            wandb.log({"train_loss": loss.item()})

        running_loss += loss.item()

        if batch_idx % cfg["training"].get("log_interval", 10) == 0:
            log.info(
                f"Train Epoch: {epoch+1} [{batch_idx * len(image)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
            )
        if batch_idx % 50 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    epoch_loss = running_loss / len(train_loader)
    log.info(f"Train Epoch: {epoch+1} Average Loss: {epoch_loss:.6f}")

    if cfg.get("use_wandb", False):
        wandb.log({"epoch_train_loss": epoch_loss}, step=epoch)

    return epoch_loss


def validate_with_metrics(
    model: nn.Module,
    val_loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    cfg: DictConfig,
    log: logging.Logger,
    epoch: Optional[int] = None,
):
    """
    Enhanced validation function that calculates comprehensive metrics.
    """
    model.eval()
    val_loss = 0.0
    num_classes = cfg.model.classes
    # Initialize metric accumulators
    total_precision = torch.zeros(num_classes, device=device)
    total_recall = torch.zeros(num_classes, device=device)
    total_f1 = torch.zeros(num_classes, device=device)
    total_class_ious = torch.zeros(num_classes, device=device)

    num_batches = 0
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(val_loader):
            images = images.to(device)
            masks = masks.to(device)

            masks = masks / 255
            
            one_hot_masks = (
                F.one_hot(masks.long(), num_classes=2).squeeze(1).permute(0, 3, 1, 2)
            )
            #########################
            #angepasst an das Hedunet mit Deep Supervision
            #########################
            outputs = model(images)

            # Loss kann mit Liste umgehen
            loss = loss_function(outputs, one_hot_masks)
            val_loss += loss.item()

            # Für Metriken nur Hauptoutput nehmen
            if isinstance(outputs, (list, tuple)):
                main_output = outputs[0]
            else:
                main_output = outputs

            # Get predictions
            preds = torch.argmax(main_output, dim=1)

            # Calculate metrics for this batch
            batch_precision, batch_recall, batch_f1 = calculate_metrics(
                masks, preds, num_classes, device
            )
            batch_class_ious, batch_mean_iou = calculate_iou_metrics(
                masks, preds, num_classes, device
            )
            ###########################

            # Accumulate metrics
            total_precision += torch.tensor(batch_precision, device=device)
            total_recall += torch.tensor(batch_recall, device=device)
            total_f1 += torch.tensor(batch_f1, device=device)
            total_class_ious += torch.tensor(batch_class_ious, device=device)

            num_batches += 1

    # Calculate average metrics
    avg_val_loss = val_loss / len(val_loader)
    avg_precision = total_precision / num_batches
    avg_recall = total_recall / num_batches
    avg_f1 = total_f1 / num_batches
    avg_class_ious = total_class_ious / num_batches
    mean_iou = avg_class_ious.mean().item()

    # Log detailed metrics
    if epoch is not None:
        print(f"Logging metrics for epoch {epoch + 1}...")
        log.info(f"Epoch {epoch + 1} Validation Results:")
        log.info(f"  Loss: {avg_val_loss:.4f}")
        log.info(f"  Mean IoU: {mean_iou:.4f}")
        log.info(f"  Mean Precision: {avg_precision.mean().item():.4f}")
        log.info(f"  Mean Recall: {avg_recall.mean().item():.4f}")
        log.info(f"  Mean F1: {avg_f1.mean().item():.4f}")
        print("Metrics logged.")
        # Log class-wise metrics if class names are available
        if hasattr(cfg, "class_names") and cfg.class_names:
            for i, class_name in enumerate(cfg.class_names):
                log.info(
                    f"  {class_name} - IoU: {avg_class_ious[i]:.4f}, "
                    f"Precision: {avg_precision[i]:.4f}, "
                    f"Recall: {avg_recall[i]:.4f}, F1: {avg_f1[i]:.4f}"
                )
        print("Class-wise metrics logged if available.")

    metrics = {
        "val_loss": avg_val_loss,
        "val_iou": mean_iou,
        "mean_precision": avg_precision.mean().item(),
        "mean_recall": avg_recall.mean().item(),
        "mean_f1": avg_f1.mean().item(),
        "class_ious": avg_class_ious.cpu().numpy(),
        "precision_per_class": avg_precision.cpu().numpy(),
        "recall_per_class": avg_recall.cpu().numpy(),
        "f1_per_class": avg_f1.cpu().numpy(),
    }
    print("Returning metrics dictionary.")

    if cfg.get("use_wandb", False):
        wandb.log({
            "val_loss": avg_val_loss,
            "val_iou": mean_iou,
            "val_precision": avg_precision.mean().item(),
            "val_recall": avg_recall.mean().item(),
            "val_f1": avg_f1.mean().item()
        }, step=epoch)

    return metrics
