from monai.losses import DiceLoss, DiceCELoss, FocalLoss
import torch
import torch.nn as nn
import torch.nn.functional as F

class HEDUNetLoss(nn.Module):
    """
    Loss für HED-UNet mit Deep Supervision:
    Hauptausgabe + Side-Outputs
    """
    def __init__(self, side_weight=0.5, num_classes=2):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.side_weight = side_weight
        self.num_classes = num_classes

    def forward(self, outputs, targets):
        # outputs = [main_output, side1, side2, ...]
        main_out = outputs[0]
        side_outs = outputs[1:]

        # Targets von (B,H,W) → (B,num_classes,H,W), Float
        targets_oh = F.one_hot(targets.long(), num_classes=self.num_classes)  # (B,H,W,C)
        targets_oh = targets_oh.permute(0, 3, 1, 2).float()                   # (B,C,H,W)

        loss_main = self.bce(main_out, targets_oh)
        if side_outs:
            loss_sides = sum(self.bce(side, targets_oh) for side in side_outs) / len(side_outs)
        else:
            loss_sides = 0.0

        return loss_main + self.side_weight * loss_sides


class CombinedLoss(nn.Module):
    def __init__(self, weights=None, dice_weight=0.5, focal_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.dice_loss = DiceLoss(
            include_background=True,
            to_onehot_y=False,
            softmax=True,
            squared_pred=False,
            smooth_nr=1e-5,
            smooth_dr=1e-5,
        )
        self.focal_loss = FocalLoss(
            include_background=True, to_onehot_y=False, gamma=2.0
        )
        self.weights = weights
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(self, y_pred, y_true):
        # Calculate losses
        dice_loss = self.dice_loss(y_pred, y_true)
        focal_loss = self.focal_loss(y_pred, y_true)

        # Apply class weights if provided
        if self.weights is not None:
            weights = self.weights.to(dice_loss.device)
            dice_loss = dice_loss * weights
            focal_loss = focal_loss * weights

        # Calculate weighted sum
        total_loss = (
            self.dice_weight * dice_loss.mean() + self.focal_weight * focal_loss.mean()
        )

        return total_loss
