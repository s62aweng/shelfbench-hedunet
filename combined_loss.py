from monai.losses import DiceLoss, DiceCELoss, FocalLoss
import torch
import torch.nn as nn
import torch.nn.functional as F

class HEDUNetLoss(nn.Module):
    """
    BCEWithLogits over 2 classes with deep supervision.
    Converts integer targets (B,H,W) to one-hot (B,2,H,W).
    """
    def __init__(self, side_weight=0.5, num_classes=2, debug_once=True):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.side_weight = side_weight
        self.num_classes = num_classes
        self._debug_once = debug_once

    def _ensure_label_shape(self, targets: torch.Tensor) -> torch.Tensor:
        # Remove any singleton dims beyond (B,H,W), e.g. (B,1,H,W), (B,H,W,1), (B,1,H,W,1)
        # Keep the batch dimension intact.
        while targets.dim() > 3:
            # Prefer removing trailing singleton first, then channel singleton
            if targets.size(-1) == 1:
                targets = targets.squeeze(-1)
            elif targets.size(1) == 1 and targets.dim() == 4:
                targets = targets.squeeze(1)
            else:
                # If there’s an unexpected extra dim, try a generic squeeze
                squeezed = targets.squeeze()
                # Restore batch dimension if squeeze collapsed it
                if squeezed.dim() == targets.dim() - 1:
                    targets = squeezed
                else:
                    break
        return targets

    def forward(self, outputs, targets):
        main_out = outputs[0]
        side_outs = outputs[1:]

        if self._debug_once:
            print(f"[HEDUNetLoss] incoming main_out shape: {tuple(main_out.shape)}")
            print(f"[HEDUNetLoss] incoming targets shape: {tuple(targets.shape)}")
            self._debug_once = False  # only print once

        # 1) Ensure targets are integer class indices and shape (B,H,W)
        if targets.dtype != torch.long:
            targets = targets.long()

        targets = self._ensure_label_shape(targets)
        assert targets.dim() == 3, f"Targets must be (B,H,W), got {tuple(targets.shape)}"

        # 2) One-hot to (B,C,H,W)
        targets_oh = F.one_hot(targets, num_classes=self.num_classes)  # (B,H,W,C)
        targets_oh = targets_oh.permute(0, 3, 1, 2).contiguous().float()  # (B,C,H,W)

        # 3) Compute loss
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
