from monai.losses import DiceLoss, DiceCELoss, FocalLoss
import torch
import torch.nn as nn
import torch.nn.functional as F

class HEDUNetLoss(nn.Module):
    """
    BCEWithLogits over 2 classes with deep supervision.
    - Targets can be (B,H,W) integer labels or (B,2,H,W) one-hot.
    - Side outputs can be nested lists/tuples; we pick tensors and match shapes.
    - For 1-channel side outputs, we compare against the foreground mask.
    """
    def __init__(self, side_weight=0.5, num_classes=2, debug_once=True):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.side_weight = side_weight
        self.num_classes = num_classes
        self._debug_once = debug_once

    def _normalize_targets(self, targets):
        # Squeeze stray singleton channel if present
        if targets.dim() == 4 and targets.size(1) == 1:
            targets = targets.squeeze(1)

        # Branch: indices vs one-hot
        if targets.dim() == 3:
            if targets.dtype != torch.long:
                targets = targets.long()
            t_oh = F.one_hot(targets, num_classes=self.num_classes)  # (B,H,W,C)
            t_oh = t_oh.permute(0, 3, 1, 2).contiguous().float()     # (B,C,H,W)
        elif targets.dim() == 4 and targets.size(1) == self.num_classes:
            t_oh = targets.float()
        else:
            raise AssertionError(
                f"Targets must be (B,H,W) or (B,{self.num_classes},H,W), got {tuple(targets.shape)}"
            )

        # Foreground single-channel target for 1-channel side outputs
        t_fg = t_oh[:, 1:2, :, :]  # assume class 1 is foreground (ice)
        return t_oh, t_fg

    def _flatten_side_outputs(self, side_outs):
        tensors = []
        for s in side_outs:
            if isinstance(s, torch.Tensor):
                tensors.append(s)
            elif isinstance(s, (list, tuple)):
                for x in s:
                    if isinstance(x, torch.Tensor):
                        tensors.append(x)
        return tensors

    def _match_spatial(self, pred, ref_spatial):
        if pred.shape[2:] != ref_spatial:
            pred = F.interpolate(pred, size=ref_spatial, mode="bilinear", align_corners=False)
        return pred

    def forward(self, outputs, targets):
        main_out = outputs[0]
        side_outs = outputs[1:]

        if self._debug_once:
            print(f"[HEDUNetLoss] incoming main_out shape: {tuple(main_out.shape)}")
            print(f"[HEDUNetLoss] incoming targets shape: {tuple(targets.shape)}")
            self._debug_once = False

        # Normalize targets
        targets_oh, targets_fg = self._normalize_targets(targets)

        # Main loss: expect (B,2,H,W)
        main_out = self._match_spatial(main_out, targets_oh.shape[2:])
        if main_out.size(1) == self.num_classes:
            loss_main = self.bce(main_out, targets_oh)
        elif main_out.size(1) == 1:
            # If main is 1-channel, compare to foreground only
            loss_main = self.bce(main_out, targets_fg)
        else:
            raise AssertionError(f"Unexpected main_out channels: {main_out.size(1)}")

        # Side losses: flatten, match spatial, choose target per channel count
        side_tensors = self._flatten_side_outputs(side_outs)
        side_losses = []
        for side in side_tensors:
            side = self._match_spatial(side, targets_oh.shape[2:])
            if side.size(1) == self.num_classes:
                side_losses.append(self.bce(side, targets_oh))
            elif side.size(1) == 1:
                side_losses.append(self.bce(side, targets_fg))
            else:
                # Skip side outputs with incompatible channel count
                continue

        loss_sides = torch.stack(side_losses).mean() if side_losses else torch.tensor(0.0, device=main_out.device)

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
