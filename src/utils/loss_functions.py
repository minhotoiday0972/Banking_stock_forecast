# src/utils/loss_functions.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Implements the Focal Loss function.

    Focal Loss was introduced in the paper "Focal Loss for Dense Object Detection" by Lin et al.
    It is designed to address class imbalance by down-weighting the loss assigned to well-classified examples.
    This allows the model to focus more on hard-to-classify examples.

    Args:
        alpha (float or list, optional): Weighting factor for each class. If a float, the same weight is applied to all classes. 
                                         If a list, it must have the same size as the number of classes. Defaults to 0.25.
        gamma (float, optional): Focusing parameter. Higher values give more weight to hard-to-classify examples. 
                                 Defaults to 2.0.
        reduction (str, optional): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. 
                                   'none': no reduction will be applied.
                                   'mean': the sum of the output will be divided by the number of elements in the output.
                                   'sum': the output will be summed.
                                   Defaults to 'mean'.
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Calculates the focal loss.

        Args:
            inputs (torch.Tensor): The model's predictions (logits), with shape (N, C) where N is the batch size 
                                   and C is the number of classes.
            targets (torch.Tensor): The ground truth labels, with shape (N,).

        Returns:
            torch.Tensor: The calculated focal loss.
        """
        # Calculate cross-entropy loss without reduction
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # Get the probabilities of the correct class
        pt = torch.exp(-ce_loss)

        # Prepare alpha tensor
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                # If alpha is a scalar, create a tensor of that value for all classes
                alpha_t = torch.tensor([float(self.alpha)] * inputs.size(1), device=inputs.device)
            elif isinstance(self.alpha, torch.Tensor):
                # If alpha is already a tensor, use it directly, ensuring it's on the correct device
                alpha_t = self.alpha.to(inputs.device)
            else:
                raise TypeError("self.alpha must be a float, int, None, or a torch.Tensor")
            
            # Select alpha for each example based on its target class
            alpha_t = alpha_t.gather(0, targets.data.view(-1))
        else:
            alpha_t = 1.0

        # Calculate focal loss
        focal_loss = alpha_t * (1 - pt)**self.gamma * ce_loss

        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
