import torch

class LabelSmoothingCTCLoss(torch.nn.Module):
    """
    CTC Loss with label smoothing for better generalization.
    
    Args:
        blank: Index of blank token (default: 0)
        smoothing: Smoothing parameter epsilon (default: 0.1)
        reduction: 'mean', 'sum', or 'none'
        zero_infinity: Whether to zero out infinite losses
    """
    def __init__(self, blank=0, smoothing=0.1, reduction='mean', zero_infinity=True):
        super().__init__()
        self.blank = blank
        self.smoothing = smoothing
        self.reduction = reduction
        self.zero_infinity = zero_infinity
        self.ctc_loss = torch.nn.CTCLoss(blank=blank, reduction='none', zero_infinity=zero_infinity)

    def forward(self, log_probs, targets, input_lengths, target_lengths):
        """
        Compute label-smoothed CTC loss.
        
        Args:
            log_probs: Log probabilities (T, N, C) where T=time, N=batch, C=classes
            targets: Target sequences (N, S) where S=target sequence length
            input_lengths: Lengths of input sequences (N,)
            target_lengths: Lengths of target sequences (N,)
        """
        # Standard CTC loss
        loss = self.ctc_loss(log_probs, targets, input_lengths, target_lengths)

        # Add label smoothing by mixing with uniform distribution
        if self.smoothing > 0:
            # Compute entropy of the predictions as regularization
            # Higher entropy = more uniform = less confident
            probs = torch.exp(log_probs)
            entropy = -(probs * log_probs).sum(dim=-1).mean() # last dimension = classes, sum across classes

            # Mix CTC loss with entropy regularization
            # (1 - smoothing) weight on correct predictions
            # smoothing weight on encouraging uncertainty
            loss = ((1 - self.smoothing) * loss) - (self.smoothing * entropy)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

def create_warmup_scheduler(optimizer, warmup_steps, total_steps, lr_start, lr_end):
    """
    Create learning rate scheduler with warmup + linear decay.
    
    Args:
        optimizer: PyTorch optimizer
        warmup_steps: Number of warmup steps (e.g., 500)
        total_steps: Total training steps (e.g., 10000)
        lr_start: Starting learning rate (e.g., 0.02)
        lr_end: Ending learning rate (e.g., 0.002)
    """
    def lr_lambda(step):
        if step < warmup_steps:
            # Warmup phase: linearly increase from 0 to 1
            # Step 0: 0/500 = 0.0 → LR = 0.02 × 0.0 = 0.0
            # Step 250: 250/500 = 0.5 → LR = 0.02 × 0.5 = 0.01
            # Step 500: 500/500 = 1.0 → LR = 0.02 × 1.0 = 0.02
            return step / warmup_steps
        else:
            # Decay phase: linearly decrease from 1.0 to (lr_end/lr_start)
            remaining_steps = total_steps - warmup_steps
            current_step = step - warmup_steps
            
            # Linear decay from 1.0 to (lr_end/lr_start)
            decay_factor = 1.0 - (current_step / remaining_steps) * (1.0 - lr_end / lr_start)
            return decay_factor
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)