import torch

def compute_gradient(values: torch.Tensor) -> torch.Tensor:
    """
    Returns a float32 gradient tensor with the same shape as values.
    """
    values.requires_grad=True
    y = values**3 + 2 * values
    loss = y.sum()
    loss.backward()
    
    return values.grad
