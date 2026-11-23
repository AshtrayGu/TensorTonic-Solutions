import torch

def activate(x: torch.Tensor, method: str = "relu") -> torch.Tensor:
    """
    Returns a float32 tensor with the same shape as x.
    """
    if method =="relu":
        activation = torch.nn.ReLU()
    elif method == "sigmoid":
        activation = torch.nn.Sigmoid()
    elif method == "tanh":
        activation = torch.nn.Tanh()
    elif method == "leaky_relu":
        activation = torch.nn.LeakyReLU()
    return activation(x)
        
    

