import torch

def create_tensor(method: str, shape: list, value: float = 0.0) -> torch.Tensor:
    """
    Returns a float32 tensor with the requested shape.
    """
    # torch.set_default_dtype(torch.float)
    # shape = torch.tensor(shape)
    if method=="zeros":
        return torch.zeros(shape)
    elif method=="ones":
        return torch.ones(shape)
    elif method =="full":
        return torch.full(shape,value,dtype=torch.float32)
        
    
