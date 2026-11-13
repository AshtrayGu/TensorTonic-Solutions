import numpy as np

def pooling(input: list, pool_size: int, stride: int, pool_type: str) -> list:
    """
    Returns the pooled channel-first tensor.
    """
    value = np.asarray(input, dtype = np.float64)
    channel, height, width = value.shape
    out_h = (height-pool_size)//stride+1
    out_w = (width-pool_size)//stride+1
    output = np.empty((channel,out_h,out_w), dtype = np.float64)
    for channel in range(channel):
        for row in range(out_h):
            for column in range(out_w):
                window = value[channel,row*stride:row*stride+pool_size,column*stride:column*stride+pool_size]
                output[channel, row, column] = np.max(window) if pool_type=="max" else np.mean(window)
    return np.round(output, 4).tolist()