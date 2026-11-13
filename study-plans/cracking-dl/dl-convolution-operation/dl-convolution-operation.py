import numpy as np

def conv2d(input: list, filters: list, bias=None, padding: int = 0, stride: int = 1) -> list:
    """
    Returns the channel-first cross-correlation output.
    """
    value = np.asarray(input, dtype = np.float64)
    kernal = np.asarray(filters, dtype = np.float64)

    if padding !=0 :
        value = np.pad(value,pad_width = ((0,0),(padding,padding),(padding,padding)), mode="constant", constant_values = 0)
    input_channel, height, width = value.shape
    kernal_number, kernal_channel, kernal_height, kernal_width = kernal.shape
    output_h = (height-kernal_height)//stride+1
    output_w = (width-kernal_width)//stride+1
    output = np.empty((kernal_number,output_h, output_w), dtype = np.float64)
    for kn in range(kernal_number):
        for row in range(output_h):
            for column in range(output_w):
                    window = value[:,row*stride:row*stride+kernal_height, column*stride:column*stride+kernal_width]
                    kn_window = kernal[kn]
                    output[kn,row,column] = np.sum(window*kn_window)
    if bias is not None:
        output = output+bias
                    
    return np.round(output, 4).tolist()
