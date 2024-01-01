t1 = conv(input_tensor) # Apply pointwise convolution with kernel size 1 to the input tensor
t2 = t1 - other # Subtract a tensor or scalar "other" from the output of the convolution
t3 = relu(t2) # Apply the ReLU (Rectified Linear Unit) activation function to the result
