t1 = conv(input_tensor) # Apply pointwise convolution with kernel size 1 to the input tensor
t2 = torch.sigmoid(t1) # Apply the sigmoid function to the output of the convolution
