t1 = conv_transpose(input_tensor) # Apply pointwise transposed convolution to the input tensor
t2 = torch.sigmoid(t1) # Apply the sigmoid function to the output of the transposed convolution
