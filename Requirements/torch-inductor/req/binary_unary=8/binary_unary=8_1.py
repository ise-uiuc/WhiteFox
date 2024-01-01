t1 = conv(input_tensor) # Apply pointwise convolution with kernel size 1 to the input tensor
t2 = t1 + other # Add another tensor to the output of the convolution
t3 = torch.relu(t2) # Apply the ReLU activation function to the result
