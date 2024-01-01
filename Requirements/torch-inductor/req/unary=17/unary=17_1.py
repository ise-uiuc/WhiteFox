t1 = conv_transpose(input_tensor) # Apply pointwise transposed convolution to the input tensor
t2 = relu(t1) # Apply the ReLU activation function to the output of the transposed convolution
