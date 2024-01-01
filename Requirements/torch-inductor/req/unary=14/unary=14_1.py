t1 = conv_transpose(input_tensor) # Apply pointwise transposed convolution to the input tensor
t2 = sigmoid(t1) # Apply the sigmoid function to the output of the transposed convolution
t3 = t1 * t2 # Multiply the output of the transposed convolution by the output of the sigmoid function
