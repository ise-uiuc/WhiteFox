t1 = conv(input_tensor) # Apply pointwise convolution with kernel size 1 to the input tensor
t2 = sigmoid(t1) # Apply the sigmoid function to the output of the convolution
t3 = t1 * t2 # Multiply the output of the convolution by the output of the sigmoid function
