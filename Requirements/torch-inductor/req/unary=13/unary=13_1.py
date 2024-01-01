t1 = linear(input_tensor) # Apply a linear transformation to the input tensor
t2 = sigmoid(t1) # Apply the sigmoid function to the output of the linear transformation
t3 = t1 * t2 # Multiply the output of the linear transformation by the output of the sigmoid function
