t1 = linear(input_tensor) # Apply a linear transformation to the input tensor
t2 = t1 - other # Subtract 'other' from the output of the linear transformation
t3 = relu(t2) # Apply the ReLU activation function to the result
