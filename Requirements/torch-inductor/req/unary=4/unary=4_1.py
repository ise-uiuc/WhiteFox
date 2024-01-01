t1 = linear(input_tensor) # Apply pointwise linear transformation to the input tensor
t2 = t1 * 0.5 # Multiply the output of the linear transformation by 0.5
t3 = t1 * 0.7071067811865476 # Multiply the output of the linear transformation by 0.7071067811865476
t4 = torch.erf(t3) # Apply the error function to the output of the linear transformation
t5 = t4 + 1 # Add 1 to the output of the error function
t6 = t2 * t5 # Multiply the output of the linear transformation by the output of the error function
