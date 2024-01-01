t1 = linear(input_tensor) # Apply a linear transformation to the input tensor
t2 = torch.clamp_min(t1, min_value) # Clamp the output of the linear transformation to a minimum value
t3 = torch.clamp_max(t2, max_value) # Clamp the output of the previous operation to a maximum value
