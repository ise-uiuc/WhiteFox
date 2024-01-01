l1 = linear(input_tensor) # Apply a linear transformation to the input tensor
l2 = l1 + 3 # Add 3 to the output of the linear transformation
l3 = torch.clamp_min(l2, 0) # Clamp the output of the addition operation to a minimum of 0
l4 = torch.clamp_max(l3, 6) # Clamp the output of the previous operation to a maximum of 6
l5 = l4 / 6 # Divide the output of the previous operation by 6
