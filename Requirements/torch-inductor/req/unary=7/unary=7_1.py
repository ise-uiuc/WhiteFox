l1 = linear(input_tensor) # Apply linear transformation to the input tensor
l2 = l1 * clamp(min=0, max=6, l1 + 3) # Multiply the output of the linear transformation by the clamped output of the linear transformation added with 3
l3 = l2 / 6 # Divide the output of the multiplication by 6
