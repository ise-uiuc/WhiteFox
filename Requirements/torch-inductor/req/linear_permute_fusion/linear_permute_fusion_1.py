t1 = torch.nn.functional.linear(input_tensor, ...) # Apply linear transformation to the input tensor.
t2 = t1.permute(...) # Permute the output tensor from the linear transformation.
