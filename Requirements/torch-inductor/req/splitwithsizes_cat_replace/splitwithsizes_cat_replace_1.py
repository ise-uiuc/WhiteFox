split_tensors = torch.split(input_tensor, split_sizes, dim) # Split the input tensor into several tensors along a given dimension
concatenated_tensor = torch.cat([split_tensors[i] for i in range(len(split_sizes))], dim) # Concatenate the split tensors along the same dimension
