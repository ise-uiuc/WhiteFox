qk = torch.matmul(query, key.transpose(-2, -1)) # Compute the dot product of the query and key tensors
scaled_qk = qk.div(inv_scale_factor) # Scale the dot product by the inverse scale factor
softmax_qk = scaled_qk.softmax(dim=-1) # Apply softmax to the scaled dot product
dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p) # Apply dropout to the softmax output
output = dropout_qk.matmul(value) # Compute the dot product of the dropout output and the value tensor
