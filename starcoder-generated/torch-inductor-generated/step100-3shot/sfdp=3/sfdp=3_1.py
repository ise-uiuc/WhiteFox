 inputs
query = torch.randn(4, 5, 6) # Queries for the attention mechanism, which are also values that are multiplied by the output results of the attention mechanism. Here, the batch size is 4, the length of the sequences is 5, and the length of the queries is 6.
key = torch.randn(4, 7, 6) # Keys for the attention mechanism. Here, the batch size is 4, the length of the sequences is 7, and the length of the keys is 6.
value = torch.randn(4, 7, 8) # Values for the attention mechanism. Here, the batch size is 4, the length of the sequences is 7, and the length of the values is 8. The length of these tensors does not need to be the same as the length of the keys and queries tensors.
scale_factor = torch.rand((4, 1, 6)) # A constant factor used to scale the dot product. Here, the batch size is 4, the length of the input tensors is 6, and the size of the factor tensor is (4, 1, 6). The factor tensor represents a scalar for the different query-key pairs.
dropout_p = 0.3 # A dropout parameter used to randomly shut down a portion of the attention mask.

# Initializing the model
m = Model()

# Inputs to the model
