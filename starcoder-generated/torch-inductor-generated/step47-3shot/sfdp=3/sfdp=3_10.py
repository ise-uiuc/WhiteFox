
_query = torch.randn(2, 4, 10) # Query tensor
_key = torch.randn(2, 10, 20) # Key tensor
_value = torch.randn(2, 10, 20) # Value tensor
with nn.utils.weight_norm(_query) as wn_query: # Perform weighted normalization on the query tensor
    q_normalized = nn.utils.weight_norm(_query) 
with nn.utils.weight_norm(_key) as wn_key: # Perform weighted normalization on the key tensor
    k_normalized = nn.utils.weight_norm(_key)
with nn.utils.weight_norm(_value) as wn_value: # Perform weighted normalization on the value tensor
    v_normalized = nn.utils.weight_norm(_value)
scale_factor = 1 / math.sqrt(20) # Scale factor
dropout_p = 0.1 # Dropout probability
qk = torch.matmul(_query, _key.transpose(-2, -1)) # Compute the dot product of the query and key tensors
scaled_qk = qk.mul(scale_factor) # Scale the dot product by a factor
softmax_qk = scaled_qk.softmax(dim=-1) # Apply softmax to the scaled dot product
dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p) # Apply dropout to the softmax output
output = dropout_qk.matmul(_value) # Compute the dot product of the dropout output and the value tensor

# Initializing the model
m = Model()

# Inputs to the model
