
class Model(torch.nn.Module):
    def __init__(self, num_heads, inv_scale_factor=1.0):
        super().__init__()
        self.num_heads = num_heads
        self.wq = torch.nn.parameter.Parameter(torch.randn(1, num_heads, 16, 16)) # Tensor with shape [1, num_heads, 16, 16].
        self.wk = torch.nn.parameter.Parameter(torch.randn(1, num_heads, 16, 16)) # Tensor with shape [1, num_heads, 16, 16].
        self.wv = torch.nn.parameter.Parameter(torch.randn(1, num_heads, 16, 16)) # Tensor with shape [1, num_heads, 16, 16].
        self.inv_scale_factor = inv_scale_factor
 
    def forward(self, query, key, value, dropout_p=0.0):
        qk = query.matmul(key.transpose(-1, -2)) # Compute the dot product of the query and key tensors
        scaled_qk = qk.div(self.inv_scale_factor) # Scale the dot product by the inverse scale factor
        softmax_qk = scaled_qk.softmax(dim=-1) # Apply softmax to the scaled dot product
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p) # Apply dropout to the softmax output
        output = dropout_qk.matmul(self.wv) # Compute the dot product of the dropout output and the value tensor
        return output

# Initializing the model with the specified parameters
m = Model(num_heads=4)

# Inputs to the model
query = torch.randn(1, 4, 16, 16)
key = torch.randn(1, 4, 16, 16)
value = torch.randn(1, 4, 16, 16)
