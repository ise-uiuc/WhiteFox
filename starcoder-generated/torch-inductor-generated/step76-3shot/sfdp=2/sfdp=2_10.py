
class Model(torch.nn.Module):
    def __init__(self, num_heads, dim, dropout_p, inv_scale_factor):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.inv_scale_factor = inv_scale_factor
        self.dropout_p = dropout_p
 
    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1)) # Compute the dot product of the query and the key
        scaled_qk = qk.div(self.inv_scale_factor) # Scale the dot product by the inverse scale factor
        softmax_qk = scaled_qk.softmax(dim=-1) # Apply softmax to the scaled dot product
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p) # Apply dropout to the softmax output
        output = dropout_qk.matmul(value) # Compute the dot product of the dropout output and the value
        return output

# Initializing the model
m = Model(8, 1536, 0.2, 1/math.sqrt(1536))

# Inputs to the model
query = torch.randn(16, 8, 1536)
key = torch.randn(16, 8, 1536)
value = torch.randn(16, 8, 1536)
