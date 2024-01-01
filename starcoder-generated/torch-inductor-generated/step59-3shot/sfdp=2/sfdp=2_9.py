
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_factor = math.sqrt(self.head_size)
 
    def forward(self, query, key, value, inv_scale_factor, dropout_p):
        qk = torch.matmul(query, key.transpose(-2, -1)) # Compute the dot product of the query and the key
        scaled_qk = qk.div(inv_scale_factor) # Scale the dot product by the inverse scale factor
        softmax_qk = scaled_qk.softmax(dim=-1) # Apply softmax to the scaled dot product
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p) # Apply dropout to the softmax output
        output = torch.matmul(dropout_qk, value) # Compute the dot product of the dropout output and the value
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 32, 8, 64)
key = torch.randn(1, 32, 8, 64)
value = torch.randn(1, 32, 8, 64)
inv_scale_factor = torch.full([1], 1.0 / 8)
dropout_p = torch.full([1], 0.0, dtype=torch.float32)
