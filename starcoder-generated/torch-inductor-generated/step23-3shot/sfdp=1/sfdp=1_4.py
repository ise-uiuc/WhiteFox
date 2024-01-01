
class Model(torch.nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, dropout_p):
        super().__init__()
        self.query = torch.nn.Parameter(torch.randn(1, query_dim))
        self.key = torch.nn.Parameter(torch.randn(1, key_dim))
        self.value = torch.nn.Parameter(torch.randn(1, value_dim))
        self.dropout = 0.3
 
    def forward(self, x1, x2, inv_scale_factor):
        qk = torch.matmul(query, key.transpose(-2, -1)) # Compute the dot product of the query and key tensors
        scaled_qk = qk.div(inv_scale_factor) # Scale the dot product by the inverse scale factor
        softmax_qk = scaled_qk.softmax(dim=-1) # Apply softmax to the scaled dot product
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p) # Apply dropout to the softmax output
        output = dropout_qk.matmul(value) # Compute the dot product of the dropout output and the value tensor
        return output

# Initializing the model
query_dim = 128
key_dim = 128
value_dim= 512
m = Model(query_dim, key_dim, value_dim)

# Inputs to the model
x1 = torch.randn(1, 20, query_dim)
x2 = torch.randn(1, 30, key_dim)
inv_scale_factor = torch.randn(1, query_dim, 1)
