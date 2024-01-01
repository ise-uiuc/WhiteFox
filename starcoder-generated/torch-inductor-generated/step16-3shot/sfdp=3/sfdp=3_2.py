
class Model(torch.nn.Module):
    def __init__(self, query_dim: int, key_dim: int, value_dim: int):
        super().__init__()
        self.scale_factor = (key_dim ** -0.5)
        self.dropout_p = 0.1
 
    def forward(self, query, key, value):
        # Compute the dot product of the query and key tensors
        qk = torch.matmul(query, key.transpose(-2, -1))
 
        # Scale the dot product by a factor
        scaled_qk = qk * self.scale_factor
 
        # Apply softmax to the scaled dot product
        softmax_qk = scaled_qk.softmax(dim=-1)
 
        # Apply dropout to the softmax output
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
 
        # Compute the dot product of the dropout output and the value tensor
        output = dropout_qk.matmul(value)
 
        return output

# Initialize constants
query_dim, key_dim, value_dim = 16, 16, 16

# Initialize model object
m = Model(query_dim, key_dim, value_dim)

# Inputs to the model
query = torch.randn(1, 5, query_dim)
key = torch.zeros(1, 4, key_dim)
value = torch.zeros(1, 4, value_dim)
