
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, query, key, value, scale_factor, dropout_p):
        v1 = torch.matmul(query, key.transpose(-2, -1)) # Compute the dot product of the query and key tensors
        v2 = v1 * scale_factor # Scale the dot product by a factor
        v3 = v2.softmax(dim=-1) # Apply softmax to the scaled dot product
        v4 = torch.nn.functional.dropout(v3, p=dropout_p) # Apply dropout to the softmax output
        v5 = torch.matmul(v4, value) # Compute the dot product of the dropout output and the value tensor
        return v5

# Inputs to the model
query = torch.randn(1, 5, 128)
key = torch.randn(1, 6, 128)
value = torch.randn(1, 6, 128)
scale_factor = torch.tensor(0.7, dtype=torch.float32)
dropout_p = torch.tensor(0.5, dtype=torch.float32)
