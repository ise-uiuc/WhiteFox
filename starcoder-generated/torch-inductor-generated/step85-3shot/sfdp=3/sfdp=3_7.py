
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q = torch.nn.Parameter(torch.randn(16, 1024, 64))
        self.v = torch.nn.Parameter(torch.randn(16, 1024, 128))
 
    def forward(self, query, value, scale_factor, dropout_p):
        qk = torch.matmul(query, self.v.transpose(-2, -1)) # Compute the dot product of the query and key tensors
        scaled_qk = qk.mul(scale_factor) # Scale the dot product by a factor
        softmax_qk = scaled_qk.softmax(dim=-1) # Apply softmax to the scaled dot product
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p) # Apply dropout to the softmax output
        output = torch.matmul(dropout_qk, self.v) # Compute the dot product of the dropout output and the value tensor
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(2, 16, 1024, 64)
value = torch.randn(2, 16, 1024, 128)
scale_factor = torch.randn(2, 16, 1, 1)
dropout_p = 0.2
