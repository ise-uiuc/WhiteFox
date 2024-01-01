
class Model(torch.nn.Module):
    def __init__(self, query, key, value, scale_factor):
        super().__init__()
        self.dropout_p = 0.1
 
    def forward(self, mask):
        # Compute the dot product of the query and key tensors
        qk = torch.matmul(query, key.transpose(-2, -1))
 
        # Scale the dot product by a factor
        scaled_qk = qk.mul(self.scale_factor)
 
        # Apply softmax to the scaled dot product
        softmax_qk = scaled_qk.softmax(dim=-1)
 
        # Apply dropout to the softmax output
        dropout_qk = torch.nn.functional.dropout(softmax_qk, self.dropout_p)
 
        # Compute the dot product of the dropout output and the value tensor
        output = dropout_qk.matmul(value)
 
        # Return the dot product
        return output, dropout_qk

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 1, 2)
key = torch.randn(1, 2, 2)
value = torch.randn(1, 2, 8)
mask = torch.ones([1, 1, 1])
