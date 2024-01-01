
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.key = torch.nn.Linear(4, 4)
 
    def forward(self, query, value1, value2, dropout_p):
        k = self.key(query)
        # Compute the dot product of the query and key tensors
        scaled_qk = torch.matmul(query, k.transpose(-2, -1)) \
                   .div(self.scale_factor) # Scale the dot product by the scale factor
        softmax_qk = scaled_qk.softmax(dim=-1) # Apply softmax to the scaled dot product
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p) # Apply dropout to the softmax output
        # Compute the dot product of the dropout output and the value tensor
        return torch.matmul(dropout_qk, value1) + torch.matmul(softmax_qk.sub(dropout_qk), value2)

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(2, 4)
value1 = torch.randn(2, 2, 4)
value2 = torch.randn(2, 2, 4)
dropout_p = 0.5
