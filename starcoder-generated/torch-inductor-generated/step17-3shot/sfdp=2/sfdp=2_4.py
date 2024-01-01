
class Model(torch.nn.Module):
    def __init__(self, *, q, k, v, n_head, scale_factor, dropout_p):
        super().__init__()
        self.scale_factor = scale_factor
        self.dropout = torch.nn.Dropout(self.dropout_p)
        self.softmax = torch.nn.Softmax(dim=-1)
 
    def forward(self, *, q, k, v):
        qk = q.matmul(k.transpose(-2, -1)) # Compute the dot product of the query and the key
        scaled_qk = qk.div(self.scale_factor) # Scale the dot product by the inverse scale factor
        softmax_qk = self.softmax(scaled_qk) # Apply softmax to the scaled dot product
        dropout_qk = self.dropout(softmax_qk) # Apply dropout to the softmax output
        output = dropout_qk.matmul(v) # Compute the dot product of the dropout output and the value
        return output

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(1, 4, 16)
k = torch.randn(2, 4, 16)
v = torch.randn(1, 4, 128)
