
class Model(torch.nn.Module):
    def __init__(self, dropout_p, scale_factor):
        super().__init__()
        self.dropout_p = dropout_p
        self.softmax = torch.nn.Softmax(dim=-1)
        self.dropout = torch.nn.Dropout2d(p=dropout_p)
        self.scale_factor = scale_factor
 
    def forward(self, x1, x2):
        b, n, c, _ = x1.size()
        v1 = torch.matmul(x1, x2.transpose(2, 3)) # Compute the dot product of the query and key tensors
        v2 = v1.mul(self.scale_factor) # Scale the dot product by a factor
        v3 = self.softmax(v2) # Apply softmax to the scaled dot product
        v4 = self.dropout(v3) # Apply dropout to the softmax output
        v5 = torch.matmul(v4, x2) # Compute the dot product of the dropout output and the value tensor
        return v5, x2

# Initializing the model
m = Model(dropout_p = 0., scale_factor=0.5)

# Inputs to the model
x1 = torch.randn(13, 16, 512, 64)
x2 = torch.randn(13, 16, 64, 64)
__output__, __hidden__ = m(x1, x2)

