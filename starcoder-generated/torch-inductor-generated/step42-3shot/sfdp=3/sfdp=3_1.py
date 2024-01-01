
class Model(torch.nn.Module):
    def __init__(self, scale_factor=None, dropout_p=0):
        super().__init__()
        self.scale_factor = scale_factor
        self.dropout_p = dropout_p

    def forward(self, query, key, value):
        # Compute the dot product of the query and key tensors
        qk = torch.matmul(query, key.transpose(-2, -1))
        # Scale the dot product
        if (self.scale_factor!= None):
            qk = qk.mul(self.scale_factor)
        # Apply softmax to the scaled dot product
        softmax_qk = qk.softmax(dim=-1)
        # Apply dropout to the softmax output
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        # Compute the dot product of the dropout output and the value tensor
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model(scale_factor=1, dropout_p=0.1)

# Inputs to the model
x1 = torch.randn(10, 32, 64)
x2 = torch.randn(10, 32, 64)
x3 = torch.randn(10, 64, 32)
