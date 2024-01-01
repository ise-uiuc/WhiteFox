
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout()
 
    def forward(self, q, k, v, scale_factor, dropout_p):
        x = torch.matmul(q, k.transpose(-2, -1))
        x = x.mul(scale_factor)
        x = self.dropout(x, p=dropout_p)
        output = torch.matmul(x, v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(1, 32, 64)
k = torch.randn(1, 32, 64)
v = torch.randn(1, 32, 64)
scale_factor = 20.0
dropout_p = 0.1
