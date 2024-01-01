
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2):
        v1 = torch.matmul(x1, x2.transpose(-2, -1))
        v2 = v1.div(1e-06)
        v3 = v2.softmax(dim=-1)
        v4 = torch.nn.functional.dropout(v3, p=0.5)
        out = v4.matmul(x2)
        return out

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(16, 50, 17)
x2 = torch.randn(16, 17, 64)
__output_size__ = m(x1, x2).size()
