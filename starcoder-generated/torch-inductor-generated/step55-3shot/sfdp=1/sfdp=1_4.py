
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3, x4, x5, x6):
        v2 = torch.matmul(x1, x2.transpose(-2, -1))
        v3 = v2.div(100)
        v4 = v3.softmax(dim=-1)
        v5 = torch.nn.functional.dropout(v4, p=0.5)
        v6 = v5.matmul(x3)
        v7 = torch.matmul(v6, x4.transpose(-2, -1))
        v8 = v7.div(100)
        v9 = v8.softmax(dim=-1)
        v10 = v9.matmul(x5)
        v11 = v10 + x6
        return v11

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(4, 24, 1000)
x2 = torch.randn(4, 24, 1000)
x3 = torch.randn(64, 24, 128)
x4 = torch.randn(64, 24, 1000)
x5 = torch.randn(64, 24, 128)
x6 = torch.randn(4, 24, 128)
__ouput__ = m(x1, x2, x3, x4, x5, x6)

