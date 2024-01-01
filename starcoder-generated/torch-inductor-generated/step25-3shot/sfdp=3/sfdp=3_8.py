
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3, x4):
        v1 = torch.matmul(x1, x2.transpose(-2, -1))
        v2 = v1.mul(0.5)
        v3 = v1.softmax(dim=-1)
        v4 = torch.nn.functional.dropout(v3, p = 0.1)
        v5 = torch.matmul(v4, x3)
        v6 = v5 + x4
        return v6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10, 20)
x2 = torch.randn(1, 20, 5)
x3 = torch.randn(1, 5, 10)
x4 = torch.randn(1, 10, 30)
