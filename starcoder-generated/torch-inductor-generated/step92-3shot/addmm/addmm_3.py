
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, v0):
        v1 = torch.mm(x1, x2)
        t1 = torch.tanh(v1)
        v2 = t1 + v1
        v3 = v1 + v1
        v4 = t1 + v1
        v5 = v1 + v1
        return v4 - t1
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3)
v0 = torch.randn(3, 3)
