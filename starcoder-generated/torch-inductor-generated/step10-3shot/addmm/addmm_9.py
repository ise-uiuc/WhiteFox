
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp):
        v1 = torch.matmul(x1, x2)
        v2 = v1 + inp
        return v2
# Inputs to the model
x1 = torch.randn(666, 3, 1)
x2 = torch.randn(1, 1)
inp = torch.randn(666, 3, 3)
