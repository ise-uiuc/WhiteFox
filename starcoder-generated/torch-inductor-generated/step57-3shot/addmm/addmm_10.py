
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, x2):
        x1 = x ** 2
        v1 = x2 ** 2
        v2 = v1.t()
        return torch.mm(x1, v2 + x)
# Inputs to the model
x = torch.randn(3, 3)
x2 = torch.randn(3, 3)
