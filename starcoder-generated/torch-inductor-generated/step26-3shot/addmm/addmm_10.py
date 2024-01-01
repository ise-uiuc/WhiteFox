
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp):
        v1 = x1.t() @ inp
        v2 = v1 + x1
        v3 = x2 @ v2
        v4 = v3.t() @ v3
        return x1 @ v4
# Inputs to the model
x1 = torch.randn(3, 3, requires_grad=True)
x2 = torch.randn(3, 3, requires_grad=True)
inp = torch.randn(3, 3)
