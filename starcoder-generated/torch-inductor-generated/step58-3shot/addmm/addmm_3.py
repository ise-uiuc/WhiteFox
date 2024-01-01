
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, x3, inp):
        v1 = torch.mm(x1, x3)
        v2 = v1 + x1 + x3+x3
        return v2, x1, inputs
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3, requires_grad=True)
x3 = torch.randn(3, 3)
inp = torch.randn(3, 3, 1, 3)
