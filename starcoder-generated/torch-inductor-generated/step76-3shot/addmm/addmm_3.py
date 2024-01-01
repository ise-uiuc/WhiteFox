
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp):
        v1 = x1 + inp
        v2 = v1 * x2 # 'x2' is passed as an argument, not as a tensor
        v3 = 2 * x1 # 'x1' is passed as an argument, not as a tensor
        v4 = v3 + x2
        v5 = v1 + v4
        return v5
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3, requires_grad=True)
inp = torch.randn(3, 3)
