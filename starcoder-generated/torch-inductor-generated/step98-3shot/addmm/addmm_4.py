
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, x3):
        v = x2 * x1
        v1 = x2 * x3
        v2 = v + x3
        v3 = v1 * v
        v4 = v1 + v2
        return v4
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3)
x3 = torch.randn(3, 3, requires_grad=True)
