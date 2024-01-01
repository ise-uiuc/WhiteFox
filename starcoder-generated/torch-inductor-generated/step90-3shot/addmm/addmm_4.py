
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = torch.einsum("bij, j", x1, x2)
        v2 = v1 + x1 + x2
        return x1**2
# Inputs to the model
x1 = torch.randn(3, 3, 3, requires_grad=True)
x2 = torch.randn(3, requires_grad=True)
