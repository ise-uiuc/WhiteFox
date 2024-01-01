
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, bias):
        v1 = torch.matmul(x1, x2)
        x2 = x1
        x1 = v1 + x2
        return torch.add(x1, bias)
# Inputs to the model
x1 = torch.randn(3, 3, requires_grad=True)
x2 = torch.randn(3, 3)
bias = torch.randn(3, 3, requires_grad=True)
