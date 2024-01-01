
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, x3, bias):
        v1 = torch.mm(x1, x2)
        v2 = v1 + bias
        x1 = v2
        v1 = torch.mm(v2, x3)
        return v2
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3)
x3 = torch.randn(3, 3, requires_grad=True)
bias = torch.randn(3, 3, requires_grad=True)
