
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, weight, bias):
        linear = torch.nn.Linear(2, 1)
        linear.weight = torch.nn.Parameter(weight, requires_grad=True)
        linear.bias = torch.nn.Parameter(bias, requires_grad=True)
        x1_ = linear(x1)
        x2_ = linear(x2)
        return x1_ + x2_
# Inputs to the model
x1 = torch.randn(2, 1, requires_grad=True)
x2 = torch.randn(2, 1)
weight = torch.randn(1, 2, requires_grad=True)
bias = torch.randn(1, requires_grad=True)
