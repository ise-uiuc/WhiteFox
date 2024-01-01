
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v2 = torch.abs(v2)
        v3 = torch.nn.functional.relu(v2)
        v2 = torch.nn.functional.gelu(v3)
        v4 = v2 + x1
        c1 = torch.nn.functional.celu(x1, alpha=-1.0)
        c2 = torch.nn.functional.celu(x1, alpha=1.0)
        c3 = torch.nn.functional.hardshrink(x1, lambd=3)
        v2 = v2.permute(0, 2, 1)
        return v2 + v4 + c1 + c2 + c3
# Inputs to the model
x1 = torch.randn(1, 2, 2)
