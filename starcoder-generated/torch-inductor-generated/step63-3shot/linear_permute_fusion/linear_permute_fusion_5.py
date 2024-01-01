
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)
    def forward(self, x0, x1):
        v2 = torch.nn.functional.linear(x1 + x0, self.linear.weight, self.linear.bias)
        v3 = x1.permute(0, 2, 1)
        v4 = x1 - v3
        v1 = torch.nn.functional.linear(v4 + v3, self.linear.weight, self.linear.bias)
        return v2 + v1
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
