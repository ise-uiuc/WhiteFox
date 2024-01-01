
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 4)
    def forward(self, x0, x1):
        v0 = x0
        v1 = x1
        v2 = torch.nn.functional.linear(v0, self.linear.weight, self.linear.bias)
        v3 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v4 = v3.permute(0, 2, 1)
        v5 = v2.permute(0, 2, 1)
        return v4 + v5
# Inputs to the model
x0 = torch.randn(1, 3, 3)
x1 = torch.randn(1, 4, 3)
