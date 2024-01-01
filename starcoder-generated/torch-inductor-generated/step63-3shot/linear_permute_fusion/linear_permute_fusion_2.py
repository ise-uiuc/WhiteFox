
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 2)
        self.linear2 = torch.nn.Linear(2, 1)
    def forward(self, x0, x1):
        v0 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        v1 = torch.nn.functional.linear(x0, self.linear.weight, self.linear.bias)
        v2 = v0.permute(0, 2, 1) + v1.permute(0, 2, 1)
        v4 = x1 - 2 * v0
        v3 = torch.nn.functional.linear(x1 - 2 * v0, self.linear.weight, self.linear.bias)
        return v3
# Inputs to the model
x0 = torch.randn(1, 2, 2)
x1 = torch.randn(1, 2, 2)
