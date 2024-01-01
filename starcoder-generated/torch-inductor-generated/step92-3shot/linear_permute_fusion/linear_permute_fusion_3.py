
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear2 = torch.nn.Linear(1, 2, False)
        self.linear = torch.nn.Linear(1, 2, False)
        self.relu = torch.nn.ReLU()
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        v2 = v1.permute(0, 2, 1)
        v3 = v1.permute(0, 1, 3, 2)
        v4 = torch.nn.functional.linear(v2, self.linear2.weight)
        v5 = v3 - v4
        v6 = torch.nn.functional.linear(v5, self.linear2.weight.permute(0, 2, 1))
        return torch.nn.functional.linear(torch.nn.functional.linear(torch.nn.functional.linear(v6, self.linear2.weight + 0.25, 2.41), self.linear2.weight + 0.25, -0.46), self.linear2.weight + 0.25) - self.linear2.weight
# Inputs to the model
x1 = torch.randn(1, 1, 1)
