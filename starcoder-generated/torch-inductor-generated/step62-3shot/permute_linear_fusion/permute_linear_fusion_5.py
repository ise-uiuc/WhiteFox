
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.ReLU = torch.nn.ReLU6()
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        x2 = x1 + v2
        v3 = torch.sum(self.linear.bias)
        v3 = x1 * self.linear.weight
        v4 = self.ReLU(v3)
        v5 = v3 * 2
        return torch.nn.functional.hardtanh(v5, -3.0, 3.0) * x2
# Inputs to the model
x1 = torch.randn(1, 2, 2)
