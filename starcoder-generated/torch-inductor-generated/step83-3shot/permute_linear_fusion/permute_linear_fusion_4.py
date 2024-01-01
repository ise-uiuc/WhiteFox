
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 3)
        self.linear_1 = torch.nn.Linear(3, 2)
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        v2 = torch.nn.functional.linear(v1, self.linear_1.weight, self.linear_1.bias)
        v3 = torch.argmax(v1, dim = -1)
        v5 = torch.sum(self.linear_1.weight, dim = 1)
        v4 = v2 - v3 - v5
        v6 = torch.abs(v4 * self.linear_1.weight)
        v7 = v3 * torch.tanh(v6)
        v8 = v4 / (v7 + 2)
        return v7
# Inputs to the model
x1 = torch.randn(1, 2)
