
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v3 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v4 = torch.add(v2, x1)
        v4 = v3 * torch.relu(v4)
        v5 = torch.tanh(v4)
        v6 = v5 * v5
        v6 = v5 + v6
        v7 = v6.permute(0, 2, 1)
        return v7
# Inputs to the model
x1 = torch.randn(1, 2, 2)
