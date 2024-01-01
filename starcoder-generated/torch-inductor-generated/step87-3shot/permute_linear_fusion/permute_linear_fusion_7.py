
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 3)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v3 = v2.permute(0, 2, 1)
        v4 = torch.nn.functional.linear(v3, self.linear.weight)
        v5 = v3.permute(0, 2, 1)
        v6 = torch.nn.functional.linear(v5, self.linear.weight)
        x2 = torch.nn.functional.relu(v4 + v6)
        return x2
# Inputs to the model
x1 = torch.randn(1, 2, 3)
