
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        x2 = torch.nn.functional.relu(v2)
        v3 = x2 + v1
        v5 = torch.max(v3, dim=1)[-1]
        v4 = torch.ones_like(v5)
        v3 = v4 * v5 * v3
        return v3
# Inputs to the model
x1 = torch.randn(1, 2, 2)
