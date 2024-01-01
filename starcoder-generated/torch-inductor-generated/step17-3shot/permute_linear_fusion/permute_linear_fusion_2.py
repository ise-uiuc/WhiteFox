
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        x2 = torch.nn.functional.relu(v1)
        v3 = torch.ones(v2)
        x1 = x1.permute(0, 2, 1)
        v1 = x1.permute(0, 2, 1)
        v4 = torch.sqrt(v1)
        v4 = torch.abs(v1)
        v3 = v3 * v4
        v3 = v3 * 5
        return v3
# Inputs to the model
x1 = torch.randn(1, 2, 2)
