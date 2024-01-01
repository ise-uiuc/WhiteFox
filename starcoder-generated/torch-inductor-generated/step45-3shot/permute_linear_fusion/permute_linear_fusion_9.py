
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 16)
    def forward(self, x1, x2):
        v1 = x1.permute(0, 2, 1)
        v2 = x2.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v2, self.linear.weight, self.linear.bias)
        return torch.stack([v1, v2])
# Inputs to the model
x1 = torch.randn(1, 32, 16)
x2 = torch.randn(1, 64, 16)
