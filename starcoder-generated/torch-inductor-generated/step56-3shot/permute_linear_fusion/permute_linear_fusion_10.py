
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        x2 = torch.nn.functional.relu(v2)
        v3 = x2.detach()
        x3 = torch.nn.functional.relu(x2)
        v4 = torch.detach(v3)
        x4 = torch.max(x3, dim=-1)[0]
        return torch.sum(v4.permute(0, 2, 1)) + 0.5 / 24. * torch.sum(x4) + 1.7731
# Inputs to the model
x1 = torch.randn(1, 2, 2)
