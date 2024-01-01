
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)
        self.view = torch.reshape
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        x2 = torch.nn.functional.relu(v2)
        v3 = x2.detach()
        v4 = x2.detach()
        v5 = self.view(v3, (1, 4))
        v6 = self.view(v4, (1, 4))
        return torch.cat([v6, v5], dim=1)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
