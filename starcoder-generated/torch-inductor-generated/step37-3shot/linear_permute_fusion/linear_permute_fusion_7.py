
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 3)
        self.linear2 = torch.nn.Linear(3, 4)
        self.relu = torch.nn.ReLU()
    def forward(self, x1):
        v4 = x1
        v1 = torch.nn.functional.linear(v4, self.linear1.weight, self.linear1.bias)
        v3 = v1
        v2 = v1.permute(0, 2, 1)
        v6 = self.relu(v3)
        v5 = v2
        v7 = torch.nn.functional.linear(v2, self.linear2.weight, self.linear2.bias)
        return torch.cat([v5, v6, v7], dim=1)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
