
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(24, 1)
        self.linear2 = torch.nn.Linear(1, 24)
        self.relu = torch.nn.ReLU()
    def forward(self, x1):
        v1 = x1.permute(1, 0, 2)
        v2 = torch.nn.functional.linear(v1, self.linear1.weight)
        v3 = v2.permute(1, 0, 2)
        v4 = torch.nn.functional.linear(v3, self.linear2.weight)
        v3 = self.relu(v4)
        x2 = torch.max(v3.flatten(start_dim=1), dim=1)[0]
        return x2
# Inputs to the model
x1 = torch.randn(2, 2, 6)
