
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.linear2 = torch.nn.Linear(2, 1)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, torch.nn.functional.relu(self.linear.weight), torch.nn.functional.relu(self.linear.bias))
        v3 = torch.nn.functional.relu(v2)
        v4 = torch.sum(v3)
        return torch.max(v2) + v4
# Inputs to the model
x1 = torch.randn(1, 2, 2)
