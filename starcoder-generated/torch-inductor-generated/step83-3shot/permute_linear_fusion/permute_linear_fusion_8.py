
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 2)
        self.linear2 = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.relu(v1)
        v3 = torch.nn.functional.linear(v2, self.linear1.weight, self.linear1.bias)
        v4 = torch.nn.functional.relu(v3)
        v5 = torch.nn.functional.linear(v4, self.linear2.weight)
        v6 = torch.nn.functional.relu(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 2, 2)
