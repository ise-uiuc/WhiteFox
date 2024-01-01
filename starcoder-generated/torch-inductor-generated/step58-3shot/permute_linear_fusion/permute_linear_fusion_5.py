
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 4)
        self.linear2 = torch.nn.Linear(4, 8)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear1.weight, self.linear1.bias)
        x1 = torch.nn.functional.relu(v2)
        x1 = torch.nn.functional.gelu(x1)
        x1 = x1.permute(0, 2, 1)
        x2 = torch.nn.functional.linear(x1, self.linear2.weight, self.linear2.bias)
        x3 = torch.nn.functional.relu(x2)
        return torch.add(x1, x3)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
