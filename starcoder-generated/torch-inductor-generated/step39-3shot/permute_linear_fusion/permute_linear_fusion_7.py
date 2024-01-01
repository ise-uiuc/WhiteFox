
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 2)
        self.linear2 = torch.nn.Linear(2, 1)
    def forward(self, x1):
        x2 = torch.nn.functional.linear(x1, self.linear1.weight, self.linear1.bias)
        x2 = torch.nn.functional.relu(x2)
        x2 += 1
        x2 = torch.nn.functional.relu(x2)
        v1 = torch.max(x2, dim=-1)[0]
        v2 = 2.0
        v3 = v2 + 1.0
        v2 = v1 + v3.to(v1.dtype)
        return torch.nn.functional.linear(v2.unsqueeze(dim=-1), self.linear2.weight, self.linear2.bias)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
