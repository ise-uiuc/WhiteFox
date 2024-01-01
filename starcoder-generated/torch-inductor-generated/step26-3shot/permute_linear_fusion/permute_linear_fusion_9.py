
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(2, 1)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v4 = self.relu(v2)
        v3 = torch.max(v4, dim=-1)[0]
        v3 = v3.unsqueeze(dim=-1)
        v4 = v3 == -1
        v4 = torch.nn.functional.linear(v4, self.linear2.weight, self.linear2.bias)
        return v4
# Inputs to the model
x1 = torch.randn(1, 2, 2)
