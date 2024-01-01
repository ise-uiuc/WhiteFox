
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.linear_1 = torch.nn.Linear(2, 3)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v3 = torch.max(v2, dim=-1)[0]
        v1 = v1.permute(0, 2, 1)
        v4 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v5 = torch.max(v4, dim=-1)[0]
        v2 = v2.permute(0, 2, 1)
        v6 = torch.nn.functional.relu(v5)
        v4 = torch.nn.functional.relu(v3)
        v7 = torch.max(v6, dim=-1)[0]
        v5 = v5.permute(0, 2, 1)
        v8 = torch.nn.functional.linear(v5, self.linear.weight, self.linear.bias)
        v9 = torch.max(v8, dim=-1)[0]
        v6 = torch.nn.functional.relu(v9)
        v10 = torch.max(v6, dim=-1)[0]
        return v10
# Inputs to the model
x1 = torch.randn(1, 2, 2)
