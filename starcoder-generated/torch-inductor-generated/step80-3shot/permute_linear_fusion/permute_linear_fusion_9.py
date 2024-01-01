
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 2)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v9 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v6 = x1.permute(0, 2, 1)
        v7 = torch.max(v6, dim=-1)[0]
        v7 = v7.unsqueeze(dim=-1)
        v8 = v7.to(v7.dtype)
        v10 = (v8 == -1).to(v6.dtype)
        v7 = v8 + v10
        v8 = v9 + v7
        v6 = v6.permute(0, 2, 1)
        v7 = torch.max(v6, dim=-1)[0]
        v6 = torch.nn.functional.linear(v6, self.linear.weight, self.linear.bias)
        x2 = torch.nn.functional.relu(v7)
        return torch.max(v6, dim=-1)[0]
# Inputs to the model
x1 = torch.randn(1, 3, 3)
