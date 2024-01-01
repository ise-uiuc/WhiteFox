
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 2)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v1 = v1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v1 = v1.permute(0, 2, 1)
        v3 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        x2 = torch.nn.functional.relu(v2)
        v4 = torch.max(x2, dim=-1)[0]
        v5 = v4.unsqueeze(dim=-1)
        v4 = v4 + v5.to(v4.dtype)
        v1 = v1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v6 = torch.max(v2, dim=-1)[0]
        v4 = v4 + v6.unsqueeze(dim=-1).to(v4.dtype)
        v7 = torch.max(v4, dim=-1)[0]
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 3)
