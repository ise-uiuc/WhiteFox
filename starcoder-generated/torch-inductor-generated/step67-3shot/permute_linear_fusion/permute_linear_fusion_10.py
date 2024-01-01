
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 2)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v1 = v1.permute(0, 2, 1)
        v3 = torch.max(v2, dim=-1)[0]
        v3 = torch.sqrt(torch.sum(torch.abs(v3), dim=1, keepdim=True))
        v2 = torch.max(v2, dim=-1)[0]
        v4 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v5 = (v2 - v3)
        v6 = torch.cat([v4, v5], dim=-1)
        v7 = v6.permute(0, 2, 1)
        v8 = torch.sum(torch.abs(v6), dim=-1)
        return torch.sigmoid(v8) + torch.nn.functional.linear(v7, self.linear.weight, self.linear.bias)
# Inputs to the model
x1 = torch.randn(1, 3, 3)
