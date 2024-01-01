
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v5 = torch.randn_like(x1)
        v1 = torch.nn.functional.relu(x1.permute(0, 2, 1))
        v3 = x1.detach()
        v4 = torch.sum(v3, dim=-1, keepdim=True)
        v2 = x1.permute(0, 2, 1)
        v3 = torch.sum(v3)
        return torch.nn.functional.linear(v2, torch.where(v3 > 0, v4 * v5, v3), self.linear.bias)
x = torch.randn(1, 2, 2)
