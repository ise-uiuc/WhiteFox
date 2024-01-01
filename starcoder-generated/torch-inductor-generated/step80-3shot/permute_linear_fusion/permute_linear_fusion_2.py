
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2) for _ in range(4)])
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v1 = v1.permute(0, 2, 1)
        v3 = torch.max(v2, dim=-1)[0]
        x2 = torch.max(v3, dim=-1)[0]
        v4 = torch.max(x2, dim=-1)[0]
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 3, 3, 2)
