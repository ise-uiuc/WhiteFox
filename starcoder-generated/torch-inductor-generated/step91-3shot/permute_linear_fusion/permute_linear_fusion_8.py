
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        i1 = torch.sum(x1, dim=-1)
        v1 = x1.permute(0, 2, 1)
        i2 = torch.sum(v1, dim=-1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        i3 = torch.sum(v2, dim=-1)
        v3 = i1 + i2 + i3
        return v3
# Inputs to the model
x1 = torch.randn(1, 2, 2)
