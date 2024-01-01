
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)
        self.norm = torch.nn.GroupNorm(1, 2)
    def forward(self, x1, x2):
        v1 = self.norm(x1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v3 = v2.permute(0, 2, 1)
        v4 = self.norm(x2)
        v5 = torch.nn.functional.linear(v4, self.linear.weight, self.linear.bias)
        v6 = v5.permute(0, 2, 1)
        v7 = torch.add(v3, v6)
        return (v3, v3)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
