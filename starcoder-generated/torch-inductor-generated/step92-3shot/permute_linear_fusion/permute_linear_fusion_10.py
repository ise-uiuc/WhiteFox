
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.norm = torch.nn.LayerNorm((4, 2))
    def forward(self, x):
        v1 = x.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v3 = x + v2
        v4 = v3.permute(0, 2, 1)
        v5 = self.norm(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 2, 2)
