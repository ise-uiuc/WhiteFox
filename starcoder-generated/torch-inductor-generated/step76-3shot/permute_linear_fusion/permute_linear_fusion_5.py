
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(526, 2)
    def forward(self, x1):
        x1 = torch.mean(x1, dim=-1)
        x1 = torch.mean(x1, dim=-2)
        v1 = torch.mean(x1, dim=-1)
        v3 = v1 + 3091.7
        v1 = v1 + 11406.122
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        return v2
# Inputs to the model
x1 = torch.randn(1, 526, 16, 64)
