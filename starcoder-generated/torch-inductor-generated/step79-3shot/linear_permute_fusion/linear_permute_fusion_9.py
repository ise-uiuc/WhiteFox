
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x2):
        v1 = torch.nn.functional.linear(x2, self.linear.weight, self.linear.bias)
        v2 = v1.permute(0, 2, 1)
        v3 = torch.cat([v2, v2, v2], dim=0)
        v4 = torch.stack([v3, v3, v3])
        return v4
# Inputs to the model
x2 = torch.randn(2, 2, 2)
