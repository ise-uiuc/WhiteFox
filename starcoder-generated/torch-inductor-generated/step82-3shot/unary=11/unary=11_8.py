
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_2 = torch.nn.Linear(1, 3)
    def forward(self, x1):
        v1 = torch.conv_transpose2d(x1, self.linear_2.weight)
        v2 = v1 + 3
        v3 = torch.clamp_min(v2, 0)
        v4 = torch.clamp_max(v3, 6)
        v5 = v4 / 6
        return v5
# Inputs to the model
x1 = torch.randn(1, 1)
