
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu6 = torch.nn.ReLU6()
    def forward(self, x1):
        v1 = torch.add(x1, x1)
        v2 = v1 + 3
        v3 = torch.clamp_min(v2, 0)
        v4 = torch.clamp_max(v3, 6)
        v5 = v1 * v4
        v6 = v5 / 6
        v7 = self.relu6(v6)
        return v7
# Inputs to the model
x_1 = torch.randn(1, 3, 256, 256)
