
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        v1 = self.relu(x)
        v1 = v1 * negative_slope
        # Comment below line
        # negative_slope = 0.0001
        v2 = v1 ** 0.000416
        v3 = F.relu(v2)
        v4 = torch.where(y, v1, v2)
        return v4
# Inputs to the model
x1 = torch.randn(1, 72844)
