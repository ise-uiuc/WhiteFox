
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU(inplace=False)
        self.conv2d = torch.nn.Conv2d(32, 15, 3, padding=1)
    def forward(self, x1):
        v1 = self.conv2d(x1)
        v2 = self.relu(v1)
        v3 = v2 + 3
        v4 = torch.clamp(v3, min=0)
        v5 = torch.clamp(v4, max=6)
        v6 = v2 * v5
        v7 = v6 + v5
        return v7
# Inputs to the model
x1 = torch.randn(1, 32, 36, 36)
