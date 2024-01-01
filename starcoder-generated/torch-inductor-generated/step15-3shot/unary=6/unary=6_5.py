
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1920648_1 = torch.nn.Conv2d(3, 16, 3, stride=2, padding=1)
        self.conv161920648_2 = torch.nn.Conv2d(16, 8, 2, stride=2, padding=0)
    def forward(self, x1):
        v1 = self.conv1920648_1(x1)
        v2 = self.conv161920648_2(v1)
        v3 = torch.clamp_min(v2, 0)
        v4 = torch.clamp_max(v3, 6)
        v5 = v2 * v4
        v6 = v5 / 6
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 928, 512)
