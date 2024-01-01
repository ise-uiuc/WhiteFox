
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(18, 8, (1, 3), stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(8, 12, (3, 1), stride=1, padding=0)
    def forward(self, x1):
        h1 = 3 - x1
        v1 = self.conv1(h1)
        v2 = self.conv2(v1)
        v3 = 7.7 - v2
        v4 = torch.clamp_max(v3, 6)
        v5 = v1 + v4
        v6 = 6 + v5
        v7 = torch.clamp(v6, min=0)
        return v7
# Inputs to the model
x = torch.randn(1, 18, 512, 512)
