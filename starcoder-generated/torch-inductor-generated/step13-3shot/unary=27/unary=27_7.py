
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 1, 1)
        self.conv2 = torch.nn.Conv2d(1, 4, 1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.clamp_min(v1, min=0.45)
        v3 = torch.clamp_max(v2, min=0.26)
        v4 = self.conv2(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
