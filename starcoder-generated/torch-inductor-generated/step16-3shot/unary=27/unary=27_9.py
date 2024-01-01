
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 64, 3, stride=3, padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp_max(v1, 1.4)
        v3 = torch.clamp_min(v2, -0.2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 128, 128)
