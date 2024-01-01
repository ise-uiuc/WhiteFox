
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 - 0.0125
        v3 = (v2 - 2.123130) * 2.123130
        v4 = (v3 - 3.141592) + 3.141592
        return v4
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
