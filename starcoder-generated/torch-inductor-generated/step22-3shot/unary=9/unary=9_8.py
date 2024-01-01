
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.other_conv = torch.nn.Conv2d(8, 8, 9)
    def forward(self, x1):
        v0 = x1.detach()
        v1 = self.conv(x1)
        v2 = 3 + v1
        v3 = v2.clamp(min=0, max=6)
        v4 = torch.div(v3, 6)
        v5 = self.conv(v4)
        v6 = 3 + v5
        v7 = v6.clamp(min=0, max=6)
        v8 = torch.div(v7, 6)
        return v8
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
