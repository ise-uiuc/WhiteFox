
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.other_conv = torch.nn.Conv2d(8, 8, 9)
    def forward(self, x1):
        v1 = self.conv.forward(x1)
        v2 = self.other_conv.forward(v1.add(3).clamp_min(0).clamp_max(6).div(6))
        v3 = self.other_conv.forward(v2.add(3).clamp_min(0).clamp_max(6).div(6))
        v4 = 3 + v3
        v5 = self.other_conv.forward(v4.clamp_min(0).clamp_max(6).div(6))
        v6 = v5 + 3
        v7 = self.other_conv.forward(v6.clamp_min(0).clamp_max(6).div(6))
        v8 = 3 + v7
        return v8
# Inputs to the model
x1 = torch.randn(5, 3, 64, 64)
