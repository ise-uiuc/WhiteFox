
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
    def forward(self, x1):
        u1 = self.conv(x1)
        u2 = u1 + 0
        u3 = u1 + 3
        u4 = torch.clamp_min(u2, 0)
        u5 = torch.clamp_min(u3, 0)
        u6 = torch.clamp_min(u4, 0)
        u7 = torch.clamp_min(u5, 0)
        u8 = torch.clamp_min(u7, 0)
        return u8
# Inputs to the model
x1 = torch.randn(2, 3, 64, 64)
