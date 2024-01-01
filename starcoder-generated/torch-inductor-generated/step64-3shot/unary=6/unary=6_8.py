
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, kernel_size=2, stride=1, padding=1, dilation=2)
    def forward(self, x1):
        o1 = self.conv(x1)
        o2 = o1.add(3)
        o3 = o2.clamp_min(0)
        o4 = o3.clamp_max(6)
        o5 = o1.mul(o4)
        o6 = o5.div(6)
        o7 = o1 + o6
        o7 = o7.add(6)
        o8 = o7.mul(o7)
        o9 = o8.clamp_min(0)
        o10 = o9.clamp_max(6)
        o11 = o8.mul(o10)
        o12 = o11.div(36)
        return o12
# Inputs to the model
x1 = torch.randn(1, 3, 128, 128)
