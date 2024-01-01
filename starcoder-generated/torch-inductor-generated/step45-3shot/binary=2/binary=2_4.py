
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 5, stride=1, padding=2, dilation=1, groups=1, bias=False)
    def forward(self, t1):
        t5 = self.conv(t1)
        v2 = t5 - t1
        return v2
# Inputs to the model
t1 = torch.randn(1, 3, 64, 64)
