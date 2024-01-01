
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.other = torch.nn.Conv2d(8, 5, 1, stride=(1, 2), padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1.add(3)
        v3 = v2.clamp(min=0, max=6)
        v4 = v3.div(6)
        v5 = self.other(v4)
        v6 = v5.add(3)
        v7 = v6.clamp(min=0, max=6)
        v8 = v7.div(6)
        return v8
# Inputs to the model
x1 = torch.randn(5, 3, 64, 64)
