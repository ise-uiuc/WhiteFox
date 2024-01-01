
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.relu6 = torch.nn.ReLU6()
    def forward(self, x1):
        v1 = self.conv(x1)
        a1 = 3
        v2 = v1.add(a1)
        v3 = v2.clamp(min=0, max=6)
        v4 = v3.div(6)
        return self.relu6(v4)
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
