
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.relu6 = torch.nn.modules.activation.ReLU6()
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1.add(3)
        v3 = v2.add(0)
        v4 = v3.add(-3)
        v5 = v4.clamp_min(0)
        v6 = v5.clamp_max(6)
        v7 = v6.sub(6)
        v8 = v7.sub(-6)
        v9 = v8.div(6)
        v10 = v9.mul(1)
        return self.relu6(v10)
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
