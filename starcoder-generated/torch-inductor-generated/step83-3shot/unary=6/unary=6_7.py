
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
        self.relu6 = torch.nn.ReLU6()
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp(v1, min=0, max=6)
        v3 = v2.mul(6)
        v4 = v3.div(6)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
