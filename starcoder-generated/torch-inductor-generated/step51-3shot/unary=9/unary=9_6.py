
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 8, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 + 3
        v3 = v2.clamp(0, 6)
        v4 = v3.div(6)
        v5 = self.conv2(v4)
        v6 = v5 + 3
        v7 = v6.clamp(0, 6)
        v8 = v7.div(6)
        return v8
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
